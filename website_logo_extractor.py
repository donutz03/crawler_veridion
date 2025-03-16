import os
import time
import requests
import urllib.parse
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import concurrent.futures
import hashlib
import re
import logging
from PIL import Image
from io import BytesIO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logo_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LogoExtractor")

# Create output directory
OUTPUT_DIR = "extracted_logos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set timeout for requests to avoid hanging on slow websites
TIMEOUT = 10  # seconds

# Headers to mimic a browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

def prepare_url(url):
    """Ensure the URL has a proper scheme."""
    if not url.startswith(('http://', 'https://')):
        return f'https://{url}'
    return url

def extract_domain(url):
    """Extract base domain from URL."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc or parsed_url.path.split('/')[0]
    return domain.replace('www.', '')

def is_likely_logo(img_url, img_tag):
    """Heuristic to determine if an image is likely a logo."""
    img_url_lower = img_url.lower()
    
    # Check URL patterns
    if any(keyword in img_url_lower for keyword in ['logo', 'brand', 'header', 'site-icon']):
        return True
    
    # Check tag attributes
    for attr, value in img_tag.attrs.items():
        if attr in ['class', 'id', 'alt', 'title'] and isinstance(value, str):
            value_lower = value.lower()
            if any(keyword in value_lower for keyword in ['logo', 'brand', 'header']):
                return True
        elif attr in ['class', 'id'] and isinstance(value, list):
            for v in value:
                if any(keyword in v.lower() for keyword in ['logo', 'brand', 'header']):
                    return True
    
    return False

def find_favicon(soup, base_url):
    """Find favicon link in the HTML."""
    favicon_links = soup.find_all('link', rel=lambda x: x and ('icon' in x.lower() or 'shortcut icon' in x.lower()))
    
    if favicon_links:
        favicon_href = favicon_links[0].get('href', '')
        if favicon_href:
            if not favicon_href.startswith(('http://', 'https://')):
                # Handle relative URLs
                return urllib.parse.urljoin(base_url, favicon_href)
            return favicon_href
    
    # Try the default favicon location
    domain = urlparse(base_url).netloc
    return f"{base_url}/favicon.ico"

def extract_css_background_images(soup, base_url):
    """Extract background images from inline CSS."""
    images = []
    
    # Look for inline style attributes
    for tag in soup.find_all(style=True):
        style = tag['style']
        background_urls = re.findall(r'background-image\s*:\s*url\([\'"]?([^\'"]+)[\'"]?\)', style)
        for url in background_urls:
            full_url = url if url.startswith(('http://', 'https://')) else urllib.parse.urljoin(base_url, url)
            images.append((full_url, tag))
    
    return images

def is_valid_image(response):
    """Check if the response contains a valid image."""
    try:
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            return False
        
        # Try to open with PIL to validate it's a proper image
        img = Image.open(BytesIO(response.content))
        img.verify()  # Verify it's a valid image
        
        # Check image size - very small images are unlikely to be logos
        if img.width < 16 or img.height < 16:
            return False
            
        return True
    except Exception as e:
        logger.debug(f"Invalid image: {e}")
        return False

def download_image(img_url, timeout=TIMEOUT):
    """Download image and return the content if successful."""
    try:
        response = requests.get(img_url, timeout=timeout, headers=HEADERS)
        if response.status_code == 200 and is_valid_image(response):
            return response.content
        return None
    except Exception as e:
        logger.debug(f"Error downloading image {img_url}: {e}")
        return None

def save_image(img_data, domain, img_url, index):
    """Save image data to a file."""
    if not img_data:
        return None
    
    # Generate a filename from the domain and a hash of the URL
    url_hash = hashlib.md5(img_url.encode()).hexdigest()[:10]
    ext = os.path.splitext(img_url)[1]
    if not ext or len(ext) > 5:  # If extension is missing or suspicious
        ext = '.png'  # Default to PNG
    
    filename = f"{domain}_{index}_{url_hash}{ext}"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    with open(filepath, 'wb') as f:
        f.write(img_data)
    
    logger.info(f"Saved logo to {filepath}")
    return filepath

def extract_logos_from_website(url, index=0):
    """Extract potential logo images from a website."""
    domain = extract_domain(url)
    full_url = prepare_url(url)
    
    logger.info(f"[{index}] Processing {full_url} ({domain})")
    
    try:
        response = requests.get(full_url, timeout=TIMEOUT, headers=HEADERS)
        if response.status_code != 200:
            logger.warning(f"Failed to get {full_url}: Status code {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        potential_logos = []
        
        # 1. Check for images with "logo" in attributes
        for img in soup.find_all('img'):
            img_src = img.get('src', '')
            if not img_src:
                continue
                
            img_url = img_src if img_src.startswith(('http://', 'https://')) else urllib.parse.urljoin(full_url, img_src)
            
            if is_likely_logo(img_url, img):
                potential_logos.append(img_url)
        
        # 2. Check CSS background images
        bg_images = extract_css_background_images(soup, full_url)
        for img_url, tag in bg_images:
            if 'logo' in img_url.lower() or 'logo' in str(tag).lower():
                potential_logos.append(img_url)
        
        # 3. Always include favicon as fallback
        favicon_url = find_favicon(soup, full_url)
        if favicon_url:
            potential_logos.append(favicon_url)
            
        # Remove duplicates
        potential_logos = list(dict.fromkeys(potential_logos))
        
        # Download and save logos
        saved_logos = []
        for i, logo_url in enumerate(potential_logos[:3]):  # Limit to first 3 potential logos
            img_data = download_image(logo_url)
            if img_data:
                filepath = save_image(img_data, domain, logo_url, i)
                if filepath:
                    saved_logos.append({
                        'domain': domain,
                        'url': full_url,
                        'logo_url': logo_url,
                        'filepath': filepath
                    })
        
        return saved_logos
        
    except Exception as e:
        logger.error(f"Error processing {full_url}: {e}")
        return []

def process_websites(websites, max_workers=10, limit=200):
    """Process multiple websites concurrently."""
    websites = websites[:limit]  # Limit to first 200
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(extract_logos_from_website, url, i): url 
                        for i, url in enumerate(websites)}
        
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                logos = future.result()
                if logos:
                    results.extend(logos)
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
    
    return results

def read_website_list(file_path):
    """Read website URLs from a file."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    try:
        # Try to read from a file first
        websites = read_website_list('logos_list')
        logger.info(f"Read {len(websites)} websites from logos_list file")
    except:
        # Fallback to a sample list
        logger.info("logos_list file not found, using sample websites")
        websites = [
            "google.com",
            "facebook.com",
            "amazon.com",
            "apple.com",
            "microsoft.com",
            # Add more websites to make up 200...
        ]
    
    start_time = time.time()
    results = process_websites(websites, max_workers=10, limit=200)
    end_time = time.time()
    
    logger.info(f"Processed {len(websites)} websites in {end_time - start_time:.2f} seconds")
    logger.info(f"Successfully extracted {len(results)} logos")
    
    # Save results to a CSV file
    with open('extracted_logos_info.csv', 'w') as f:
        f.write('domain,url,logo_url,filepath\n')
        for logo in results:
            f.write(f"{logo['domain']},{logo['url']},{logo['logo_url']},{logo['filepath']}\n")
    
    logger.info("Results saved to extracted_logos_info.csv")

if __name__ == "__main__":
    main()