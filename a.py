import os
import re
import time
import urllib.parse
import hashlib
import pandas as pd
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from PIL import Image
from io import BytesIO
import logging
import argparse
import csv
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logo_extractor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LogoExtractor")

# Constants
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}
TIMEOUT = 10  # seconds
LOGO_KEYWORDS = ['logo', 'brand', 'header-logo', 'site-logo', 'navbar-logo']

class LogoExtractor:
    def __init__(self, output_dir="extracted_logos", stats_file="logo_stats.csv"):
        self.output_dir = output_dir
        self.stats_file = stats_file
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize stats
        self.stats = {
            'total_websites': 0,
            'websites_with_logo': 0,
            'websites_without_logo': 0,
            'failed_connections': 0
        }
        
    def extract_domain(self, url):
        """Extract base domain from URL."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc or parsed_url.path.split('/')[0]
        return domain.replace('www.', '')
    
    def prepare_url(self, url):
        """Ensure the URL has a proper scheme."""
        if not url.startswith(('http://', 'https://')):
            return f'https://{url}'
        return url
    
    def is_valid_image(self, response):
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
    
    def download_image(self, img_url, base_url):
        """Download image and return the content if successful."""
        try:
            # Make sure the image URL is absolute
            if not img_url.startswith(('http://', 'https://')):
                img_url = urllib.parse.urljoin(base_url, img_url)
                
            response = requests.get(img_url, timeout=TIMEOUT, headers=HEADERS)
            if response.status_code == 200 and self.is_valid_image(response):
                return response.content
            return None
        except Exception as e:
            logger.debug(f"Error downloading image {img_url}: {e}")
            return None
    
    def save_image(self, img_data, domain, img_url):
        """Save image data to a file."""
        if not img_data:
            return None
        
        # Generate a filename from the domain and a hash of the URL
        url_hash = hashlib.md5(img_url.encode()).hexdigest()[:10]
        ext = os.path.splitext(img_url)[1]
        if not ext or len(ext) > 5:  # If extension is missing or suspicious
            ext = '.png'  # Default to PNG
        
        filename = f"{domain}_{url_hash}{ext}"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'wb') as f:
            f.write(img_data)
        
        logger.info(f"Saved logo to {filepath}")
        return filepath
    
    def find_logo_in_html(self, soup, base_url):
        """Find logo elements in HTML."""
        potential_logos = []
        
        # 1. Look for img tags with "logo" in attributes
        for img in soup.find_all('img'):
            img_src = img.get('src', '')
            if not img_src:
                continue
                
            # Check if any logo keywords are in attributes
            attrs_text = ' '.join([f"{k}={v}" for k, v in img.attrs.items() if isinstance(v, str)])
            if any(keyword in attrs_text.lower() for keyword in LOGO_KEYWORDS):
                potential_logos.append(img_src)
                
        # 2. Look for div/span elements with "logo" in class or id
        logo_elements = []
        for keyword in LOGO_KEYWORDS:
            # Find by class
            logo_elements.extend(soup.find_all(class_=re.compile(keyword, re.I)))
            # Find by id
            logo_elements.extend(soup.find_all(id=re.compile(keyword, re.I)))
            # Find by aria-label
            logo_elements.extend(soup.find_all(attrs={"aria-label": re.compile(keyword, re.I)}))
        
        # Check for background images in these logo elements
        for element in logo_elements:
            # Check for inline style with background-image
            style = element.get('style', '')
            bg_match = re.search(r'background-image\s*:\s*url\([\'"]?([^\'"]+)[\'"]?\)', style)
            if bg_match:
                potential_logos.append(bg_match.group(1))
            
            # Check for img children
            for img in element.find_all('img'):
                img_src = img.get('src', '')
                if img_src and img_src not in potential_logos:
                    potential_logos.append(img_src)
                    
            # Check for SVG children
            svg = element.find('svg')
            if svg:
                # We found an inline SVG logo - save the HTML
                potential_logos.append(('svg', str(svg)))
        
        # 3. Look for SVG elements with "logo" in attrs
        for svg in soup.find_all('svg'):
            attrs_text = ' '.join([f"{k}={v}" for k, v in svg.attrs.items() if isinstance(v, str)])
            if any(keyword in attrs_text.lower() for keyword in LOGO_KEYWORDS):
                potential_logos.append(('svg', str(svg)))
                
        # 4. Check CSS for logo references
        # Extract all CSS files
        css_files = []
        for link in soup.find_all('link', rel='stylesheet'):
            href = link.get('href')
            if href:
                css_files.append(urllib.parse.urljoin(base_url, href))
                
        # TODO: Implement if needed - download CSS files and parse for background-image with logo keywords
        
        # 5. Check for favicon as fallback
        favicon_links = soup.find_all('link', rel=lambda x: x and ('icon' in x.lower() or 'shortcut icon' in x.lower()))
        if favicon_links:
            favicon_href = favicon_links[0].get('href', '')
            if favicon_href:
                potential_logos.append(favicon_href)
        
        # 6. Try default favicon location as absolute last resort
        potential_logos.append(urllib.parse.urljoin(base_url, '/favicon.ico'))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_logos = []
        for logo in potential_logos:
            if logo not in seen:
                seen.add(logo)
                unique_logos.append(logo)
                
        return unique_logos
    
    def extract_logo_from_website(self, url, index=0):
        """Extract logo from a specific website."""
        domain = self.extract_domain(url)
        full_url = self.prepare_url(url)
        
        logger.info(f"[{index}] Processing {full_url} ({domain})")
        
        try:
            response = requests.get(full_url, timeout=TIMEOUT, headers=HEADERS)
            if response.status_code != 200:
                logger.warning(f"Failed to get {full_url}: Status code {response.status_code}")
                self.stats['failed_connections'] += 1
                return {
                    'domain': domain,
                    'url': full_url,
                    'success': False,
                    'error': f'HTTP {response.status_code}'
                }
            
            soup = BeautifulSoup(response.text, 'html.parser')
            potential_logos = self.find_logo_in_html(soup, full_url)
            
            if not potential_logos:
                logger.info(f"No logos found for {domain}")
                self.stats['websites_without_logo'] += 1
                return {
                    'domain': domain,
                    'url': full_url,
                    'success': False,
                    'error': 'No logos found'
                }
            
            # Process only the first valid logo
            for logo in potential_logos:
                if isinstance(logo, tuple) and logo[0] == 'svg':
                    # Handle SVG content
                    svg_content = logo[1]
                    filepath = os.path.join(self.output_dir, f"{domain}_svg.svg")
                    with open(filepath, 'w') as f:
                        f.write(svg_content)
                    
                    self.stats['websites_with_logo'] += 1
                    return {
                        'domain': domain,
                        'url': full_url,
                        'logo_url': 'inline_svg',
                        'filepath': filepath,
                        'success': True,
                        'error': None
                    }
                else:
                    # Handle image URLs
                    img_data = self.download_image(logo, full_url)
                    if img_data:
                        filepath = self.save_image(img_data, domain, logo)
                        if filepath:
                            self.stats['websites_with_logo'] += 1
                            return {
                                'domain': domain,
                                'url': full_url,
                                'logo_url': logo,
                                'filepath': filepath,
                                'success': True,
                                'error': None
                            }
            
            # If we get here, no valid logo was found
            self.stats['websites_without_logo'] += 1
            return {
                'domain': domain,
                'url': full_url,
                'success': False,
                'error': 'No valid logos found'
            }
                
        except Exception as e:
            logger.error(f"Error processing {full_url}: {e}")
            self.stats['failed_connections'] += 1
            return {
                'domain': domain,
                'url': full_url,
                'success': False,
                'error': str(e)
            }
    
    def process_websites(self, websites, max_workers=10):
        """Process multiple websites concurrently."""
        self.stats['total_websites'] = len(websites)
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self.extract_logo_from_website, url, i): url 
                            for i, url in enumerate(websites)}
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
        
        return results
    
    def save_results(self, results):
        """Save results to CSV file."""
        with open('extracted_logos_results.csv', 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['domain', 'url', 'logo_url', 'filepath', 'success', 'error']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        # Save stats
        with open(self.stats_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in self.stats.items():
                writer.writerow([key, value])
                
        logger.info(f"Results saved to extracted_logos_results.csv")
        logger.info(f"Stats saved to {self.stats_file}")
        
        # Print summary
        print("\nLogo Extraction Summary:")
        print("=======================")
        print(f"Total websites processed: {self.stats['total_websites']}")
        print(f"Websites with logos: {self.stats['websites_with_logo']} ({self.stats['websites_with_logo']/self.stats['total_websites']*100:.1f}%)")
        print(f"Websites without logos: {self.stats['websites_without_logo']} ({self.stats['websites_without_logo']/self.stats['total_websites']*100:.1f}%)")
        print(f"Failed connections: {self.stats['failed_connections']} ({self.stats['failed_connections']/self.stats['total_websites']*100:.1f}%)")
    
def read_websites_from_parquet(parquet_path):
    """Read website URLs from a Parquet file."""
    try:
        logger.info(f"Reading websites from Parquet file: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        
        # Log the DataFrame structure
        logger.info(f"Parquet file columns: {df.columns.tolist()}")
        logger.info(f"Number of records: {len(df)}")
        
        # Determine which column contains website URLs
        # Common column names for website URLs
        url_column_candidates = ['url', 'website', 'domain', 'site', 'website_url', 'domain_name', 'homepage']
        
        # Find the first matching column
        url_column = None
        for candidate in url_column_candidates:
            if candidate in df.columns:
                url_column = candidate
                break
                
        # If no match in common names, assume first string column might contain URLs
        if not url_column:
            for col in df.columns:
                if df[col].dtype == 'object' and isinstance(df[col].iloc[0], str):
                    if any(('.' in str(url) for url in df[col].head(10))):  # Simple heuristic for URLs
                        url_column = col
                        break
        
        if not url_column:
            # If we still can't find it, just use the first column
            url_column = df.columns[0]
            logger.warning(f"Couldn't identify URL column, using first column: {url_column}")
        else:
            logger.info(f"Using column '{url_column}' for website URLs")
        
        # Extract URLs
        websites = df[url_column].tolist()
        
        # Clean URLs (remove protocols if present, etc.)
        cleaned_websites = []
        for site in websites:
            # Ensure we have a string
            site = str(site).strip()
            
            # Skip empty strings
            if not site:
                continue
                
            # Remove protocol if present
            if site.startswith(('http://', 'https://')):
                site = site.split('://', 1)[1]
                
            # Remove trailing slash if present
            if site.endswith('/'):
                site = site[:-1]
                
            cleaned_websites.append(site)
        
        logger.info(f"Extracted {len(cleaned_websites)} website URLs")
        
        # Log a sample of websites
        logger.info(f"Sample websites: {cleaned_websites[:5]}")
        
        # Remove duplicates if any
        cleaned_websites = list(dict.fromkeys(cleaned_websites))
        logger.info(f"Final count after removing duplicates: {len(cleaned_websites)} websites")
        
        return cleaned_websites
        
    except Exception as e:
        logger.error(f"Error reading Parquet file: {e}")
        # Return an empty list in case of error
        return []

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract logos from websites')
    parser.add_argument('--parquet', '-p', type=str, default='logos.snappy.parquet',
                        help='Path to the Parquet file containing website URLs')
    parser.add_argument('--output-dir', '-o', type=str, default='extracted_logos',
                        help='Directory to save extracted logos')
    parser.add_argument('--workers', '-w', type=int, default=10,
                        help='Number of worker threads')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit the number of websites to process')
    
    args = parser.parse_args()
    
    # Read websites from Parquet file
    websites = read_websites_from_parquet(args.parquet)
    
    # Limit websites if specified
    if args.limit:
        websites = websites[:args.limit]
    
    # Initialize logo extractor
    extractor = LogoExtractor(output_dir=args.output_dir)
    
    # Process websites
    start_time = time.time()
    results = extractor.process_websites(websites, max_workers=args.workers)
    
    # Save results
    extractor.save_results(results)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Processed {len(websites)} websites in {elapsed_time:.2f} seconds")
    
if __name__ == "__main__":
    main()