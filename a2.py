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
from collections import defaultdict
import difflib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_company_extractor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedCompanyExtractor")

# Constants
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}
TIMEOUT = 10  # seconds
LOGO_KEYWORDS = ['logo', 'brand', 'header-logo', 'site-logo', 'navbar-logo']

# TLDs to remove when extracting company names
COMMON_TLDS = ['com', 'org', 'net', 'io', 'co', 'info', 'biz', 'eu', 'xyz', 'online']

def extract_domain(url):
    """Extract base domain from URL."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc or parsed_url.path.split('/')[0]
    return domain.replace('www.', '')

def prepare_url(url):
    """Ensure the URL has a proper scheme."""
    if not url.startswith(('http://', 'https://')):
        return f'https://{url}'
    return url

def normalize_string(s):
    """Normalize string for comparison."""
    if not s:
        return ""
    # Convert to lowercase
    s = s.lower()
    # Remove common TLDs
    for tld in COMMON_TLDS:
        if s.endswith('.' + tld):
            s = s[:-len(tld)-1]
    # Remove non-alphanumeric characters
    return re.sub(r'[^a-z0-9]', '', s)

def extract_company_names(domain):
    """Extract potential company names from domain, returning multiple candidates."""
    if not domain:
        return []
    
    candidates = []
    
    # 1. First, split by common domain separators for the primary name
    parts = re.split(r'[-_.:()]', domain)
    primary_name = parts[0] if parts else domain
    
    # Remove TLD if present
    domain_parts = domain.split('.')
    if len(domain_parts) > 1:
        # Add the base domain without TLD
        base_domain = domain_parts[0]
        candidates.append(normalize_string(base_domain))
    
    # 2. Add the primary name (before first separator)
    candidates.append(normalize_string(primary_name))
    
    # 3. For domains without separators, try to extract potential company names
    if len(parts) <= 1 and len(domain_parts[0]) > 4:
        # Try common suffixes to identify if this is a pattern like "companyabc"
        common_suffixes = ['inc', 'llc', 'ltd', 'corp', 'co', 'grp', 'group', 'tech', 'abc', 'xyz']
        for suffix in common_suffixes:
            if domain_parts[0].endswith(suffix) and len(domain_parts[0]) > len(suffix) + 2:
                potential_name = domain_parts[0][:-len(suffix)]
                candidates.append(normalize_string(potential_name))
    
    # 4. For longer domains, add substrings as candidates
    if len(domain_parts[0]) > 8:
        # Add first 5, 6, 7, 8 characters as potential prefixes
        for length in range(5, 9):
            if len(domain_parts[0]) > length + 2:  # Ensure there's some suffix too
                candidates.append(normalize_string(domain_parts[0][:length]))
    
    # Remove duplicates while preserving order
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen and len(candidate) > 2:
            seen.add(candidate)
            unique_candidates.append(candidate)
    
    return unique_candidates

def find_similar_domains(domains, similarity_threshold=0.7):
    """Group domains that likely belong to the same company."""
    # First, get all candidate company names for each domain
    domain_candidates = {domain: extract_company_names(domain) for domain in domains}
    
    # Create company groups with domains that share similar candidates
    company_groups = defaultdict(list)
    processed_domains = set()
    
    # Sort domains to ensure consistent results
    sorted_domains = sorted(domains)
    
    for i, domain1 in enumerate(sorted_domains):
        if domain1 in processed_domains:
            continue
            
        # Start a new group with this domain
        current_group = [domain1]
        processed_domains.add(domain1)
        
        candidates1 = domain_candidates[domain1]
        if not candidates1:
            continue
            
        # Find similar domains
        for domain2 in sorted_domains[i+1:]:
            if domain2 in processed_domains:
                continue
                
            candidates2 = domain_candidates[domain2]
            if not candidates2:
                continue
            
            # Check for shared candidate names
            is_similar = False
            
            # Check direct matches
            for candidate1 in candidates1:
                for candidate2 in candidates2:
                    # Check if one is contained within the other
                    if candidate1 in candidate2 or candidate2 in candidate1:
                        is_similar = True
                        break
                    
                    # Check similarity ratio
                    similarity = difflib.SequenceMatcher(None, candidate1, candidate2).ratio()
                    if similarity >= similarity_threshold:
                        is_similar = True
                        break
                        
                if is_similar:
                    break
            
            if is_similar:
                current_group.append(domain2)
                processed_domains.add(domain2)
        
        # Find a good company name for this group
        if len(current_group) > 0:
            # Use the most common candidate among the domains
            all_candidates = []
            for d in current_group:
                all_candidates.extend(domain_candidates[d])
            
            # Count occurrences of each candidate
            candidate_counts = {}
            for candidate in all_candidates:
                candidate_counts[candidate] = candidate_counts.get(candidate, 0) + 1
            
            # Sort by count (descending) and then by length (descending)
            best_candidates = sorted(candidate_counts.items(), 
                                 key=lambda x: (-x[1], -len(x[0])))
            
            company_name = best_candidates[0][0] if best_candidates else current_group[0]
            company_groups[company_name] = current_group
    
    # Handle remaining domains as individual companies
    for domain in domains:
        if domain not in processed_domains:
            candidates = domain_candidates[domain]
            company_name = candidates[0] if candidates else domain
            company_groups[company_name].append(domain)
    
    return company_groups

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

def download_image(img_url, base_url):
    """Download image and return the content if successful."""
    try:
        # Make sure the image URL is absolute
        if not img_url.startswith(('http://', 'https://')):
            img_url = urllib.parse.urljoin(base_url, img_url)
            
        response = requests.get(img_url, timeout=TIMEOUT, headers=HEADERS)
        if response.status_code == 200 and is_valid_image(response):
            return response.content
        return None
    except Exception as e:
        logger.debug(f"Error downloading image {img_url}: {e}")
        return None

def save_image(img_data, company_name, domain, img_url, output_dir):
    """Save image data to a file."""
    if not img_data:
        return None
    
    # Generate a filename from the company and domain
    url_hash = hashlib.md5(img_url.encode()).hexdigest()[:10]
    ext = os.path.splitext(img_url)[1]
    if not ext or len(ext) > 5:  # If extension is missing or suspicious
        ext = '.png'  # Default to PNG
    
    filename = f"{company_name}_{domain}_{url_hash}{ext}"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'wb') as f:
        f.write(img_data)
    
    logger.info(f"Saved logo to {filepath}")
    return filepath

def find_logo_in_html(soup, base_url):
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
            
    # 4. Check for favicon as fallback
    favicon_links = soup.find_all('link', rel=lambda x: x and ('icon' in x.lower() or 'shortcut icon' in x.lower()))
    if favicon_links:
        favicon_href = favicon_links[0].get('href', '')
        if favicon_href:
            potential_logos.append(favicon_href)
    
    # 5. Try default favicon location as absolute last resort
    potential_logos.append(urllib.parse.urljoin(base_url, '/favicon.ico'))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_logos = []
    for logo in potential_logos:
        if logo not in seen:
            seen.add(logo)
            unique_logos.append(logo)
            
    return unique_logos

def extract_logo_from_website(url, company_name, output_dir):
    """Extract logo from a specific website."""
    domain = extract_domain(url)
    full_url = prepare_url(url)
    
    logger.info(f"Processing {full_url} (Company: {company_name}, Domain: {domain})")
    
    try:
        response = requests.get(full_url, timeout=TIMEOUT, headers=HEADERS)
        if response.status_code != 200:
            logger.warning(f"Failed to get {full_url}: Status code {response.status_code}")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        potential_logos = find_logo_in_html(soup, full_url)
        
        if not potential_logos:
            logger.info(f"No logos found for {domain}")
            return None
        
        # Process only the first valid logo
        for logo in potential_logos:
            if isinstance(logo, tuple) and logo[0] == 'svg':
                # Handle SVG content
                svg_content = logo[1]
                filepath = os.path.join(output_dir, f"{company_name}_{domain}_svg.svg")
                with open(filepath, 'w') as f:
                    f.write(svg_content)
                
                return {
                    'company': company_name,
                    'domain': domain,
                    'url': full_url,
                    'logo_url': 'inline_svg',
                    'filepath': filepath,
                    'success': True
                }
            else:
                # Handle image URLs
                img_data = download_image(logo, full_url)
                if img_data:
                    filepath = save_image(img_data, company_name, domain, logo, output_dir)
                    if filepath:
                        return {
                            'company': company_name,
                            'domain': domain,
                            'url': full_url,
                            'logo_url': logo,
                            'filepath': filepath,
                            'success': True
                        }
        
        # If we get here, no valid logo was found
        return None
            
    except Exception as e:
        logger.error(f"Error processing {full_url}: {e}")
        return None

def extract_company_logos(domains_by_company, output_dir, max_workers=10):
    """Extract one logo per company."""
    os.makedirs(output_dir, exist_ok=True)
    
    company_logos = {}
    results = []
    
    # Use a thread pool to process companies in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        company_futures = {}
        
        # Submit one job per company to extract its logo
        for company, domains in domains_by_company.items():
            # Create a future for this company
            future = executor.submit(extract_logo_for_company, company, domains, output_dir)
            company_futures[future] = company
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(company_futures):
            company = company_futures[future]
            try:
                result = future.result()
                if result:
                    company_logos[company] = result
                    results.append(result)
                    logger.info(f"Successfully extracted logo for company: {company}")
                else:
                    logger.warning(f"Failed to extract logo for company: {company}")
            except Exception as e:
                logger.error(f"Error extracting logo for company {company}: {e}")
    
    # Save results
    save_results(results, output_dir)
    
    return results

def extract_logo_for_company(company, domains, output_dir):
    """Try to extract a logo from any of the company's domains."""
    # Try each domain until we find a valid logo
    for domain in domains:
        logo_result = extract_logo_from_website(domain, company, output_dir)
        if logo_result:
            return logo_result
    
    # If all domains failed, return None
    return None

def save_results(results, output_dir):
    """Save extraction results to CSV."""
    results_file = os.path.join(output_dir, "company_logos_results.csv")
    
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['company', 'domain', 'url', 'logo_url', 'filepath', 'success']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    logger.info(f"Saved results to {results_file}")
    
    # Print summary
    print("\nCompany Logo Extraction Summary:")
    print("================================")
    print(f"Total companies processed: {len(set(r['company'] for r in results))}")
    print(f"Total logos extracted: {len(results)}")
    print(f"Results saved to {results_file}")

def read_websites_from_parquet(parquet_path):
    """Read website URLs from a Parquet file."""
    try:
        logger.info(f"Reading websites from Parquet file: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        
        # Log the DataFrame structure
        logger.info(f"Parquet file columns: {df.columns.tolist()}")
        logger.info(f"Number of records: {len(df)}")
        
        # Determine which column contains website URLs
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
    parser = argparse.ArgumentParser(description='Extract one logo per company from websites')
    parser.add_argument('--parquet', '-p', type=str, default='logos.snappy.parquet',
                      help='Path to the Parquet file containing website URLs')
    parser.add_argument('--output-dir', '-o', type=str, default='company_logos',
                      help='Directory to save extracted logos')
    parser.add_argument('--workers', '-w', type=int, default=10,
                      help='Number of worker threads')
    parser.add_argument('--similarity', '-s', type=float, default=0.7,
                      help='Similarity threshold for company name matching (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Read websites from Parquet file
    websites = read_websites_from_parquet(args.parquet)
    
    # Group domains by company using enhanced algorithm
    domains_by_company = find_similar_domains(websites, args.similarity)
    
    logger.info(f"Identified {len(domains_by_company)} unique companies")
    
    # Log some example companies and their domains
    for i, (company, domains) in enumerate(domains_by_company.items()):
        if i < 5:  # Limit to first 5 for brevity
            logger.info(f"Company: {company}, Domains: {domains[:3]}{'...' if len(domains) > 3 else ''}")
    
    # Extract one logo per company
    start_time = time.time()
    results = extract_company_logos(domains_by_company, args.output_dir, args.workers)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Extracted {len(results)} company logos in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()