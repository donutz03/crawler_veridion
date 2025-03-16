import requests
import time
import concurrent.futures
import csv
import logging
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("website_timing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WebsiteTiming")

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

def measure_load_time(url, timeout=10):
    """Measure the load time of a website."""
    domain = extract_domain(url)
    full_url = prepare_url(url)
    
    try:
        start_time = time.time()
        response = requests.get(full_url, timeout=timeout, headers=HEADERS)
        end_time = time.time()
        
        load_time = end_time - start_time
        status_code = response.status_code
        
        return {
            'domain': domain,
            'url': full_url,
            'load_time': load_time,
            'status_code': status_code,
            'success': status_code == 200
        }
    except requests.exceptions.Timeout:
        return {
            'domain': domain,
            'url': full_url,
            'load_time': timeout,
            'status_code': None,
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        return {
            'domain': domain,
            'url': full_url,
            'load_time': None,
            'status_code': None,
            'success': False,
            'error': str(e)
        }

def process_websites_timing(websites, max_workers=10, timeout=2.0):
    """Process multiple websites concurrently with incremental timeout."""
    all_results = []
    remaining_websites = websites.copy()
    
    # Start with a small timeout and gradually increase
    timeouts = [0.5, 1.0, 1.5, 2.0]
    
    for current_timeout in timeouts:
        logger.info(f"Processing {len(remaining_websites)} websites with timeout {current_timeout}s")
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(measure_load_time, url, current_timeout): url 
                            for url in remaining_websites}
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error measuring {url}: {e}")
        
        # Process results for this timeout level
        successful = [r for r in results if r['success']]
        timeout_exceeded = [r['domain'] for r in results if not r['success'] and r.get('error') == 'Timeout']
        
        # Add successful results to all_results
        all_results.extend(successful)
        
        # Update remaining websites
        remaining_websites = [url for url in remaining_websites 
                             if extract_domain(url) in timeout_exceeded]
        
        # Save intermediate results
        with open(f'website_timing_{current_timeout}s.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['domain', 'url', 'load_time', 'status_code', 'success', 'error'])
            writer.writeheader()
            writer.writerows(successful)
        
        if not remaining_websites:
            break
    
    # Save final list of websites that exceeded even the highest timeout
    if remaining_websites:
        with open('slow_websites.txt', 'w') as f:
            for url in remaining_websites:
                f.write(f"{url}\n")
    
    return all_results

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
    
    # Limit to first 200 websites
    websites = websites[:200]
    
    start_time = time.time()
    results = process_websites_timing(websites, max_workers=20)
    end_time = time.time()
    
    logger.info(f"Processed {len(websites)} websites in {end_time - start_time:.2f} seconds")
    
    # Save final results to a CSV file
    with open('website_timing_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['domain', 'url', 'load_time', 'status_code', 'success', 'error'])
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda x: x.get('load_time', float('inf'))))
    
    logger.info("Results saved to website_timing_results.csv")

if __name__ == "__main__":
    main()