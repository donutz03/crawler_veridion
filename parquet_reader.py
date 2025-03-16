import pandas as pd
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ParquetReader")

def read_websites_from_parquet(parquet_path):
    """
    Read website URLs from a Parquet file.
    
    Args:
        parquet_path (str): Path to the Parquet file
        
    Returns:
        list: List of website URLs
    """
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
        
        return cleaned_websites
        
    except Exception as e:
        logger.error(f"Error reading Parquet file: {e}")
        # Return an empty list in case of error
        return []

if __name__ == "__main__":
    parquet_path = "logos.snappy.parquet"
    websites = read_websites_from_parquet(parquet_path)
    print(f"Read {len(websites)} websites")
    print("Sample websites:")
    for site in websites[:10]:
        print(f"  - {site}")