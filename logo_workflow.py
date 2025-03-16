import os
import argparse
import logging
import time
import sys
from website_logo_extractor import process_websites, read_website_list
from logo_grouping import LogoSimilarityAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logo_analysis_workflow.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LogoAnalysisWorkflow")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Logo Analysis Workflow')
    parser.add_argument('--input', '-i', default='logos_list',
                      help='Input file with list of websites (default: logos_list)')
    parser.add_argument('--limit', '-l', type=int, default=200,
                      help='Limit the number of websites to process (default: 200)')
    parser.add_argument('--workers', '-w', type=int, default=10,
                      help='Number of parallel workers (default: 10)')
    parser.add_argument('--epsilon', '-e', type=float, default=0.3,
                      help='DBSCAN epsilon parameter for clustering (default: 0.3)')
    parser.add_argument('--min-samples', '-m', type=int, default=2,
                      help='DBSCAN min_samples parameter for clustering (default: 2)')
    parser.add_argument('--skip-extraction', action='store_true',
                      help='Skip logo extraction if already done')
    parser.add_argument('--output-dir', '-o', default='results',
                      help='Output directory for results (default: results)')
    
    return parser.parse_args()

def setup_directories(output_dir):
    """Create necessary directories."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'extracted_logos'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logo_clusters'), exist_ok=True)

def main():
    """Run the complete logo analysis workflow."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup directories
    setup_directories(args.output_dir)
    
    # Start timing
    start_time = time.time()
    
    try:
        # Step 1: Read website list
        try:
            websites = read_website_list(args.input)
            logger.info(f"Read {len(websites)} websites from {args.input}")
        except Exception as e:
            logger.error(f"Error reading website list: {e}")
            logger.info("Using default sample websites")
            websites = [
                "google.com",
                "facebook.com",
                "amazon.com",
                "apple.com",
                "microsoft.com",
                "youtube.com",
                "twitter.com",
                "instagram.com",
                "linkedin.com",
                "netflix.com"
            ]
        
        # Limit the number of websites
        websites = websites[:args.limit]
        
        # Step 2: Extract logos from websites
        if not args.skip_extraction:
            logger.info(f"Extracting logos from {len(websites)} websites")
            results = process_websites(websites, max_workers=args.workers, limit=args.limit)
            logger.info(f"Extracted {len(results)} logos")
        else:
            logger.info("Skipping logo extraction")
        
        # Step 3: Analyze logo similarities
        logger.info("Analyzing logo similarities")
        analyzer = LogoSimilarityAnalyzer(
            logos_dir=os.path.join(args.output_dir, 'extracted_logos'),
            results_file="extracted_logos_info.csv"
        )
        
        # Run analysis with provided parameters
        clustered_logos = analyzer.analyze(
            epsilon=args.epsilon,
            min_samples=args.min_samples
        )
        
        # Step 4: Print summary
        print("\nLogo Analysis Results:")
        print("=====================")
        
        total_clusters = sum(1 for k in clustered_logos.keys() if k != -1)
        total_logos = sum(len(logos) for logos in clustered_logos.values())
        unclustered = len(clustered_logos.get(-1, []))
        
        print(f"Total websites processed: {len(websites)}")
        print(f"Total logos extracted: {total_logos}")
        print(f"Number of logo clusters: {total_clusters}")
        print(f"Unclustered logos: {unclustered}")
        
        print("\nTop clusters by size:")
        sorted_clusters = sorted(
            [(k, logos) for k, logos in clustered_logos.items() if k != -1],
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for i, (cluster_id, logos) in enumerate(sorted_clusters[:5], 1):
            domains = ", ".join(logo["domain"] for logo in logos[:3])
            if len(logos) > 3:
                domains += f", ... ({len(logos) - 3} more)"
            print(f"{i}. Cluster {cluster_id}: {len(logos)} logos - {domains}")
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        print(f"\nAnalysis completed in {elapsed_time:.2f} seconds")
        print(f"Results saved to {args.output_dir} directory")
        
    except Exception as e:
        logger.error(f"Error in workflow: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())