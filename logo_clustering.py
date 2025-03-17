import os
import numpy as np
import cv2
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import imagehash
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logo_clustering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LogoClustering")

class LogoClusterAnalyzer:
    def __init__(self, logos_dir="logo_clusters/unique", results_dir="cluster_results"):
        self.logos_dir = logos_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'clusters'), exist_ok=True)
        
    def load_logos(self):
        """Load all logos from the directory."""
        logger.info(f"Loading logos from {self.logos_dir}")
        logo_files = []
        
        for filename in os.listdir(self.logos_dir):
            filepath = os.path.join(self.logos_dir, filename)
            if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico')):
                try:
                    # Get domain from filename (format is typically domain_hash.ext)
                    domain = filename.split('_')[0]
                    logo_files.append({
                        'domain': domain,
                        'filename': filename,
                        'filepath': filepath
                    })
                except Exception as e:
                    logger.error(f"Error processing filename {filename}: {e}")
        
        logger.info(f"Found {len(logo_files)} logo files")
        return logo_files
    
    def extract_features(self, logo_files):
        """Extract features from logo images for clustering."""
        logger.info("Extracting features from logos")
        features = []
        valid_logos = []
        
        for logo in logo_files:
            try:
                filepath = logo['filepath']
                
                # Skip SVG files - they need special handling
                if filepath.lower().endswith('.svg'):
                    logger.info(f"Skipping SVG file: {filepath}")
                    continue
                
                # Open image with PIL first to handle different formats
                pil_img = Image.open(filepath)
                
                # Calculate perceptual hash (good for logo comparison)
                phash = imagehash.phash(pil_img)
                dhash = imagehash.dhash(pil_img)
                
                # Convert hash to feature vector - using hexadecimal value for more reliability
                # This fix replaces the original code that tried to parse individual hash bits
                hash_str = str(phash) + str(dhash)
                # Convert the hash string directly to a numerical value
                hash_features = np.array([int(h, 16) for h in hash_str.split()], dtype=np.float32)
                
                # Convert to OpenCV format for more features
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                img = np.array(pil_img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Resize for consistency
                img = cv2.resize(img, (100, 100))
                
                # Calculate color histograms
                color_features = []
                for i in range(3):  # BGR channels
                    hist = cv2.calcHist([img], [i], None, [16], [0, 256])
                    hist = cv2.normalize(hist, hist).flatten()
                    color_features.extend(hist)
                
                # Try to extract SIFT features if image is suitable
                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    sift = cv2.SIFT_create()
                    keypoints, sift_descriptors = sift.detectAndCompute(gray, None)
                    
                    if sift_descriptors is not None and len(sift_descriptors) > 0:
                        # Take mean of SIFT descriptors
                        sift_features = np.mean(sift_descriptors, axis=0)
                    else:
                        sift_features = np.zeros(128)  # SIFT features are 128-dimensional
                except Exception as e:
                    logger.warning(f"Could not extract SIFT features from {filepath}: {e}")
                    sift_features = np.zeros(128)
                
                # Combine features - now we'll normalize them to have similar weights
                hash_features_normalized = hash_features / np.max(hash_features) if np.max(hash_features) > 0 else hash_features
                color_features = np.array(color_features)
                color_features_normalized = color_features / np.max(color_features) if np.max(color_features) > 0 else color_features
                sift_features_normalized = sift_features / np.max(sift_features) if np.max(sift_features) > 0 else sift_features
                
                # Combine all normalized features
                combined_features = np.concatenate([
                    hash_features_normalized,  # Hash features (perceptual similarity)
                    color_features_normalized,  # Color distribution
                    sift_features_normalized   # Shape features
                ])
                
                features.append(combined_features)
                valid_logos.append(logo)
                
            except Exception as e:
                logger.error(f"Error extracting features from {logo['filepath']}: {e}")
        
        logger.info(f"Successfully extracted features from {len(features)} logos")
        return np.array(features), valid_logos
    
    def cluster_logos(self, features, valid_logos, eps=0.5, min_samples=2):
        """Cluster logos based on feature similarity."""
        logger.info(f"Clustering {len(features)} logos with DBSCAN (eps={eps}, min_samples={min_samples})")
        
        if len(features) == 0:
            logger.warning("No features to cluster")
            return {}
        
        # Normalize features for better clustering
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Apply DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        clusters = db.fit_predict(scaled_features)
        
        # Group by cluster
        clustered_logos = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            clustered_logos[int(cluster_id)].append(valid_logos[i])
        
        n_clusters = len(clustered_logos) - (1 if -1 in clustered_logos else 0)
        unclustered = len(clustered_logos.get(-1, []))
        
        logger.info(f"Found {n_clusters} clusters, with {unclustered} unclustered logos")
        return clustered_logos
    
    def visualize_clusters(self, clustered_logos):
        """Create visual representations of the clusters."""
        logger.info("Visualizing clusters")
        
        for cluster_id, logos in clustered_logos.items():
            if cluster_id == -1 or len(logos) < 2:  # Skip noise points or tiny clusters
                continue
                
            # Create a figure to display logos in this cluster
            n_logos = len(logos)
            cols = min(5, n_logos)
            rows = (n_logos + cols - 1) // cols
            
            plt.figure(figsize=(15, 3 * rows))
            
            for i, logo in enumerate(logos):
                try:
                    img = Image.open(logo['filepath'])
                    plt.subplot(rows, cols, i + 1)
                    plt.imshow(img)
                    plt.title(logo['domain'], fontsize=10)
                    plt.axis('off')
                except Exception as e:
                    logger.error(f"Error displaying {logo['filepath']}: {e}")
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'clusters', f'cluster_{cluster_id}.png'))
            plt.close()
            
        # Create a summary visualization for top clusters
        top_clusters = sorted(
            [(k, logos) for k, logos in clustered_logos.items() if k != -1],
            key=lambda x: len(x[1]),
            reverse=True
        )[:10]  # Top 10 clusters
        
        if top_clusters:
            plt.figure(figsize=(15, 15))
            for i, (cluster_id, logos) in enumerate(top_clusters):
                try:
                    # Show first logo of each top cluster
                    plt.subplot(3, 4, i + 1)
                    img = Image.open(logos[0]['filepath'])
                    plt.imshow(img)
                    plt.title(f"Cluster {cluster_id}: {len(logos)} logos", fontsize=12)
                    plt.axis('off')
                except Exception as e:
                    logger.error(f"Error in summary for cluster {cluster_id}: {e}")
                    
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'top_clusters_summary.png'))
            plt.close()
    
    def save_cluster_data(self, clustered_logos):
        """Save cluster data to CSV and JSON files."""
        cluster_data = []
        
        for cluster_id, logos in clustered_logos.items():
            cluster_name = 'unclustered' if cluster_id == -1 else f'cluster_{cluster_id}'
            
            for logo in logos:
                cluster_data.append({
                    'domain': logo['domain'],
                    'filename': logo['filename'],
                    'filepath': logo['filepath'],
                    'cluster': cluster_name
                })
        
        # Save to CSV
        df = pd.DataFrame(cluster_data)
        csv_path = os.path.join(self.results_dir, 'logo_clusters.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved cluster data to {csv_path}")
        
        # Generate summary statistics
        summary = df['cluster'].value_counts().reset_index()
        summary.columns = ['cluster', 'count']
        summary_path = os.path.join(self.results_dir, 'cluster_summary.csv')
        summary.to_csv(summary_path, index=False)
        logger.info(f"Saved cluster summary to {summary_path}")
        
        return df
    
    def analyze(self, eps=0.5, min_samples=2):
        """Run the complete clustering analysis pipeline."""
        # Load logos
        logo_files = self.load_logos()
        
        # Extract features
        features, valid_logos = self.extract_features(logo_files)
        
        # Cluster logos
        clustered_logos = self.cluster_logos(features, valid_logos, eps, min_samples)
        
        # Visualize clusters
        self.visualize_clusters(clustered_logos)
        
        # Save cluster data
        cluster_df = self.save_cluster_data(clustered_logos)
        
        # Print summary report
        self.print_summary(clustered_logos)
        
        return clustered_logos, cluster_df
    
    def print_summary(self, clustered_logos):
        """Print a summary of the clustering results."""
        print("\nLogo Clustering Summary:")
        print("========================")
        
        cluster_sizes = [(k, len(logos)) for k, logos in clustered_logos.items()]
        total_logos = sum(size for _, size in cluster_sizes)
        
        # Count meaningful clusters (size >= 2)
        meaningful_clusters = [
            (k, size) for k, size in cluster_sizes 
            if k != -1 and size >= 2
        ]
        
        print(f"Total logos analyzed: {total_logos}")
        print(f"Number of clusters formed: {len(meaningful_clusters)}")
        
        if -1 in clustered_logos:
            unclustered = len(clustered_logos[-1])
            print(f"Unclustered logos: {unclustered} ({unclustered/total_logos*100:.1f}%)")
        
        # Show top clusters
        print("\nTop 5 largest clusters:")
        sorted_clusters = sorted(
            [(k, size) for k, size in cluster_sizes if k != -1],
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (cluster_id, size) in enumerate(sorted_clusters[:5], 1):
            domains = [logo['domain'] for logo in clustered_logos[cluster_id][:3]]
            print(f"  {i}. Cluster {cluster_id}: {size} logos - Examples: {', '.join(domains)}...")
        
        print(f"\nFull results saved to {self.results_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Cluster logo images based on visual similarity')
    parser.add_argument('--input-dir', '-i', default='logo_clusters/unique',
                      help='Directory containing extracted logos (default: logo_clusters/unique)')
    parser.add_argument('--output-dir', '-o', default='cluster_results',
                      help='Directory to save clustering results (default: cluster_results)')
    parser.add_argument('--epsilon', '-e', type=float, default=0.5,
                      help='DBSCAN epsilon parameter for clustering (default: 0.5)')
    parser.add_argument('--min-samples', '-m', type=int, default=2,
                      help='DBSCAN min_samples parameter for clustering (default: 2)')
    
    args = parser.parse_args()
    
    # Initialize the analyzer
    analyzer = LogoClusterAnalyzer(
        logos_dir=args.input_dir,
        results_dir=args.output_dir
    )
    
    # Run the analysis
    analyzer.analyze(eps=args.epsilon, min_samples=args.min_samples)

if __name__ == "__main__":
    main()