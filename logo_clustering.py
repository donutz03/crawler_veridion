import os
import numpy as np
import cv2
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
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

class ImprovedLogoClusterAnalyzer:
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
                    # Extract domain and company from filename
                    parts = filename.split('_')
                    domain = parts[1] if len(parts) > 1 else 'unknown'
                    company = parts[0] if parts else 'unknown'
                    logo_files.append({
                        'domain': domain,
                        'company': company,
                        'filename': filename,
                        'filepath': filepath
                    })
                except Exception as e:
                    logger.error(f"Error processing filename {filename}: {e}")
        
        logger.info(f"Found {len(logo_files)} logo files")
        return logo_files
    
    def extract_advanced_features(self, logo_files):
        """Extract multiple types of features from logo images."""
        logger.info("Extracting advanced features from logos")
        features_list = []
        valid_logos = []
        
        for logo in logo_files:
            try:
                filepath = logo['filepath']
                
                # Skip SVG files
                if filepath.lower().endswith('.svg'):
                    logger.info(f"Skipping SVG file: {filepath}")
                    continue
                
                # Open image with PIL
                pil_img = Image.open(filepath)
                
                # Robust conversion to RGB
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                # Convert to OpenCV format
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                # Consistent resize
                img = cv2.resize(img, (100, 100))
                
                # Grayscale for some features
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # 1. Perceptual Hashing Features
                phash = imagehash.phash(pil_img)
                dhash = imagehash.dhash(pil_img)
                hash_features = np.array([int(str(h), 16) for h in [phash, dhash]], dtype=np.float32)
                
                # 2. Color Histogram Features
                color_hist = []
                for i in range(3):  # BGR channels
                    hist = cv2.calcHist([img], [i], None, [32], [0, 256])
                    hist = cv2.normalize(hist, hist).flatten()
                    color_hist.extend(hist)
                
                # 3. Edge and Texture Features
                edges = cv2.Canny(gray, 100, 200)
                edge_hist = np.histogram(edges, bins=16, range=(0, 255))[0]
                
                # 4. SIFT Features (optional, as it can be computationally expensive)
                try:
                    sift = cv2.SIFT_create()
                    keypoints, descriptors = sift.detectAndCompute(gray, None)
                    
                    # If SIFT fails or finds no descriptors
                    if descriptors is None or len(descriptors) == 0:
                        sift_features = np.zeros(128)
                    else:
                        # Take mean of descriptors
                        sift_features = np.mean(descriptors, axis=0)
                except Exception as e:
                    logger.warning(f"SIFT feature extraction failed: {e}")
                    sift_features = np.zeros(128)
                
                # 5. Dominant Color Features
                pixels = img.reshape(-1, 3)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                flags = cv2.KMEANS_RANDOM_CENTERS
                compactness, labels, centers = cv2.kmeans(pixels.astype(np.float32), 3, None, criteria, 10, flags)
                
                # Combine all features
                combined_features = np.concatenate([
                    hash_features,          # Perceptual hash
                    color_hist,             # Color distribution
                    edge_hist,              # Edge characteristics
                    sift_features,          # Shape features
                    centers.flatten()       # Dominant colors
                ])
                
                features_list.append(combined_features)
                valid_logos.append(logo)
                
            except Exception as e:
                logger.error(f"Error extracting features from {logo['filepath']}: {e}")
        
        logger.info(f"Successfully extracted features from {len(features_list)} logos")
        return np.array(features_list), valid_logos
    
    def cluster_logos(self, features, valid_logos, clustering_method='dbscan'):
        """Cluster logos using different methods."""
        logger.info(f"Clustering {len(features)} logos using {clustering_method}")
        
        if len(features) == 0:
            logger.warning("No features to cluster")
            return {}
        
        # Normalize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Dimensionality reduction for visualization and clustering
        pca = PCA(n_components=min(10, features.shape[1]))
        reduced_features = pca.fit_transform(scaled_features)
        
        # Different clustering approaches
        if clustering_method == 'dbscan':
            # More flexible DBSCAN parameters
            db = DBSCAN(eps=0.3, min_samples=2, metric='euclidean')
            clusters = db.fit_predict(reduced_features)
        elif clustering_method == 'hierarchical':
            # Hierarchical clustering with predetermined number of clusters
            n_clusters = max(3, int(len(features) * 0.1))  # Dynamic cluster count
            db = AgglomerativeClustering(n_clusters=n_clusters)
            clusters = db.fit_predict(reduced_features)
        
        # Group by cluster
        clustered_logos = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            clustered_logos[int(cluster_id)].append(valid_logos[i])
        
        # Calculate statistics
        n_clusters = len(clustered_logos) - (1 if -1 in clustered_logos else 0)
        unclustered = len(clustered_logos.get(-1, []))
        
        logger.info(f"Found {n_clusters} clusters, with {unclustered} unclustered logos")
        return clustered_logos
    
    def analyze(self, clustering_method='dbscan'):
        """Run the complete clustering analysis pipeline."""
        # Load logos
        logo_files = self.load_logos()
        
        # Extract advanced features
        features, valid_logos = self.extract_advanced_features(logo_files)
        
        # Cluster logos
        clustered_logos = self.cluster_logos(features, valid_logos, clustering_method)
        
        # Visualize and save results
        self.visualize_clusters(clustered_logos)
        cluster_df = self.save_cluster_data(clustered_logos)
        
        # Print summary
        self.print_summary(clustered_logos)
        
        return clustered_logos, cluster_df

    def visualize_clusters(self, clustered_logos):
        """Visualize logo clusters."""
        logger.info("Visualizing clusters")
        
        # Similar to previous implementation
        for cluster_id, logos in clustered_logos.items():
            if cluster_id == -1 or len(logos) < 2:  # Skip noise or tiny clusters
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
                    plt.title(f"{logo['company']}\n{logo['domain']}", fontsize=8)
                    plt.axis('off')
                except Exception as e:
                    logger.error(f"Error displaying {logo['filepath']}: {e}")
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'clusters', f'cluster_{cluster_id}.png'))
            plt.close()
    
    def save_cluster_data(self, clustered_logos):
        """Save cluster data to CSV."""
        cluster_data = []
        
        for cluster_id, logos in clustered_logos.items():
            cluster_name = 'unclustered' if cluster_id == -1 else f'cluster_{cluster_id}'
            
            for logo in logos:
                cluster_data.append({
                    'company': logo['company'],
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
    
    def print_summary(self, clustered_logos):
        """Print a detailed summary of clustering results."""
        print("\nLogo Clustering Summary:")
        print("========================")
        
        cluster_sizes = [(k, len(logos)) for k, logos in clustered_logos.items()]
        total_logos = sum(size for _, size in cluster_sizes)
        
        # Categorize clusters
        meaningful_clusters = [
            (k, size) for k, size in cluster_sizes 
            if k != -1 and size >= 2
        ]
        
        print(f"Total logos analyzed: {total_logos}")
        print(f"Number of meaningful clusters: {len(meaningful_clusters)}")
        
        if -1 in clustered_logos:
            unclustered = len(clustered_logos[-1])
            print(f"Unclustered logos: {unclustered} ({unclustered/total_logos*100:.1f}%)")
        
        # Detailed cluster analysis
        print("\nCluster Details:")
        sorted_clusters = sorted(
            [(k, logos) for k, logos in clustered_logos.items() if k != -1],
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for i, (cluster_id, logos) in enumerate(sorted_clusters[:10], 1):
            # Calculate domain and company distribution
            domains = [logo['domain'] for logo in logos]
            companies = [logo['company'] for logo in logos]
            
            unique_domains = set(domains)
            unique_companies = set(companies)
            
            print(f"\nCluster {cluster_id}:")
            print(f"  Size: {len(logos)} logos")
            print(f"  Unique Domains: {len(unique_domains)}")
            print(f"  Unique Companies: {len(unique_companies)}")
            print("  Sample Domains:", ', '.join(list(unique_domains)[:3]))
            print("  Sample Companies:", ', '.join(list(unique_companies)[:3]))
        
        print(f"\nFull results saved to {self.results_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Advanced Logo Clustering')
    parser.add_argument('--input-dir', '-i', default='logo_clusters/unique',
                      help='Directory containing logos')
    parser.add_argument('--output-dir', '-o', default='cluster_results',
                      help='Directory to save clustering results')
    parser.add_argument('--method', '-m', choices=['dbscan', 'hierarchical'], 
                      default='dbscan', help='Clustering method')
    
    args = parser.parse_args()
    
    # Initialize the analyzer
    analyzer = ImprovedLogoClusterAnalyzer(
        logos_dir=args.input_dir,
        results_dir=args.output_dir
    )
    
    # Run the analysis
    analyzer.analyze(clustering_method=args.method)

if __name__ == "__main__":
    main()