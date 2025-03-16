import os
import csv
import numpy as np
import logging
from PIL import Image
import cv2
from sklearn.cluster import DBSCAN
from collections import defaultdict
import json
import imagehash
import matplotlib.pyplot as plt
from io import BytesIO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logo_similarity.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LogoSimilarity")

class LogoSimilarityAnalyzer:
    def __init__(self, logos_dir="extracted_logos", results_file="extracted_logos_info.csv"):
        self.logos_dir = logos_dir
        self.results_file = results_file
        self.logo_data = []
        self.load_logo_data()
    
    def load_logo_data(self):
        """Load information about extracted logos."""
        try:
            with open(self.results_file, 'r') as f:
                reader = csv.DictReader(f)
                self.logo_data = list(reader)
            
            logger.info(f"Loaded information about {len(self.logo_data)} logos")
        except Exception as e:
            logger.error(f"Error loading logo data: {e}")
            self.logo_data = []
    
    def compute_color_histogram(self, img):
        """Compute color histogram features for an image."""
        # Convert to RGB if it's in a different mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize for consistency
        img = img.resize((100, 100))
        
        # Convert to numpy array and then to OpenCV format
        np_img = np.array(img)
        cv_img = cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        
        # Compute histogram for each channel
        hist_b = cv2.calcHist([cv_img], [0], None, [8], [0, 256])
        hist_g = cv2.calcHist([cv_img], [1], None, [8], [0, 256])
        hist_r = cv2.calcHist([cv_img], [2], None, [8], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist_b, hist_b)
        cv2.normalize(hist_g, hist_g)
        cv2.normalize(hist_r, hist_r)
        
        # Concatenate histograms
        hist = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
        
        return hist
    
    def compute_perceptual_hash(self, img):
        """Compute perceptual hash for an image."""
        # Ensure the image is in a proper format
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Compute different types of hashes
        phash = imagehash.phash(img)
        dhash = imagehash.dhash(img)
        whash = imagehash.whash(img)
        
        # Convert hash objects to numpy arrays for easier comparison
        phash_array = np.array([int(bit) for bit in str(phash)], dtype=np.float32)
        dhash_array = np.array([int(bit) for bit in str(dhash)], dtype=np.float32)
        whash_array = np.array([int(bit) for bit in str(whash)], dtype=np.float32)
        
        # Combine the hashes
        combined_hash = np.concatenate([phash_array, dhash_array, whash_array])
        
        return combined_hash
    
    def extract_sift_features(self, img):
        """Extract SIFT features from an image."""
        # Convert the PIL image to an OpenCV image
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        np_img = np.array(img)
        cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is None:
            # No features found, return empty array
            return np.zeros((1, 128), dtype=np.float32)
        
        # If too many keypoints, sample a subset
        if len(keypoints) > 100:
            indices = np.random.choice(len(keypoints), 100, replace=False)
            descriptors = descriptors[indices]
        
        # If too few, pad with zeros
        if len(keypoints) < 10:
            padding = np.zeros((10 - len(keypoints), 128), dtype=np.float32)
            descriptors = np.vstack([descriptors, padding]) if descriptors.size > 0 else padding
        
        # Compute average descriptor
        avg_descriptor = np.mean(descriptors, axis=0)
        
        return avg_descriptor
    
    def compute_features(self):
        """Compute features for all logos."""
        features = []
        
        for logo in self.logo_data:
            try:
                filepath = logo['filepath']
                if not os.path.exists(filepath):
                    logger.warning(f"Logo file not found: {filepath}")
                    continue
                
                # Open the image
                img = Image.open(filepath)
                
                # Compute features
                hist_features = self.compute_color_histogram(img)
                hash_features = self.compute_perceptual_hash(img)
                sift_features = self.extract_sift_features(img)
                
                # Combine features
                combined_features = np.concatenate([
                    hist_features * 0.3,  # Weight for color histograms
                    hash_features * 0.4,  # Weight for perceptual hashes
                    sift_features * 0.3    # Weight for SIFT features
                ])
                
                features.append({
                    'domain': logo['domain'],
                    'url': logo['url'],
                    'filepath': filepath,
                    'features': combined_features
                })
            except Exception as e:
                logger.error(f"Error computing features for {logo.get('filepath', 'unknown')}: {e}")
        
        logger.info(f"Computed features for {len(features)} logos")
        return features
    
    def cluster_logos(self, features, epsilon=0.3, min_samples=2):
        """Cluster logos based on feature similarity."""
        if not features:
            logger.warning("No features provided for clustering")
            return []
        
        # Extract feature vectors
        feature_vectors = np.array([f['features'] for f in features])
        
        # Normalize feature vectors
        feature_vectors = feature_vectors / np.linalg.norm(feature_vectors, axis=1, keepdims=True)
        
        # Use DBSCAN for clustering
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric='cosine')
        clusters = dbscan.fit_predict(feature_vectors)
        
        # Group by cluster
        clustered_logos = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            clustered_logos[int(cluster_id)].append(features[i])
        
        logger.info(f"Found {len(clustered_logos)} clusters")
        return clustered_logos
    
    def visualize_clusters(self, clustered_logos, output_dir="logo_clusters"):
        """Visualize the clusters by creating montage images."""
        os.makedirs(output_dir, exist_ok=True)
        
        for cluster_id, logos in clustered_logos.items():
            if cluster_id == -1:  # Noise points in DBSCAN
                continue
                
            # Create a figure to display the logos
            n_logos = len(logos)
            cols = min(5, n_logos)
            rows = (n_logos + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
            if rows * cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            # Add each logo to the figure
            for i, logo in enumerate(logos):
                try:
                    img = Image.open(logo['filepath'])
                    axes[i].imshow(img)
                    axes[i].set_title(logo['domain'], fontsize=10)
                    axes[i].axis('off')
                except Exception as e:
                    logger.error(f"Error displaying logo {logo['filepath']}: {e}")
            
            # Hide unused subplots
            for j in range(n_logos, len(axes)):
                axes[j].axis('off')
            
            plt.tight_layout()
            
            # Save the figure
            output_file = os.path.join(output_dir, f"cluster_{cluster_id}.png")
            plt.savefig(output_file)
            plt.close(fig)
            
            logger.info(f"Saved cluster visualization to {output_file}")
    
    def save_results(self, clustered_logos, output_file="logo_clusters.json"):
        """Save clustering results to a JSON file."""
        results = {}
        
        for cluster_id, logos in clustered_logos.items():
            if cluster_id == -1:
                cluster_name = "unclustered"
            else:
                cluster_name = f"cluster_{cluster_id}"
            
            results[cluster_name] = [
                {
                    "domain": logo["domain"],
                    "url": logo["url"],
                    "filepath": logo["filepath"]
                }
                for logo in logos
            ]
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved clustering results to {output_file}")
    
    def analyze(self, epsilon=0.3, min_samples=2):
        """Run the complete analysis pipeline."""
        # Compute features
        features = self.compute_features()
        
        # Cluster logos
        clustered_logos = self.cluster_logos(features, epsilon, min_samples)
        
        # Visualize clusters
        self.visualize_clusters(clustered_logos)
        
        # Save results
        self.save_results(clustered_logos)
        
        return clustered_logos

def main():
    # Initialize analyzer
    analyzer = LogoSimilarityAnalyzer()
    
    # Run analysis
    clustered_logos = analyzer.analyze(epsilon=0.3, min_samples=2)
    
    # Print summary
    print("\nSummary of Logo Clustering:")
    print("=========================")
    
    for cluster_id, logos in sorted(clustered_logos.items()):
        if cluster_id == -1:
            print(f"Unclustered Logos: {len(logos)}")
        else:
            print(f"Cluster {cluster_id}: {len(logos)} logos - Domains: {', '.join(logo['domain'] for logo in logos[:5])}" + 
                 (f", ..." if len(logos) > 5 else ""))
    
    print("\nComplete results saved to logo_clusters.json")
    print("Visualizations saved to the logo_clusters directory")

if __name__ == "__main__":
    main()