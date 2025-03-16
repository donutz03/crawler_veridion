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
            if not os.path.exists(self.results_file):
                logger.error(f"Logo data file not found: {self.results_file}")
                self.logo_data = []
                return
                
            with open(self.results_file, 'r') as f:
                reader = csv.DictReader(f)
                self.logo_data = list(reader)
            
            logger.info(f"Loaded information about {len(self.logo_data)} logos")
        except Exception as e:
            logger.error(f"Error loading logo data: {e}")
            self.logo_data = []
    
    def compute_color_histogram(self, img):
        """Compute color histogram features for an image."""
        try:
            # Convert to RGB if it's in a different mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize for consistency
            img = img.resize((100, 100))
            
            # Convert to numpy array and then to OpenCV format
            np_img = np.array(img)
            cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            
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
        except Exception as e:
            logger.warning(f"Error computing color histogram: {e}, using zero array")
            return np.zeros(24, dtype=np.float32)  # 8 bins × 3 channels
    
    def compute_perceptual_hash(self, img):
        """Compute perceptual hash for an image."""
        try:
            # Ensure the image is in a proper format
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Compute different types of hashes
            phash = imagehash.phash(img)
            dhash = imagehash.dhash(img)
            whash = imagehash.whash(img)
            
            # Convert hash objects to binary arrays (0s and 1s only)
            # Each hash is typically a 64-bit value (8x8 pixels)
            phash_bin = np.array([int(bit) for bit in bin(int(str(phash), 16))[2:].zfill(64)], dtype=np.float32)
            dhash_bin = np.array([int(bit) for bit in bin(int(str(dhash), 16))[2:].zfill(64)], dtype=np.float32)
            whash_bin = np.array([int(bit) for bit in bin(int(str(whash), 16))[2:].zfill(64)], dtype=np.float32)
            
            # Combine the hashes
            combined_hash = np.concatenate([phash_bin, dhash_bin, whash_bin])
            
            return combined_hash
        except Exception as e:
            logger.warning(f"Error computing perceptual hash: {e}, using zero array")
            return np.zeros(192, dtype=np.float32)  # 64 bits × 3 hash types
    
    def extract_sift_features(self, img):
        """Extract SIFT features from an image."""
        try:
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
            
            if descriptors is None or len(descriptors) == 0:
                # No features found, return zero array with expected shape
                logger.debug(f"No SIFT features detected, using zero array")
                return np.zeros(128, dtype=np.float32)
            
            # If too many keypoints, sample a subset
            if len(keypoints) > 100:
                indices = np.random.choice(len(keypoints), 100, replace=False)
                descriptors = descriptors[indices]
            
            # Compute average descriptor
            avg_descriptor = np.mean(descriptors, axis=0)
            
            return avg_descriptor
        
        except Exception as e:
            logger.warning(f"Error in SIFT extraction: {e}, using zero array")
            return np.zeros(128, dtype=np.float32)  # Standard SIFT descriptor length
    
    def compute_features(self):
        """Compute features for all logos."""
        features = []
        total_logos = len(self.logo_data)
        processed = 0
        skipped = 0
        
        for logo in self.logo_data:
            try:
                filepath = logo['filepath']
                if not os.path.exists(filepath):
                    logger.warning(f"Logo file not found: {filepath}")
                    skipped += 1
                    continue
                
                # Open the image
                img = Image.open(filepath)
                
                # Compute features
                hist_features = self.compute_color_histogram(img)
                hash_features = self.compute_perceptual_hash(img)
                sift_features = self.extract_sift_features(img)
                
                # Verify all features have proper dimensions
                if hist_features.ndim != 1 or hash_features.ndim != 1 or sift_features.ndim != 1:
                    logger.warning(f"Dimension mismatch in features for {filepath}, skipping")
                    skipped += 1
                    continue

                # Verify all features have the expected shape
                if len(hist_features) != 24:  # 8 bins per channel × 3 channels
                    logger.warning(f"Unexpected histogram dimension in {filepath}, skipping")
                    skipped += 1
                    continue
                
                if len(hash_features) != 192:  # 64 bits per hash × 3 hash types
                    logger.warning(f"Unexpected hash dimension in {filepath}, skipping")
                    skipped += 1
                    continue
                
                # Combine features with proper weights
                # Normalize the feature vectors first
                hist_norm = hist_features / (np.linalg.norm(hist_features) + 1e-10)
                hash_norm = hash_features / (np.linalg.norm(hash_features) + 1e-10)
                sift_norm = sift_features / (np.linalg.norm(sift_features) + 1e-10)
                
                combined_features = np.concatenate([
                    hist_norm * 0.3,  # Weight for color histograms
                    hash_norm * 0.4,  # Weight for perceptual hashes
                    sift_norm * 0.3   # Weight for SIFT features
                ])
                
                features.append({
                    'domain': logo['domain'],
                    'url': logo['url'],
                    'filepath': filepath,
                    'features': combined_features
                })
                
                processed += 1
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{total_logos} logos ({skipped} skipped)")
                    
            except Exception as e:
                logger.error(f"Error computing features for {logo.get('filepath', 'unknown')}: {str(e)}")
                skipped += 1
        
        logger.info(f"Computed features for {len(features)}/{total_logos} logos (skipped {skipped})")
        return features
    
    def cluster_logos(self, features, epsilon=0.3, min_samples=2):
        """Cluster logos based on feature similarity."""
        if not features or len(features) == 0:
            logger.warning("No features provided for clustering")
            return {}
        
        # Extract feature vectors
        feature_vectors = np.array([f['features'] for f in features])
        
        # Check if we have enough samples for meaningful clustering
        if len(feature_vectors) < min_samples:
            logger.warning(f"Not enough samples for clustering (have {len(feature_vectors)}, need {min_samples})")
            # Return all logos as unclustered
            return {-1: features}
        
        logger.info(f"Clustering {len(feature_vectors)} logo feature vectors")
        
        try:
            # Normalize feature vectors
            norms = np.linalg.norm(feature_vectors, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms < 1e-10] = 1.0
            feature_vectors = feature_vectors / norms
            
            # Use DBSCAN for clustering
            logger.info(f"Running DBSCAN with epsilon={epsilon}, min_samples={min_samples}")
            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric='cosine')
            clusters = dbscan.fit_predict(feature_vectors)
            
            # Group by cluster
            clustered_logos = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                clustered_logos[int(cluster_id)].append(features[i])
            
            # Count the clusters and unclustered logos
            num_clusters = len([k for k in clustered_logos.keys() if k != -1])
            num_unclustered = len(clustered_logos.get(-1, []))
            
            logger.info(f"Found {num_clusters} clusters with {len(features) - num_unclustered} logos")
            logger.info(f"Unclustered logos: {num_unclustered}")
            
            return clustered_logos
            
        except Exception as e:
            logger.error(f"Error during clustering: {str(e)}")
            # Return all logos as unclustered on error
            return {-1: features}
    
    def visualize_clusters(self, clustered_logos, output_dir="logo_clusters"):
        """Visualize the clusters by creating montage images."""
        if not clustered_logos:
            logger.warning("No clusters to visualize")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        for cluster_id, logos in clustered_logos.items():
            if cluster_id == -1:  # Noise points in DBSCAN
                continue
                
            # Create a figure to display the logos
            n_logos = len(logos)
            cols = min(5, n_logos)
            rows = (n_logos + cols - 1) // cols
            
            try:
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
            except Exception as e:
                logger.error(f"Error visualizing cluster {cluster_id}: {e}")
    
    def save_results(self, clustered_logos, output_file="logo_clusters.json"):
        """Save clustering results to a JSON file."""
        results = {}
        
        if not clustered_logos:
            logger.warning("No clusters to save")
            results["error"] = "No clusters found"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            return
            
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
    print("\nLogo Analysis Results:")
    print("=====================")
    
    if not clustered_logos:
        print("No logo clusters were found. Check the logs for errors.")
        return 1
            
    total_clusters = sum(1 for k in clustered_logos.keys() if k != -1)
    total_logos = sum(len(logos) for logos in clustered_logos.values())
    unclustered = len(clustered_logos.get(-1, []))
    
    print(f"Total logos analyzed: {total_logos}")
    print(f"Number of logo clusters: {total_clusters}")
    print(f"Unclustered logos: {unclustered}")
    
    if total_clusters > 0:
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
    else:
        print("\nNo clusters were formed. Logos might be too dissimilar or feature extraction failed.")
    
    print(f"\nResults saved to logo_clusters.json")
    print(f"Visualizations saved to the logo_clusters directory")
    
    return 0

if __name__ == "__main__":
    main()