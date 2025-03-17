import cv2
import numpy as np
import os
from PIL import Image
import shutil
import argparse
import logging
from tqdm import tqdm
from collections import defaultdict
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logo_similarity.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LogoSimilarityAnalyzer")

def preprocess_image(image_path):
    """Preprocesează imaginea pentru comparație"""
    try:
        # Deschide imaginea cu PIL
        image = Image.open(image_path)
        
        # Verifică dacă este SVG (nu poate fi procesat direct)
        if image_path.lower().endswith('.svg'):
            logger.info(f"Skipping SVG file: {image_path}")
            return None
        
        # Convertește la RGB dacă e necesar
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convertește la formatul OpenCV
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Redimensionare pentru consistență
        img = cv2.resize(img, (100, 100))
        
        # Convertire la tonuri de gri
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Egalizare histogramă pentru a reduce efectele de iluminare
        equalized = cv2.equalizeHist(gray)
        
        # Aplicăm un blur ușor pentru a reduce zgomotul
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
        
        return {
            'gray': gray,
            'color': img,
            'equalized': equalized,
            'blurred': blurred
        }
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None

def extract_domain_from_filename(filename):
    """Extrage numele domeniului din numele fișierului logo"""
    parts = filename.split('_')
    if len(parts) > 0:
        return parts[0]
    return None

def extract_company_name(domain):
    """Încearcă să extragă numele companiei din domeniu"""
    if not domain:
        return ""
    
    # Elimină extensia de domeniu
    domain_parts = domain.split('.')
    if len(domain_parts) > 1:
        domain_name = domain_parts[0]
    else:
        domain_name = domain
    
    # Elimină subdomenii comune
    common_subdomains = ['www', 'web', 'shop', 'store', 'online', 'blog', 'info']
    for subdomain in common_subdomains:
        if domain_name.startswith(subdomain + '.'):
            domain_name = domain_name[len(subdomain) + 1:]
    
    # Desparte cuvintele dacă sunt în format camelCase sau snake_case
    words = re.findall(r'[A-Za-z][a-z]*|[0-9]+', domain_name)
    
    # Elimină cuvinte comune nesemnificative
    ignored_words = ['com', 'net', 'org', 'io', 'co', 'inc', 'ltd', 'llc', 'gmbh']
    filtered_words = [w for w in words if w.lower() not in ignored_words]
    
    if filtered_words:
        return ' '.join(filtered_words)
    return domain_name

def extract_features(img_data):
    """Extrage caracteristici pentru comparație"""
    if img_data is None:
        return None
    
    try:
        features = {}
        
        # 1. Histograma pe imagine originală în tonuri de gri
        hist_gray = cv2.calcHist([img_data['gray']], [0], None, [64], [0, 256])
        hist_gray = cv2.normalize(hist_gray, hist_gray).flatten()
        features['hist_gray'] = hist_gray
        
        # 2. Histograma pe imagine egalizată
        hist_eq = cv2.calcHist([img_data['equalized']], [0], None, [64], [0, 256])
        hist_eq = cv2.normalize(hist_eq, hist_eq).flatten()
        features['hist_eq'] = hist_eq
        
        # 3. Histograma de culoare (pe canale BGR separate)
        color_hists = []
        for i in range(3):
            hist = cv2.calcHist([img_data['color']], [i], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            color_hists.extend(hist)
        features['hist_color'] = np.array(color_hists)
        
        # 4. Calculăm hash-uri perceptuale (mai robuste la mici modificări)
        img_for_hash = img_data['blurred']
        
        # Difference Hash
        dhash = compute_dhash(img_for_hash)
        features['dhash'] = dhash
        
        # Average Hash
        ahash = compute_ahash(img_for_hash)
        features['ahash'] = ahash
        
        # Perceptual Hash
        phash = compute_phash(img_for_hash)
        features['phash'] = phash
        
        # 5. Caracteristici bazate pe muchii (forme)
        # Aplicăm un detector de muchii Canny
        edges = cv2.Canny(img_data['blurred'], 100, 200)
        
        # Profiluri de contur pe X și Y
        x_profile = np.sum(edges, axis=0) / edges.shape[0]
        y_profile = np.sum(edges, axis=1) / edges.shape[1]
        
        features['edge_profile'] = np.concatenate([x_profile, y_profile])
        
        # 6. SIFT pentru potriviri de caracteristici (opțional, destul de lent)
        # sift = cv2.SIFT_create()
        # keypoints, descriptors = sift.detectAndCompute(img_data['gray'], None)
        # if descriptors is not None and len(descriptors) > 0:
        #     features['sift_descriptors'] = descriptors
        # else:
        #     features['sift_descriptors'] = np.zeros((1, 128))
        
        return features
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

def compute_dhash(image, hash_size=8):
    """Calculează un difference hash pentru imagine"""
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for i, v in enumerate(diff.flatten()) if v])

def compute_ahash(image, hash_size=8):
    """Calculează un average hash pentru imagine"""
    resized = cv2.resize(image, (hash_size, hash_size))
    mean = np.mean(resized)
    hash_value = resized > mean
    return sum([2 ** i for i, v in enumerate(hash_value.flatten()) if v])

def compute_phash(image, hash_size=8):
    """Calculează un perceptual hash pentru imagine"""
    img_size = hash_size * 4
    resized = cv2.resize(image, (img_size, img_size))
    dct = cv2.dct(np.float32(resized))
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low)
    hash_value = dct_low > med
    return sum([2 ** i for i, v in enumerate(hash_value.flatten()) if v])

def compare_images(features1, features2, domain1, domain2, company1, company2):
    """Compară două seturi de caracteristici și returnează scorul de similaritate"""
    if features1 is None or features2 is None:
        return 0.0
    
    scores = {}
    
    # 1. Compară hash-urile perceptuale (diferența Hamming)
    hash_types = ['dhash', 'ahash', 'phash']
    for hash_type in hash_types:
        hash_diff = bin(features1[hash_type] ^ features2[hash_type]).count('1')
        hash_size = 8 * 8  # Dimensiunea hash-ului (64 biți)
        scores[hash_type] = 1 - (hash_diff / hash_size)
    
    # 2. Compară histogramele
    hist_types = ['hist_gray', 'hist_eq', 'hist_color']
    for hist_type in hist_types:
        hist_sim = cv2.compareHist(
            features1[hist_type].reshape(-1, 1), 
            features2[hist_type].reshape(-1, 1), 
            cv2.HISTCMP_CORREL
        )
        scores[hist_type] = max(0, hist_sim)  # Asigurăm că e pozitiv
    
    # 3. Compară profilurile de muchii
    try:
        edge_sim = np.corrcoef(features1['edge_profile'], features2['edge_profile'])[0, 1]
        if np.isnan(edge_sim):
            edge_sim = 0
        scores['edge'] = max(0, edge_sim)
    except:
        scores['edge'] = 0
    
    # 4. Factor de bonus pentru domenii similare
    domain_similarity = 0
    if domain1 and domain2:
        # Același domeniu de bază?
        domain1_base = domain1.split('.')[-2] if len(domain1.split('.')) > 1 else domain1
        domain2_base = domain2.split('.')[-2] if len(domain2.split('.')) > 1 else domain2
        
        if domain1_base == domain2_base:
            domain_similarity = 0.5
        
        # Verificăm dacă domeniile conțin unul pe celălalt
        elif domain1_base in domain2 or domain2_base in domain1:
            domain_similarity = 0.3
    
    # 5. Bonus pentru nume de companie similar
    company_similarity = 0
    if company1 and company2 and len(company1) > 2 and len(company2) > 2:
        if company1 == company2:
            company_similarity = 0.5
        elif company1 in company2 or company2 in company1:
            company_similarity = 0.3
    
    # Calculăm scorul final combinat
    weights = {
        'dhash': 0.15,
        'ahash': 0.10,
        'phash': 0.15,
        'hist_gray': 0.10,
        'hist_eq': 0.05,
        'hist_color': 0.15,
        'edge': 0.10,
        'domain': 0.10,
        'company': 0.10
    }
    
    final_score = (
        weights['dhash'] * scores['dhash'] +
        weights['ahash'] * scores['ahash'] +
        weights['phash'] * scores['phash'] +
        weights['hist_gray'] * scores['hist_gray'] +
        weights['hist_eq'] * scores['hist_eq'] +
        weights['hist_color'] * scores['hist_color'] +
        weights['edge'] * scores['edge'] +
        weights['domain'] * domain_similarity +
        weights['company'] * company_similarity
    )
    
    return final_score

def group_similar_logos(logos_dir, output_dir, threshold=0.80, company_awareness=True):
    """Grupează logo-urile similare bazat pe caracteristicile vizuale și informații de domeniu"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "unique"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "groups"), exist_ok=True)
    
    # 1. Colectează toate fișierele imagine
    image_files = []
    for filename in os.listdir(logos_dir):
        filepath = os.path.join(logos_dir, filename)
        if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.ico', '.bmp')):
            image_files.append({
                'path': filepath,
                'filename': filename,
                'domain': extract_domain_from_filename(filename)
            })
    
    logger.info(f"Found {len(image_files)} image files")
    
    # 2. Adăugăm informații despre companie (opțional)
    if company_awareness:
        for img in image_files:
            img['company'] = extract_company_name(img['domain'])
    
    # 3. Preprocesează și extrage caracteristici pentru toate imaginile
    logger.info("Processing images and extracting features...")
    features_map = {}
    for img_data in tqdm(image_files):
        processed_img = preprocess_image(img_data['path'])
        if processed_img is not None:
            features = extract_features(processed_img)
            if features is not None:
                features_map[img_data['path']] = {
                    'features': features,
                    'domain': img_data['domain'],
                    'company': img_data.get('company', ''),
                    'filename': img_data['filename']
                }
    
    logger.info(f"Successfully processed {len(features_map)} images")
    
    # 4. Calculează similaritățile între toate perechile de imagini
    logger.info("Computing pairwise similarities...")
    image_paths = list(features_map.keys())
    similarity_matrix = np.zeros((len(image_paths), len(image_paths)))
    
    for i in tqdm(range(len(image_paths))):
        path_i = image_paths[i]
        data_i = features_map[path_i]
        
        for j in range(i+1, len(image_paths)):
            path_j = image_paths[j]
            data_j = features_map[path_j]
            
            # Calcul similaritate
            similarity = compare_images(
                data_i['features'], 
                data_j['features'],
                data_i['domain'],
                data_j['domain'],
                data_i['company'],
                data_j['company']
            )
            
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Matricea este simetrică
    
    # 5. Grupează logo-urile folosind un algoritm gready
    logger.info("Grouping similar logos...")
    groups = []
    processed = set()
    
    # Sortăm imaginile după numele de domeniu pentru grupare mai intuitivă
    sorted_indices = sorted(range(len(image_paths)), 
                           key=lambda i: features_map[image_paths[i]]['domain'])
    
    for idx in sorted_indices:
        if idx in processed:
            continue
        
        path = image_paths[idx]
        
        # Creează un nou grup
        current_group = [path]
        processed.add(idx)
        
        # Găsește toate logo-urile similare
        for j in range(len(image_paths)):
            if j in processed or j == idx:
                continue
            
            if similarity_matrix[idx, j] >= threshold:
                current_group.append(image_paths[j])
                processed.add(j)
        
        if len(current_group) > 0:
            groups.append(current_group)
    
    # 6. Sortează grupurile după mărime
    groups.sort(key=len, reverse=True)
    
    # 7. Generează rezultate și raport
    unique_count = len(groups)
    total_logos = len(image_paths)
    duplicate_count = total_logos - unique_count
    
    with open(os.path.join(output_dir, "similarity_report.txt"), "w") as report:
        report.write("=== Logo Similarity Analysis Report ===\n\n")
        report.write(f"Total logos analyzed: {total_logos}\n")
        report.write(f"Found {unique_count} unique logo groups\n")
        report.write(f"Identified {duplicate_count} similar logos\n\n")
        
        for i, group in enumerate(groups, 1):
            # Alegem un reprezentant pentru grup (primul logo din grup)
            representative = group[0]
            similar_logos = group[1:]
            
            # Copiem logo-ul unic
            dest_path = os.path.join(output_dir, "unique", os.path.basename(representative))
            shutil.copy2(representative, dest_path)
            
            # Creăm un director pentru grup dacă are mai multe logo-uri
            if len(group) > 1:
                group_dir = os.path.join(output_dir, "groups", f"group_{i}")
                os.makedirs(group_dir, exist_ok=True)
                
                for logo_path in group:
                    shutil.copy2(logo_path, os.path.join(group_dir, os.path.basename(logo_path)))
            
            # Adăugăm în raport
            domain = features_map[representative]['domain']
            company = features_map[representative]['company']
            
            report.write(f"Group {i}: {len(group)} logos - Domain: {domain}, Company: {company}\n")
            report.write(f"  Representative: {os.path.basename(representative)}\n")
            
            if similar_logos:
                report.write("  Similar logos:\n")
                for sim in similar_logos:
                    similarity = similarity_matrix[image_paths.index(representative), image_paths.index(sim)]
                    report.write(f"    - {os.path.basename(sim)} (Similarity: {similarity:.2f})\n")
            report.write("\n")
    
    logger.info(f"Found {unique_count} unique logo groups and {duplicate_count} similar logos")
    logger.info(f"Unique logos copied to {os.path.join(output_dir, 'unique')}")
    logger.info(f"Grouped logos copied to {os.path.join(output_dir, 'groups')}")
    logger.info(f"Full report saved to {os.path.join(output_dir, 'similarity_report.txt')}")
    
    return groups

def main():
    parser = argparse.ArgumentParser(description="Group similar logos based on visual features and domain info")
    parser.add_argument("--input", "-i", default="company_logos", help="Input directory containing logos")
    parser.add_argument("--output", "-o", default="similarity_results", help="Output directory for results")
    parser.add_argument("--threshold", "-t", type=float, default=0.80, help="Similarity threshold (0.0-1.0)")
    parser.add_argument("--company-awareness", "-c", action="store_true", help="Use company name information for grouping")
    
    args = parser.parse_args()
    
    group_similar_logos(args.input, args.output, args.threshold, args.company_awareness)

if __name__ == "__main__":
    main()