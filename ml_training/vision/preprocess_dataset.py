
"""
Pre-processing script for mapping Tomato Dataset to Continuous Ripeness ($R_{cont}$).

Logic:
1. Load dataset images (Organized by Class folders: Green, Breaker, etc.)
2. Detect ROI (assume center crop or crude thresholding for now)
3. Calculate Mean RGB and Standard Deviation ($C_{\mu}, C_{\sigma}$)
4. Map Class Labels + RGB fine-tuning -> Continuous Ripeness $R_{cont} \in [0.0, 5.0]$
5. Save metadata to CSV for training the Vision Model.
"""

import os
import cv2
import numpy as np
import pandas as pd
from glob import glob

# Configuration
DATASET_PATH = "data/kaggle_tomato_dataset" # Placeholder path
OUTPUT_CSV = "data/processed_ripeness_metadata.csv"

# Continuous Mapping Reference (ROYG Spectral)
# Roughly: Green=0, Breaker=1, Turning=2, Pink=3, LightRed=4, Red=5
CLASS_MAP = {
    "green": 0.0,
    "breaker": 1.0,
    "turning": 2.0,
    "pink": 3.0,
    "light_red": 4.0,
    "red": 5.0
}

def get_roi(image):
    """Simple center crop ROI to avoid background noise."""
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2
    crop_size = min(h, w) // 2
    return image[cy-crop_size//2 : cy+crop_size//2, cx-crop_size//2 : cx+crop_size//2]

def calculate_stats(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    roi = get_roi(img)
    
    # Calculate Stats
    mean_rgb = np.mean(roi, axis=(0,1)) / 255.0
    std_rgb = np.std(roi, axis=(0,1)) / 255.0
    
    return mean_rgb, std_rgb

def main():
    print(f"Scanning dataset at {DATASET_PATH}...")
    
    data = []
    
    # Iterate through class folders
    for class_name, base_ripeness in CLASS_MAP.items():
        folder_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found. Skipping.")
            continue
            
        files = glob(os.path.join(folder_path, "*.jpg")) + glob(os.path.join(folder_path, "*.png"))
        print(f"Processing {class_name}: {len(files)} images...")
        
        for f in files:
            stats = calculate_stats(f)
            if stats:
                mu, sigma = stats
                
                # Refine Continuous Ripeness based on Red/Green ratio
                # R_cont = Base + fine_tuning
                # Simple logic: If Red > Green, add small offset, etc.
                redness = mu[0] - mu[1] # R - G
                fine_tune = np.clip(redness * 0.5, -0.4, 0.4)
                
                r_continuous = np.clip(base_ripeness + fine_tune, 0.0, 5.0)
                
                data.append({
                    "filename": os.path.basename(f),
                    "class": class_name,
                    "r_continuous": round(r_continuous, 4),
                    "r_mean": round(mu[0], 4),
                    "g_mean": round(mu[1], 4),
                    "b_mean": round(mu[2], 4),
                    "r_std": round(sigma[0], 4),
                    "g_std": round(sigma[1], 4),
                    "b_std": round(sigma[2], 4)
                })
    
    if data:
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved processed metadata to {OUTPUT_CSV}")
        print(df.head())
    else:
        print("No matches found. Check dataset path.")

if __name__ == "__main__":
    main()
