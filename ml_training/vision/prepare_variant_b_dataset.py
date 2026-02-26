#!/usr/bin/env python3
"""Prepare Variant B Dataset for Edge-RL Vision Pipeline

This script processes raw tomato images from `data/tomato/` and computes the
10-dimensional regression target vector required by Variant B:
  - 3x RGB Means
  - 3x RGB Standard Deviations
  - 3x RGB Modes (16-bin)
  - 1x Continuous Chromatic Index (X)

Outputs a CSV file `data/variant_b_labels.csv` linking each image path to its
target vector, enabling the MobileNetV2 to be trained for direct state estimation
instead of arbitrary classification.
"""

import os
import glob
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

DATA_DIR = Path("data/tomato")
OUTPUT_CSV = Path("data/variant_b_labels.csv")

def extract_variant_b_features(img_path):
    """
    Extracts the 10D feature vector from an image mimicking the ESP32 C code.
    Filters out background using a simple center crop and saturation threshold.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Background Masking
    # Keep center 60% of image to avoid background distractors
    h, w, _ = img.shape
    crop_scale = 0.6
    ch, cw = int(h * crop_scale), int(w * crop_scale)
    y_start = (h - ch) // 2
    x_start = (w - cw) // 2
    img_center = img[y_start:y_start+ch, x_start:x_start+cw]
    
    # Convert to HSV to mask out pure black/white / heavy green leaves if desired
    # For now, just use the center crop as the 'tomato' representation
    pixels = img_center.reshape(-1, 3)
    
    # ESP32 C logic:
    # Compute sum, sum_sq, and 16-bin histogram
    sum_rgb = np.sum(pixels, axis=0)
    sum_sq_rgb = np.sum(pixels.astype(np.float64)**2, axis=0)
    num_pixels = pixels.shape[0]
    
    mean_rgb = sum_rgb / (num_pixels * 255.0)
    
    variance = (sum_sq_rgb / num_pixels) - ((sum_rgb / num_pixels)**2)
    std_rgb = np.sqrt(np.maximum(0, variance)) / 255.0
    
    # Mode (16 bins)
    mode_rgb = np.zeros(3)
    for c in range(3):
        bins = (pixels[:, c] >> 4)  # 0-15
        counts = np.bincount(bins, minlength=16)
        max_bin = np.argmax(counts)
        mode_rgb[c] = (max_bin * 16 + 8) / 255.0
        
    # Chromatic Index X = G / (R + G)
    r_mean = mean_rgb[0]
    g_mean = mean_rgb[1]
    chromatic_x = g_mean / (r_mean + g_mean + 1e-6)
    
    return [
        str(img_path),
        mean_rgb[0], mean_rgb[1], mean_rgb[2],
        std_rgb[0], std_rgb[1], std_rgb[2],
        mode_rgb[0], mode_rgb[1], mode_rgb[2],
        chromatic_x
    ]

def main():
    if not DATA_DIR.exists():
        print(f"Error: Dataset directory {DATA_DIR} not found.")
        return
        
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(str(DATA_DIR / '**' / ext), recursive=True))
        
    if not image_paths:
        print(f"No images found in {DATA_DIR}")
        return
        
    print(f"Found {len(image_paths)} images. Extracting Variant B labels sequentially...")
    
    results = []
    for path in tqdm(image_paths):
        res = extract_variant_b_features(path)
        if res is not None:
            results.append(res)
                
    df = pd.DataFrame(results, columns=[
        'filepath',
        'mean_r', 'mean_g', 'mean_b',
        'std_r', 'std_g', 'std_b',
        'mode_r', 'mode_g', 'mode_b',
        'chromatic_x'
    ])
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Extraction complete! Saved {len(df)} labels to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
