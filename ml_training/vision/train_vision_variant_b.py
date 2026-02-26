#!/usr/bin/env python3
"""Train MobileNetV2 for Variant B State Regression

Trains a compact MobileNetV2 model to predict the 10-dimensional Continuous 
Chromatic Index and RGB distribution parameters directly from raw 96x96 camera 
frames. This bypasses fragile manual computer vision heuristics and provides 
robust, consistent state vectors for the RL policy.

Outputs:
  - outputs/vision_variant_b/best_model.pth
  - outputs/vision_variant_b/mobilenetv2_10d.onnx
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ----------------- Configuration -----------------
IMAGE_SIZE = (96, 96)  # Matches firmware VISION_INPUT_W/H
NUM_CLASSES = 10       # (mean RGB x3, std RGB x3, mode RGB x3, X x1)
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TomatoVariantBDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        
        # Extract features as float32
        self.targets = self.data_frame.iloc[:, 1:].values.astype(np.float32)
        self.img_paths = self.data_frame['filepath'].values

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            # Handle corrupted images by returning a blank array
            image = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE)
            
        if self.transform:
            image = self.transform(image)
            
        target = self.targets[idx]
        return image, target


def train_model(csv_path="data/variant_b_labels.csv", out_dir="outputs/vision_variant_b"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Dataset & Dataloaders
    # Normalizing using standard ImageNet mean/std since we use pretrained MobileNet
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = TomatoVariantBDataset(csv_file=csv_path, transform=transform)
    
    # 80/20 train/val split
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation samples.")
    
    # 2. Build Model
    # Use MobileNetV2 (efficient for ESP32)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Modify the classifier head for 10-dimensional regression
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    model = model.to(DEVICE)
    
    # 3. Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2, factor=0.5)
    
    # 4. Training Loop
    best_val_loss = float('inf')
    best_model_path = out_dir / "best_model.pth"
    
    print(f"Beginning training on {DEVICE}...")
    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("  New best model saved!")
            torch.save(model.state_dict(), best_model_path)
            
    # 5. Export to ONNX
    print("\nTraining complete. Exporting best model to ONNX...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    dummy_input = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1], device=DEVICE)
    onnx_path = out_dir / "mobilenetv2_10d.onnx"
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX model saved to {onnx_path}")
    print("Next step: Convert ONNX to ESP-DL format (`.espdl` / `model_data.h`)!")

if __name__ == "__main__":
    train_model()
