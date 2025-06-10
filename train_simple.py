#!/usr/bin/env python3
"""Simplified training script to debug constant prediction issue"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
sys.path.append('src')

from src.models.multihead_vio import MultiHeadVIOModel
from train_improved import SeparateFeatureDataset, collate_fn
from torch.utils.data import DataLoader

def check_predictions(model, loader, device):
    """Check if model is making varied predictions"""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 5:  # Check first 5 batches
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            predictions = model(batch)
            
            # Get first sample's predictions
            trans = predictions['translation'][0].cpu().numpy()
            all_preds.append(trans)
    
    all_preds = np.array(all_preds)  # [5, 10, 3]
    
    # Check variance
    std = np.std(all_preds.reshape(-1, 3), axis=0)
    mean = np.mean(all_preds.reshape(-1, 3), axis=0)
    
    print(f"Prediction stats:")
    print(f"  Mean: {mean}")
    print(f"  Std: {std}")
    print(f"  Varied: {np.all(std > 0.01)}")
    
    return np.all(std > 0.01)  # Return True if predictions vary

def main():
    # Simple setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with balanced weights
    model = MultiHeadVIOModel(
        rotation_weight=1.0,  # Equal weights
        translation_weight=1.0,
        learning_rate=1e-4,  # Lower LR
        weight_decay=1e-5
    ).to(device)
    
    # Data
    train_dataset = SeparateFeatureDataset("/mnt/ssd_ext/incSeg-data/aria_latent_data_pretrained/train")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    val_dataset = SeparateFeatureDataset("/mnt/ssd_ext/incSeg-data/aria_latent_data_pretrained/val")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Simple optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("Starting simple training...")
    
    for epoch in range(10):
        model.train()
        total_loss = 0
        
        for i, batch in enumerate(train_loader):
            if i >= 50:  # Quick test with 50 batches
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward
            predictions = model(batch)
            loss_dict = model.compute_loss(predictions, batch)
            loss = loss_dict['total_loss']
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
        
        # Check if predictions vary
        varied = check_predictions(model, val_loader, device)
        print(f"Epoch {epoch}: Avg Loss = {total_loss/50:.4f}, Predictions varied: {varied}")
        
        if varied:
            print("Model is learning varied predictions!")
            break

if __name__ == "__main__":
    main()