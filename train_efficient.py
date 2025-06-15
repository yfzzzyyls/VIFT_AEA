#!/usr/bin/env python3
"""
Efficient VIFT training using the existing pre-extracted features.
Uses the fixed AriaLatentDataset without z-score normalization.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.data.components.aria_latent_dataset import AriaLatentDataset
from train_vift_aria_stable import VIFTStable, robust_geodesic_loss


def compute_loss(predictions, batch, trans_weight=100.0, rot_weight=1.0):
    """Compute loss with proper scaling for multi-step prediction"""
    pred_poses = predictions['poses']  # [B, 10, 7]
    gt_poses = batch['poses']  # [B, seq_len, 7]
    
    # For multi-step prediction, use all 10 transitions
    # Note: gt_poses might be [B, 11, 7] or [B, 10, 7] depending on dataset
    if gt_poses.shape[1] > 10:
        gt_poses = gt_poses[:, :10, :]  # Use first 10 transitions
    
    pred_trans = pred_poses[:, :, :3]
    pred_rot = pred_poses[:, :, 3:]
    gt_trans = gt_poses[:, :, :3]
    gt_rot = gt_poses[:, :, 3:]
    
    # Translation loss in meters
    trans_loss = F.smooth_l1_loss(pred_trans, gt_trans)
    
    # Rotation loss in radians
    rot_loss = robust_geodesic_loss(pred_rot, gt_rot)
    
    if torch.isnan(trans_loss) or torch.isnan(rot_loss):
        return None
    
    # Combined loss with proper scaling
    total_loss = trans_weight * trans_loss + rot_weight * rot_loss
    
    return {
        'total_loss': total_loss,
        'trans_loss': trans_loss,
        'rot_loss': rot_loss
    }


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        batch_gpu = {
            'visual_features': batch['visual_features'].to(device),
            'imu_features': batch['imu_features'].to(device),
            'poses': batch['poses'].to(device)
        }
        
        # Debug first batch
        if batch_idx == 0 and epoch == 1:
            print(f"\nFirst batch shapes:")
            print(f"  Visual: {batch_gpu['visual_features'].shape}")
            print(f"  IMU: {batch_gpu['imu_features'].shape}")
            print(f"  Poses: {batch_gpu['poses'].shape}")
            print(f"  GT translation scale: {batch_gpu['poses'][:, :, :3].abs().mean().item():.6f} m")
        
        predictions = model(batch_gpu)
        loss_dict = compute_loss(predictions, batch_gpu)
        
        if loss_dict is None:
            continue
            
        loss = loss_dict['total_loss']
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'trans': f"{loss_dict['trans_loss'].item():.4f}",
            'rot': f"{loss_dict['rot_loss'].item():.4f}"
        })
    
    return total_loss / num_batches if num_batches > 0 else float('inf')


def validate(model, dataloader, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_trans_error = 0
    total_rot_error_deg = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch_gpu = {
                'visual_features': batch['visual_features'].to(device),
                'imu_features': batch['imu_features'].to(device),
                'poses': batch['poses'].to(device)
            }
            
            predictions = model(batch_gpu)
            loss_dict = compute_loss(predictions, batch_gpu)
            
            if loss_dict is None:
                continue
            
            # Translation error in meters (average over all timesteps)
            pred_trans = predictions['translation']  # [B, 10, 3]
            gt_trans = batch_gpu['poses'][:, :10, :3]  # [B, 10, 3]
            trans_error = torch.mean(torch.norm(pred_trans - gt_trans, dim=-1))
            
            # Rotation error in degrees
            rot_error_deg = torch.rad2deg(loss_dict['rot_loss'])
            
            total_loss += loss_dict['total_loss'].item()
            total_trans_error += trans_error.item()
            total_rot_error_deg += rot_error_deg.item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches if num_batches > 0 else float('inf'),
        'trans_error': total_trans_error / num_batches if num_batches > 0 else float('inf'),
        'rot_error_deg': total_rot_error_deg / num_batches if num_batches > 0 else float('inf')
    }


def main():
    parser = argparse.ArgumentParser(description='Efficient VIFT Training')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--data-dir', type=str, default='aria_latent')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_efficient')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("VIFT Efficient Training")
    print("="*60)
    print("Using pre-extracted features with fixed dataset:")
    print("- No z-score normalization on translations")
    print("- Direct meter-scale training")
    print("- Multi-step prediction (all 10 transitions)")
    print("- Stride=10 for non-overlapping windows")
    print("- Consistent with evaluation pipeline")
    print("="*60 + "\n")
    
    # Load datasets with the fixed AriaLatentDataset
    print("Loading datasets...")
    train_dataset = AriaLatentDataset(os.path.join(args.data_dir, 'train'))
    val_dataset = AriaLatentDataset(os.path.join(args.data_dir, 'val'))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    model = VIFTStable().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Warmup scheduler
    warmup_steps = len(train_loader) * 2  # 2 epochs of warmup
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        total_iters=warmup_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch + 1)
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Step scheduler during warmup
        if epoch < 2:
            scheduler.step()
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_metrics['loss']:.6f}")
        print(f"  Val Trans Error: {val_metrics['trans_error']*100:.2f} cm")
        print(f"  Val Rot Error: {val_metrics['rot_error_deg']:.2f}°")
        
        # Save checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_metrics': val_metrics,
                'config': {
                    'architecture': 'VIFTStable',
                    'features': 'pre-extracted, no z-score normalization'
                }
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            print("  ✓ New best model saved")
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest_model.pt')
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()