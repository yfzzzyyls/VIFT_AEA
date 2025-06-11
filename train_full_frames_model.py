#!/usr/bin/env python3
"""
Train VIFT model with full frame data (all frames extracted)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime

# Import the model and dataset
from src.models.multihead_vio_separate_fixed import MultiHeadVIOModelSeparate
from src.data.components.aria_latent_dataset import AriaLatentDataset


class TrajectoryAwareLoss(nn.Module):
    """Loss function that encourages curved trajectories"""
    def __init__(self, frame_weight=1.0, trajectory_weight=0.1, 
                 smoothness_weight=0.1, diversity_weight=0.05):
        super().__init__()
        self.frame_weight = frame_weight
        self.trajectory_weight = trajectory_weight
        self.smoothness_weight = smoothness_weight
        self.diversity_weight = diversity_weight
        
    def forward(self, pred_trans, pred_rot, gt_trans, gt_rot, compute_trajectory=True):
        batch_size, seq_len = pred_trans.shape[:2]
        
        # Frame-to-frame loss (L1 for translation, geodesic for rotation)
        trans_loss = torch.mean(torch.abs(pred_trans - gt_trans))
        
        # Geodesic loss for rotation
        rot_loss = self.geodesic_loss(pred_rot, gt_rot)
        
        frame_loss = trans_loss + rot_loss
        
        if not compute_trajectory or seq_len < 3:
            return frame_loss
        
        # Trajectory coherence loss
        pred_positions = torch.cumsum(pred_trans, dim=1)
        gt_positions = torch.cumsum(gt_trans, dim=1)
        trajectory_loss = torch.mean(torch.abs(pred_positions - gt_positions))
        
        # Smoothness loss (penalize jerky motion)
        pred_accel = pred_trans[:, 2:] - 2 * pred_trans[:, 1:-1] + pred_trans[:, :-2]
        smoothness_loss = torch.mean(torch.abs(pred_accel))
        
        # Diversity loss (encourage curved motion)
        motion_vectors = pred_trans[:, 1:] - pred_trans[:, :-1]
        motion_directions = motion_vectors / (torch.norm(motion_vectors, dim=-1, keepdim=True) + 1e-6)
        dot_products = torch.sum(motion_directions[:, :-1] * motion_directions[:, 1:], dim=-1)
        diversity_loss = torch.mean(torch.abs(dot_products))  # Lower is more diverse
        
        total_loss = (self.frame_weight * frame_loss + 
                     self.trajectory_weight * trajectory_loss + 
                     self.smoothness_weight * smoothness_loss - 
                     self.diversity_weight * diversity_loss)
        
        return total_loss
    
    def geodesic_loss(self, pred_rot, gt_rot):
        """Compute geodesic distance between quaternions"""
        # Ensure unit quaternions
        pred_rot = pred_rot / (torch.norm(pred_rot, dim=-1, keepdim=True) + 1e-6)
        gt_rot = gt_rot / (torch.norm(gt_rot, dim=-1, keepdim=True) + 1e-6)
        
        # Compute dot product
        dot = torch.sum(pred_rot * gt_rot, dim=-1)
        dot = torch.clamp(dot, -1.0, 1.0)
        
        # Geodesic distance
        angle = 2 * torch.acos(torch.abs(dot))
        return torch.mean(angle)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_trans_error = 0
    total_rot_error = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move all batch data to device
        batch_gpu = {
            'visual_features': batch['visual_features'].to(device),
            'imu_features': batch['imu_features'].to(device),
            'poses': batch['poses'].to(device)
        }
        
        visual_features = batch_gpu['visual_features']
        imu_data = batch_gpu['imu_features']
        rel_poses = batch_gpu['poses']
        
        # Split poses (already in cm from dataset)
        gt_trans = rel_poses[..., :3]
        gt_rot = rel_poses[..., 3:]
        
        # Forward pass - model expects batch dictionary
        output = model(batch_gpu)
        pred_trans = output['translation']
        pred_rot = output['rotation']
        
        # Compute loss
        loss = criterion(pred_trans, pred_rot, gt_trans, gt_rot)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        with torch.no_grad():
            trans_error = torch.mean(torch.norm(pred_trans - gt_trans, dim=-1))
            rot_error = criterion.geodesic_loss(pred_rot, gt_rot)
            
            total_loss += loss.item()
            total_trans_error += trans_error.item()
            total_rot_error += rot_error.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'trans_err': f'{trans_error.item():.2f}cm',
                'rot_err': f'{rot_error.item():.3f}rad'
            })
    
    return {
        'loss': total_loss / num_batches,
        'trans_error': total_trans_error / num_batches,
        'rot_error': total_rot_error / num_batches
    }


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_trans_error = 0
    total_rot_error = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move all batch data to device
            batch_gpu = {
                'visual_features': batch['visual_features'].to(device),
                'imu_features': batch['imu_features'].to(device),
                'poses': batch['poses'].to(device)
            }
            
            visual_features = batch_gpu['visual_features']
            imu_data = batch_gpu['imu_features']
            rel_poses = batch_gpu['poses']
            
            gt_trans = rel_poses[..., :3]
            gt_rot = rel_poses[..., 3:]
            
            output = model(batch_gpu)
            pred_trans = output['translation']
            pred_rot = output['rotation']
            
            loss = criterion(pred_trans, pred_rot, gt_trans, gt_rot)
            trans_error = torch.mean(torch.norm(pred_trans - gt_trans, dim=-1))
            rot_error = criterion.geodesic_loss(pred_rot, gt_rot)
            
            total_loss += loss.item()
            total_trans_error += trans_error.item()
            total_rot_error += rot_error.item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'trans_error': total_trans_error / num_batches,
        'rot_error': total_rot_error / num_batches
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, 
                       default='/home/external/VIFT_AEA/aria_latent_full_frames')
    parser.add_argument('--checkpoint-dir', type=str, 
                       default='full_frames_checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    args = parser.parse_args()
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(args.checkpoint_dir) / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training config
    config = vars(args)
    config['timestamp'] = timestamp
    config['model'] = 'MultiHeadVIOModelSeparate'
    config['loss'] = 'TrajectoryAwareLoss'
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print(f"Loading data from {args.data_dir}")
    train_dataset = AriaLatentDataset(
        root_dir=os.path.join(args.data_dir, 'train')
    )
    
    val_dataset = AriaLatentDataset(
        root_dir=os.path.join(args.data_dir, 'val')
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = MultiHeadVIOModelSeparate(
        visual_dim=512,
        imu_dim=256,  # IMU features are 256-dim from encoder
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_shared_layers=4,
        num_specialized_layers=3,
        dropout=args.dropout,
        learning_rate=args.lr,
        sequence_length=10
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    criterion = TrajectoryAwareLoss(
        frame_weight=1.0,
        trajectory_weight=0.1,
        smoothness_weight=0.1,
        diversity_weight=0.05
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_trans_error = float('inf')
    patience_counter = 0
    
    print("\nStarting training with full frame data...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Trans: {train_metrics['trans_error']:.2f}cm, "
              f"Rot: {train_metrics['rot_error']:.3f}rad")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Trans: {val_metrics['trans_error']:.2f}cm, "
              f"Rot: {val_metrics['rot_error']:.3f}rad")
        
        # Save checkpoint if best
        if val_metrics['trans_error'] < best_trans_error:
            best_trans_error = val_metrics['trans_error']
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': config
            }
            
            checkpoint_path = checkpoint_dir / f'best_model_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"✅ Saved best model (trans error: {best_trans_error:.2f}cm)")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, checkpoint_path)
    
    print("\n" + "=" * 60)
    print(f"Training completed!")
    print(f"Best validation translation error: {best_trans_error:.2f}cm")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("\nNext steps:")
    print("1. Run inference on test sequences")
    print("2. Generate 3D trajectory plots")
    print("3. Analyze alignment between predictions and ground truth")


if __name__ == "__main__":
    main()