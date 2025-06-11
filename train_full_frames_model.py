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


class RelativeMotionLoss(nn.Module):
    """Fixed loss function for relative motion prediction"""
    def __init__(self, trans_weight=1.0, rot_weight=0.1, 
                 min_motion=5.0, target_motion=50.0):
        super().__init__()
        self.trans_weight = trans_weight
        self.rot_weight = rot_weight
        self.min_motion = min_motion  # Minimum expected motion in cm
        self.target_motion = target_motion  # Target average motion in cm
        
    def forward(self, pred_trans, pred_rot, gt_trans, gt_rot):
        # Translation loss - keep in cm but normalize by expected scale
        trans_loss = torch.mean(torch.abs(pred_trans - gt_trans)) / 100.0
        
        # Geodesic loss for rotation
        pred_rot_norm = pred_rot / (torch.norm(pred_rot, dim=-1, keepdim=True) + 1e-6)
        gt_rot_norm = gt_rot / (torch.norm(gt_rot, dim=-1, keepdim=True) + 1e-6)
        dot = torch.sum(pred_rot_norm * gt_rot_norm, dim=-1)
        dot = torch.clamp(dot, -1.0, 1.0)
        rot_loss = torch.mean(2 * torch.acos(torch.abs(dot)))
        
        # Motion encouragement - penalize predictions below minimum motion
        pred_motion_mag = torch.norm(pred_trans, dim=-1)
        
        # Penalize if motion is too small
        small_motion_penalty = torch.mean(torch.relu(self.min_motion - pred_motion_mag)) / 10.0
        
        # Soft constraint on motion magnitude
        motion_deviation = torch.mean((pred_motion_mag - self.target_motion).pow(2)) / 10000.0
        
        # Total loss
        total_loss = (self.trans_weight * trans_loss + 
                     self.rot_weight * rot_loss + 
                     0.1 * small_motion_penalty + 
                     0.01 * motion_deviation)
        
        return total_loss


def train_epoch(model, dataloader, optimizer, criterion, device, epoch=0, print_freq=50, scheduler=None):
    model.train()
    total_loss = 0
    total_trans_error = 0
    total_rot_error = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
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
        
        # Print diagnostics every print_freq batches
        if batch_idx % print_freq == 0 and batch_idx > 0:
            with torch.no_grad():
                print(f"\n{'='*60}")
                print(f"Epoch {epoch}, Batch {batch_idx} Diagnostics")
                print(f"{'='*60}")
                
                # Sample statistics
                pred_mean = pred_trans.mean(dim=(0,1)).cpu().numpy()
                pred_std = pred_trans.std(dim=(0,1)).cpu().numpy()
                gt_mean = gt_trans.mean(dim=(0,1)).cpu().numpy()
                gt_std = gt_trans.std(dim=(0,1)).cpu().numpy()
                
                print(f"GT   mean: [{gt_mean[0]:7.2f}, {gt_mean[1]:7.2f}, {gt_mean[2]:7.2f}] cm")
                print(f"GT   std:  [{gt_std[0]:7.2f}, {gt_std[1]:7.2f}, {gt_std[2]:7.2f}] cm")
                print(f"Pred mean: [{pred_mean[0]:7.2f}, {pred_mean[1]:7.2f}, {pred_mean[2]:7.2f}] cm")
                print(f"Pred std:  [{pred_std[0]:7.2f}, {pred_std[1]:7.2f}, {pred_std[2]:7.2f}] cm")
                
                # Check motion magnitudes
                gt_mag = torch.norm(gt_trans, dim=-1).mean().item()
                pred_mag = torch.norm(pred_trans, dim=-1).mean().item()
                print(f"\nMotion magnitude - GT: {gt_mag:.2f} cm/frame, Pred: {pred_mag:.2f} cm/frame")
                print(f"Magnitude ratio (pred/gt): {pred_mag/gt_mag:.3f}")
                
                # Sample predictions (first sample, first 3 frames)
                print(f"\nSample 7-DoF predictions (first 3 frames):")
                print(f"{'Frame':<6} {'GT Translation (cm)':<30} {'GT Rotation (quat)':<30} {'Pred Translation (cm)':<30} {'Pred Rotation (quat)':<30}")
                print("-" * 130)
                for i in range(min(3, pred_trans.shape[1])):
                    gt_t = gt_trans[0, i].cpu().numpy()
                    gt_r = gt_rot[0, i].cpu().numpy()
                    pred_t = pred_trans[0, i].cpu().numpy()
                    pred_r = pred_rot[0, i].cpu().numpy()
                    print(f"{i:<6} [{gt_t[0]:7.2f}, {gt_t[1]:7.2f}, {gt_t[2]:7.2f}] "
                          f"[{gt_r[0]:6.3f}, {gt_r[1]:6.3f}, {gt_r[2]:6.3f}, {gt_r[3]:6.3f}] "
                          f"[{pred_t[0]:7.2f}, {pred_t[1]:7.2f}, {pred_t[2]:7.2f}] "
                          f"[{pred_r[0]:6.3f}, {pred_r[1]:6.3f}, {pred_r[2]:6.3f}, {pred_r[3]:6.3f}]")
        
        # Compute loss
        loss = criterion(pred_trans, pred_rot, gt_trans, gt_rot)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        # Step scheduler if provided (for warmup)
        if scheduler is not None:
            scheduler.step()
        
        # Track metrics
        with torch.no_grad():
            trans_error = torch.mean(torch.norm(pred_trans - gt_trans, dim=-1))
            # Compute rotation error manually since we changed loss class
            pred_rot_norm = pred_rot / (torch.norm(pred_rot, dim=-1, keepdim=True) + 1e-6)
            gt_rot_norm = gt_rot / (torch.norm(gt_rot, dim=-1, keepdim=True) + 1e-6)
            dot = torch.sum(pred_rot_norm * gt_rot_norm, dim=-1)
            dot = torch.clamp(dot, -1.0, 1.0)
            rot_error = torch.mean(2 * torch.acos(torch.abs(dot)))
            
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
            
            # Compute rotation error
            pred_rot_norm = pred_rot / (torch.norm(pred_rot, dim=-1, keepdim=True) + 1e-6)
            gt_rot_norm = gt_rot / (torch.norm(gt_rot, dim=-1, keepdim=True) + 1e-6)
            dot = torch.sum(pred_rot_norm * gt_rot_norm, dim=-1)
            dot = torch.clamp(dot, -1.0, 1.0)
            rot_error = torch.mean(2 * torch.acos(torch.abs(dot)))
            
            total_loss += loss.item()
            total_trans_error += trans_error.item()
            total_rot_error += rot_error.item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'trans_error': total_trans_error / num_batches,
        'rot_error': total_rot_error / num_batches
    }


def initialize_model_properly(model):
    """Initialize model weights to encourage motion prediction"""
    # Initialize all layers
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:
                # Larger initialization for better gradient flow
                nn.init.xavier_uniform_(param, gain=2.0)
            else:
                nn.init.normal_(param, std=0.1)
        elif 'bias' in name:
            # Special handling for output biases
            if 'translation' in name and param.shape[0] == 3:
                # Initialize to predict small forward motion by default
                param.data = torch.tensor([0.0, 0.0, 5.0], device=param.device)
            elif 'rotation' in name and param.shape[0] == 4:
                # Initialize rotation bias close to identity quaternion
                param.data = torch.tensor([0.0, 0.0, 0.0, 1.0], device=param.device)
            else:
                nn.init.zeros_(param)
    
    # Find translation output layer and give it special initialization
    if hasattr(model, 'translation_head'):
        for module in model.translation_head.modules():
            if isinstance(module, nn.Linear) and module.out_features == 3:
                # Larger weight initialization for translation output
                nn.init.normal_(module.weight, mean=0, std=0.5)
                # Set bias to encourage forward motion
                module.bias.data = torch.tensor([0.0, 0.0, 5.0], device=module.bias.device)
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, 
                       default='/home/external/VIFT_AEA/aria_latent_full_frames')
    parser.add_argument('--checkpoint-dir', type=str, 
                       default='full_frames_checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)  # Small batch for stronger gradients
    parser.add_argument('--lr', type=float, default=1e-3)  # Higher learning rate to escape local minimum
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--print-freq', type=int, default=20)  # More frequent diagnostics
    parser.add_argument('--grad-clip', type=float, default=5.0)  # Allow larger gradient updates
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
    
    # Properly initialize weights to avoid constant predictions
    model = initialize_model_properly(model)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training with fixed loss that encourages motion
    criterion = RelativeMotionLoss(
        trans_weight=1.0,
        rot_weight=0.1,
        min_motion=5.0,      # Expect at least 5cm motion
        target_motion=50.0   # Target ~50cm average motion based on data
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Use warmup scheduler for stable training
    num_warmup_steps = len(train_loader) * 2  # 2 epochs warmup
    num_training_steps = len(train_loader) * args.epochs
    
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    best_val_loss = float('inf')
    best_trans_error = float('inf')
    patience_counter = 0
    
    print("\nStarting training with full frame data...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # Train with epoch number for diagnostics
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, 
                                  epoch=epoch+1, print_freq=args.print_freq, scheduler=scheduler)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler (step after each batch in train_epoch for warmup)
        # Note: scheduler step is called in train loop for warmup schedule
        
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