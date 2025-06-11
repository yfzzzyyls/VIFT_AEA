#!/usr/bin/env python3
"""
Train VIFT model with fixed gradient flow for rotation prediction
Key changes:
1. Better initialization for rotation layers
2. Adjusted loss weighting to balance gradients
3. Separate learning rates for translation and rotation
4. Gradient scaling for rotation components
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.models.multihead_vio_separate_fixed import MultiHeadVIOModelSeparate
from src.data.components.aria_latent_dataset import AriaLatentDataset


class BalancedMotionLoss(nn.Module):
    """Loss function with balanced gradients for translation and rotation"""
    def __init__(self, trans_weight=1.0, rot_weight=1.0):
        super().__init__()
        self.trans_weight = trans_weight
        self.rot_weight = rot_weight
        
    def forward(self, pred_trans, pred_rot, gt_trans, gt_rot):
        # Translation loss - scale by expected magnitude (~0.5cm)
        trans_diff = pred_trans - gt_trans
        trans_loss = torch.mean(torch.abs(trans_diff))
        
        # Rotation loss with gradient-friendly formulation
        # First normalize quaternions
        pred_rot_norm = pred_rot / (torch.norm(pred_rot, dim=-1, keepdim=True) + 1e-8)
        gt_rot_norm = gt_rot / (torch.norm(gt_rot, dim=-1, keepdim=True) + 1e-8)
        
        # Use MSE on normalized quaternions for better gradient flow
        # This avoids the acos function which can have gradient issues
        rot_loss = torch.mean((pred_rot_norm - gt_rot_norm) ** 2)
        
        # Add regularization to prevent collapse to identity
        # Penalize if all rotations are too similar
        rot_variance = torch.var(pred_rot_norm.view(-1, 4), dim=0).mean()
        variance_penalty = torch.relu(0.001 - rot_variance) * 10
        
        # Total loss with balanced weights
        total_loss = (self.trans_weight * trans_loss + 
                     self.rot_weight * rot_loss +
                     variance_penalty)
        
        return total_loss, trans_loss, rot_loss, variance_penalty


def initialize_model_with_gradient_fix(model):
    """Initialize model to ensure good gradient flow"""
    # Initialize all layers with careful scaling
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'rotation' in name:
                # Larger initialization for rotation layers to prevent vanishing gradients
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param, gain=2.0)
                else:
                    nn.init.normal_(param, std=0.1)
            elif 'translation' in name:
                # Standard initialization for translation
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param, gain=1.0)
                else:
                    nn.init.normal_(param, std=0.02)
            else:
                # Default initialization
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            if 'rotation_output' in name and param.shape[0] == 4:
                # Initialize with small random rotations instead of identity
                param.data = torch.randn(4, device=param.device) * 0.1
                param.data[3] = 1.0  # Keep w component close to 1
            elif 'translation_output' in name and param.shape[0] == 3:
                # Small forward motion bias
                param.data = torch.tensor([0.0, 0.0, 0.1], device=param.device)
            else:
                nn.init.zeros_(param)
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, print_freq=20, log_file=None):
    """Train for one epoch with detailed gradient monitoring"""
    model.train()
    total_loss = 0
    total_trans_loss = 0
    total_rot_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        # Move all batch data to device
        batch_gpu = {
            'visual_features': batch['visual_features'].to(device),
            'imu_features': batch['imu_features'].to(device),
            'poses': batch['poses'].to(device)
        }
        
        gt_poses = batch_gpu['poses']
        gt_trans = gt_poses[..., :3]
        gt_rot = gt_poses[..., 3:]
        
        # Forward pass
        output = model(batch_gpu)
        pred_trans = output['translation']
        pred_rot = output['rotation']
        
        # Compute loss with component tracking
        loss, trans_loss, rot_loss, var_penalty = criterion(pred_trans, pred_rot, gt_trans, gt_rot)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping with different thresholds
        # Clip translation and rotation gradients separately
        trans_params = []
        rot_params = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'translation' in name:
                    trans_params.append(param)
                elif 'rotation' in name:
                    rot_params.append(param)
        
        if trans_params:
            torch.nn.utils.clip_grad_norm_(trans_params, max_norm=5.0)
        if rot_params:
            torch.nn.utils.clip_grad_norm_(rot_params, max_norm=10.0)  # Allow larger gradients for rotation
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_trans_loss += trans_loss.item()
        total_rot_loss += rot_loss.item()
        num_batches += 1
        
        # Print diagnostics
        if batch_idx % print_freq == 0 and batch_idx > 0:
            with torch.no_grad():
                # Check gradient magnitudes
                rot_grad_norm = 0
                trans_grad_norm = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if 'rotation' in name:
                            rot_grad_norm += param.grad.norm().item() ** 2
                        elif 'translation' in name:
                            trans_grad_norm += param.grad.norm().item() ** 2
                rot_grad_norm = rot_grad_norm ** 0.5
                trans_grad_norm = trans_grad_norm ** 0.5
                
                log_message = f"\n{'='*80}\n"
                log_message += f"Epoch {epoch}, Batch {batch_idx} Diagnostics\n"
                log_message += f"{'='*80}\n"
                
                # Loss components
                log_message += f"Loss Components:\n"
                log_message += f"  Total Loss: {loss.item():.6f}\n"
                log_message += f"  Trans Loss: {trans_loss.item():.6f}\n"
                log_message += f"  Rot Loss:   {rot_loss.item():.6f}\n"
                log_message += f"  Var Penalty:{var_penalty.item():.6f}\n"
                
                # Gradient norms
                log_message += f"\nGradient Norms:\n"
                log_message += f"  Translation: {trans_grad_norm:.6f}\n"
                log_message += f"  Rotation:    {rot_grad_norm:.6f}\n"
                
                # Prediction statistics
                pred_rot_var = torch.var(pred_rot.view(-1, 4), dim=0)
                log_message += f"\nRotation Variance: [{pred_rot_var[0]:.6f}, {pred_rot_var[1]:.6f}, {pred_rot_var[2]:.6f}, {pred_rot_var[3]:.6f}]\n"
                
                # Sample predictions
                log_message += f"\nSample Predictions (first 3 frames):\n"
                log_message += f"{'Frame':<6} {'GT Trans (cm)':<20} {'GT Rot':<25} {'Pred Trans (cm)':<20} {'Pred Rot':<25}\n"
                log_message += "-" * 100 + "\n"
                
                for i in range(min(3, pred_trans.shape[1])):
                    gt_t = gt_trans[0, i].cpu().numpy()
                    gt_r = gt_rot[0, i].cpu().numpy()
                    pred_t = pred_trans[0, i].cpu().numpy()
                    pred_r = pred_rot[0, i].cpu().numpy()
                    
                    log_message += (f"{i:<6} [{gt_t[0]:6.3f},{gt_t[1]:6.3f},{gt_t[2]:6.3f}] "
                                  f"[{gt_r[0]:5.3f},{gt_r[1]:5.3f},{gt_r[2]:5.3f},{gt_r[3]:5.3f}] "
                                  f"[{pred_t[0]:6.3f},{pred_t[1]:6.3f},{pred_t[2]:6.3f}] "
                                  f"[{pred_r[0]:5.3f},{pred_r[1]:5.3f},{pred_r[2]:5.3f},{pred_r[3]:5.3f}]\n")
                
                print(log_message)
                if log_file:
                    log_file.write(log_message)
                    log_file.flush()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'trans': f"{trans_loss.item():.4f}",
            'rot': f"{rot_loss.item():.4f}"
        })
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_trans_error = 0
    total_rot_error = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch_gpu = {
                'visual_features': batch['visual_features'].to(device),
                'imu_features': batch['imu_features'].to(device),
                'poses': batch['poses'].to(device)
            }
            
            gt_poses = batch_gpu['poses']
            gt_trans = gt_poses[..., :3]
            gt_rot = gt_poses[..., 3:]
            
            # Forward pass
            output = model(batch_gpu)
            pred_trans = output['translation']
            pred_rot = output['rotation']
            
            # Compute loss
            loss, _, _, _ = criterion(pred_trans, pred_rot, gt_trans, gt_rot)
            
            # Compute errors
            trans_error = torch.mean(torch.norm(pred_trans - gt_trans, dim=-1))
            
            # Rotation error using quaternion distance
            pred_rot_norm = pred_rot / (torch.norm(pred_rot, dim=-1, keepdim=True) + 1e-8)
            gt_rot_norm = gt_rot / (torch.norm(gt_rot, dim=-1, keepdim=True) + 1e-8)
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


def main():
    # Configuration
    config = {
        'data_dir': '/home/external/VIFT_AEA/aria_latent_full_frames',
        'checkpoint_dir': 'checkpoints_gradient_fix',
        'epochs': 50,
        'batch_size': 8,
        'base_lr': 1e-3,
        'rot_lr_scale': 5.0,  # Higher learning rate for rotation
        'hidden_dim': 128,
        'num_heads': 4,
        'dropout': 0.1,
        'print_freq': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(config['checkpoint_dir']) / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file
    log_file_path = checkpoint_dir / 'training_log_gradient_fix.txt'
    log_file = open(log_file_path, 'w')
    
    # Save config
    config['timestamp'] = timestamp
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Log initial info
    log_message = f"Training VIFT with Gradient Flow Fix\n"
    log_message += f"{'='*80}\n"
    log_message += f"Timestamp: {timestamp}\n"
    log_message += f"Config: {json.dumps(config, indent=2)}\n"
    log_message += f"{'='*80}\n\n"
    print(log_message)
    log_file.write(log_message)
    
    # Setup device
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Load datasets
    train_dataset = AriaLatentDataset(
        root_dir=os.path.join(config['data_dir'], 'train')
    )
    
    val_dataset = AriaLatentDataset(
        root_dir=os.path.join(config['data_dir'], 'val')
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model with gradient-friendly initialization
    model = MultiHeadVIOModelSeparate(
        visual_dim=512,
        imu_dim=256,
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        dropout=config['dropout']
    ).to(device)
    
    # Apply special initialization
    model = initialize_model_with_gradient_fix(model)
    
    # Create parameter groups with different learning rates
    trans_params = []
    rot_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'translation' in name:
            trans_params.append(param)
        elif 'rotation' in name:
            rot_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {'params': trans_params, 'lr': config['base_lr']},
        {'params': rot_params, 'lr': config['base_lr'] * config['rot_lr_scale']},
        {'params': other_params, 'lr': config['base_lr']}
    ]
    
    # Loss and optimizer
    criterion = BalancedMotionLoss(trans_weight=1.0, rot_weight=10.0)  # Higher weight for rotation
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        log_message = f"\n{'='*80}\n"
        log_message += f"EPOCH {epoch + 1}/{config['epochs']}\n"
        log_message += f"{'='*80}\n"
        print(log_message)
        log_file.write(log_message)
        log_file.flush()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, 
                               device, epoch + 1, config['print_freq'], log_file)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Log epoch summary
        log_message = f"\nEpoch {epoch + 1} Summary:\n"
        log_message += f"  Train Loss: {train_loss:.6f}\n"
        log_message += f"  Val Loss: {val_metrics['loss']:.6f}\n"
        log_message += f"  Val Trans Error: {val_metrics['trans_error']:.6f} cm\n"
        log_message += f"  Val Rot Error: {val_metrics['rot_error']:.6f} rad ({np.degrees(val_metrics['rot_error']):.2f} deg)\n"
        print(log_message)
        log_file.write(log_message)
        log_file.flush()
        
        # Save checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_metrics': val_metrics,
                'config': config
            }
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ New best model saved (val_loss: {best_val_loss:.6f})")
    
    log_message = f"\n{'='*80}\n"
    log_message += f"Training Complete!\n"
    log_message += f"Best validation loss: {best_val_loss:.6f}\n"
    log_message += f"Logs saved to: {log_file_path}\n"
    log_message += f"{'='*80}\n"
    print(log_message)
    log_file.write(log_message)
    log_file.close()

if __name__ == "__main__":
    main()