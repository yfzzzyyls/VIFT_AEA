#!/usr/bin/env python3
"""
Train VIFT from scratch on Aria Everyday Activities dataset.
This trains all components: Image Encoder + IMU Encoder + Pose Transformer.
No pre-trained weights are used.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.models.components.vsvio import TransformerVIO
from src.data.components.aria_raw_dataset import AriaRawDataModule
from train_vift_aria_stable import robust_geodesic_loss


class VIFTFromScratch(nn.Module):
    """VIFT model for training from scratch with quaternion output."""
    
    def __init__(self):
        super().__init__()
        
        # Model configuration
        class Config:
            # Sequence parameters
            seq_len = 11
            
            # Image parameters
            img_w = 512
            img_h = 256
            
            # Feature dimensions
            v_f_len = 512  # Visual feature dimension
            i_f_len = 256  # IMU feature dimension
            
            # IMU encoder parameters
            imu_dropout = 0.2
            
            # Transformer parameters
            embedding_dim = 768  # v_f_len + i_f_len
            num_layers = 4
            nhead = 8
            dim_feedforward = 2048
            dropout = 0.1
            
            # For compatibility
            rnn_hidden_size = 512
            rnn_dropout_between = 0.1
            rnn_dropout_out = 0.1
            fuse_method = 'cat'
        
        self.config = Config()
        self.backbone = TransformerVIO(self.config)
        
        # Replace output layer for quaternion output (7 values instead of 6)
        # Use the transformer's output directly for multi-step prediction
        hidden_dim = self.config.embedding_dim
        self.pose_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 7)  # 3 trans + 4 quat
        )
        
        # Initialize quaternion part to favor identity rotation
        with torch.no_grad():
            self.pose_predictor[-1].bias[3:6].fill_(0.0)  # qx, qy, qz = 0
            self.pose_predictor[-1].bias[6].fill_(1.0)    # qw = 1
            self.pose_predictor[-1].weight.data *= 0.01   # Small weights
    
    def forward(self, batch):
        """Forward pass."""
        images = batch['images']  # [B, 11, 3, 256, 512]
        imu = batch['imu']        # [B, 110, 6]
        
        # Get features from backbone
        fv, fi = self.backbone.Feature_net(images, imu)  # fv: [B, 11, 512], fi: [B, 11, 256]
        
        # Concatenate visual and inertial features
        combined_features = torch.cat([fv, fi], dim=-1)  # [B, 11, 768]
        
        # Predict poses for all transitions (first 10 timesteps)
        # Each timestep predicts the transition to the next frame
        transition_features = combined_features[:, :10, :]  # [B, 10, 768]
        
        # Apply pose predictor to each timestep
        batch_size, seq_len, feat_dim = transition_features.shape
        transition_features_flat = transition_features.reshape(-1, feat_dim)  # [B*10, 768]
        poses_flat = self.pose_predictor(transition_features_flat)  # [B*10, 7]
        poses = poses_flat.reshape(batch_size, seq_len, 7)  # [B, 10, 7]
        
        # Normalize quaternion part for each pose
        trans = poses[:, :, :3]  # [B, 10, 3]
        quat = poses[:, :, 3:]   # [B, 10, 4]
        quat = F.normalize(quat, p=2, dim=-1)
        
        # Combine and return
        normalized_poses = torch.cat([trans, quat], dim=-1)  # [B, 10, 7]
        return {
            'poses': normalized_poses  # [B, 10, 7]
        }


def compute_loss(predictions, batch, trans_weight=100.0, rot_weight=1.0):
    """Compute loss with quaternion representation for multi-step prediction"""
    pred_poses = predictions['poses']  # [B, 10, 7]
    gt_poses = batch['gt_poses']  # [B, 10, 7]
    
    # Split predictions and ground truth
    pred_trans = pred_poses[:, :, :3]  # [B, 10, 3]
    pred_rot = pred_poses[:, :, 3:]    # [B, 10, 4]
    gt_trans = gt_poses[:, :, :3]      # [B, 10, 3]
    gt_rot = gt_poses[:, :, 3:]        # [B, 10, 4]
    
    # Translation loss
    trans_loss = F.smooth_l1_loss(pred_trans, gt_trans)
    
    # Rotation loss using geodesic distance
    rot_loss = robust_geodesic_loss(pred_rot, gt_rot)
    
    if torch.isnan(trans_loss) or torch.isnan(rot_loss):
        return None
    
    # Combined loss
    total_loss = trans_weight * trans_loss + rot_weight * rot_loss
    
    return {
        'total_loss': total_loss,
        'trans_loss': trans_loss,
        'rot_loss': rot_loss
    }


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        batch = {
            'images': batch['images'].to(device),
            'imu': batch['imu'].to(device).float(),
            'gt_poses': batch['gt_poses'].to(device)
        }
        
        # Forward pass
        predictions = model(batch)
        
        # Compute loss
        loss_dict = compute_loss(predictions, batch)
        
        if loss_dict is None:
            print(f"NaN loss detected at batch {batch_idx}, skipping...")
            continue
        
        loss = loss_dict['total_loss']
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"NaN loss detected at batch {batch_idx}, skipping...")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'trans': f"{loss_dict['trans_loss'].item():.4f}",
            'rot': f"{loss_dict['rot_loss'].item():.4f}"
        })
    
    return total_loss / num_batches if num_batches > 0 else float('inf')


def validate(model, dataloader, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_trans_error = 0
    total_rot_error = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move to device
            batch = {
                'images': batch['images'].to(device),
                'imu': batch['imu'].to(device).float(),
                'gt_poses': batch['gt_poses'].to(device)
            }
            
            # Forward pass
            predictions = model(batch)
            
            # Compute loss
            loss_dict = compute_loss(predictions, batch)
            
            if loss_dict is None:
                continue
            
            # Compute errors for multi-step prediction
            pred_poses = predictions['poses']  # [B, 10, 7]
            gt_poses = batch['gt_poses']  # [B, 10, 7]
            
            # Translation error
            pred_trans = pred_poses[:, :, :3]  # [B, 10, 3]
            gt_trans = gt_poses[:, :, :3]      # [B, 10, 3]
            trans_error = torch.norm(pred_trans - gt_trans, dim=-1).mean()
            
            # Rotation error
            pred_rot = pred_poses[:, :, 3:]  # [B, 10, 4]
            gt_rot = gt_poses[:, :, 3:]      # [B, 10, 4]
            pred_rot = F.normalize(pred_rot, p=2, dim=-1)
            gt_rot = F.normalize(gt_rot, p=2, dim=-1)
            
            # Compute angle between quaternions
            dot_product = torch.sum(pred_rot * gt_rot, dim=-1).abs()  # [B, 10]
            dot_product = torch.clamp(dot_product, -1.0, 1.0)
            angle_error = 2 * torch.acos(dot_product)
            rot_error = torch.rad2deg(angle_error).mean()
            
            total_loss += loss_dict['total_loss'].item()
            total_trans_error += trans_error.item()
            total_rot_error += rot_error.item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches if num_batches > 0 else float('inf'),
        'trans_error': total_trans_error / num_batches if num_batches > 0 else float('inf'),
        'rot_error': total_rot_error / num_batches if num_batches > 0 else float('inf')
    }


def main():
    parser = argparse.ArgumentParser(description='Train VIFT from scratch on Aria data')
    parser.add_argument('--data-dir', type=str, default='aria_processed',
                        help='Directory with processed Aria data')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_from_scratch',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--num-gpus', type=int, default=4,
                        help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Check available GPUs
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {n_gpus}")
        if args.num_gpus > n_gpus:
            print(f"Warning: Requested {args.num_gpus} GPUs but only {n_gpus} available")
            args.num_gpus = n_gpus
    else:
        print("No GPUs available, using CPU")
        args.num_gpus = 0
    
    print(f"Using {args.num_gpus} GPU(s)")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Training VIFT from Scratch on Aria Data")
    print("="*60)
    print("Training all components:")
    print("- Image Encoder (6-layer CNN)")
    print("- IMU Encoder (3-layer 1D CNN)")
    print("- Pose Transformer (4 layers, 8 heads)")
    print("- Quaternion output (3 trans + 4 quat)")
    print("- Multi-step prediction (all 10 transitions)")
    print(f"\nTraining Configuration:")
    print(f"- GPUs: {args.num_gpus}")
    print(f"- Batch size per GPU: {args.batch_size}")
    print(f"- Total batch size: {args.batch_size * max(1, args.num_gpus)}")
    print(f"- Learning rate: {args.lr}")
    print(f"- Window stride: 10 (non-overlapping transitions)")
    print("="*60 + "\n")
    
    # Create data module with adjusted batch size for multi-GPU
    effective_batch_size = args.batch_size * max(1, args.num_gpus)
    data_module = AriaRawDataModule(
        data_dir=args.data_dir,
        batch_size=effective_batch_size,
        num_workers=args.num_workers * max(1, args.num_gpus),  # Scale workers with GPUs
        sequence_length=11,
        stride=10  # Non-overlapping windows since we predict all 10 transitions
    )
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"Train samples: {len(data_module.train_dataset)}")
    print(f"Val samples: {len(data_module.val_dataset)}")
    
    # Create model
    model = VIFTFromScratch()
    
    # Move to GPU and wrap with DataParallel if using multiple GPUs
    if args.num_gpus > 1:
        model = model.to(device)
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpus)))
        print(f"Using DataParallel with {args.num_gpus} GPUs")
    else:
        model = model.to(device)
    
    # Count parameters
    base_model = model.module if hasattr(model, 'module') else model
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Warmup scheduler (like train_efficient.py)
    warmup_steps = len(train_loader) * 2  # 2 epochs of warmup
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        total_iters=warmup_steps
    )
    
    # Cosine annealing after warmup
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - 2, eta_min=1e-6
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Step scheduler
        if epoch <= 2:
            # Warmup phase
            warmup_scheduler.step()
        else:
            # Cosine annealing phase
            cosine_scheduler.step()
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_metrics['loss']:.6f}")
        print(f"  Val Trans Error: {val_metrics['trans_error']*100:.2f} cm")
        print(f"  Val Rot Error: {val_metrics['rot_error']:.2f}°")
        
        # Save checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            # Get the base model state dict (unwrap DataParallel if needed)
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            base_model = model.module if hasattr(model, 'module') else model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': warmup_scheduler.state_dict() if epoch <= 2 else cosine_scheduler.state_dict(),
                'train_loss': train_loss,
                'val_metrics': val_metrics,
                'config': base_model.config.__dict__
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            print("  ✓ New best model saved")
        
        # Save latest checkpoint
        checkpoint['epoch'] = epoch
        checkpoint['train_loss'] = train_loss
        checkpoint['val_metrics'] = val_metrics
        torch.save(checkpoint, checkpoint_dir / 'latest_model.pt')
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()