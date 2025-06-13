#!/usr/bin/env python3
"""
Stable VIFT training with input normalization and gradient monitoring.
"""

import os
import sys
import json
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


class InputNormalization(nn.Module):
    """Normalize inputs to prevent gradient explosion"""
    def __init__(self, visual_dim=512, imu_dim=256):
        super().__init__()
        # Learnable normalization parameters
        self.visual_scale = nn.Parameter(torch.ones(visual_dim) * 1.0)
        self.visual_bias = nn.Parameter(torch.zeros(visual_dim))
        self.imu_scale = nn.Parameter(torch.ones(imu_dim) * 1.0)  # IMU has larger values
        self.imu_bias = nn.Parameter(torch.zeros(imu_dim))
    
    def forward(self, visual_features, imu_features):
        # Normalize features
        visual_norm = visual_features * self.visual_scale + self.visual_bias
        imu_norm = imu_features * self.imu_scale + self.imu_bias
        return visual_norm, imu_norm


class VIFTStable(nn.Module):
    """Stable VIFT model with normalization and careful initialization"""
    
    def __init__(self, input_dim=768, embedding_dim=256, num_layers=4, 
                 nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        # Input normalization
        self.input_norm = InputNormalization()
        
        # Create transformer components
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),  # Add layer norm for stability
            nn.GELU()
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embedding_dim, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True  # Pre-norm for stability
            ), 
            num_layers=num_layers
        )
        
        # Direct 7DoF output with careful initialization
        self.fc2 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, 7)
        )
        
        # Proper initialization
        with torch.no_grad():
            # Initialize the output layer
            nn.init.xavier_uniform_(self.fc2[-1].weight)
            
            # Initialize the first layer with proper gain
            nn.init.xavier_uniform_(self.fc1[0].weight, gain=1.0)
    
    def positional_embedding(self, seq_length, device):
        import math
        pos = torch.arange(0, seq_length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2, device=device).float() * 
                           -(math.log(10000.0) / self.embedding_dim))
        pos_embedding = torch.zeros(seq_length, self.embedding_dim, device=device)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        return pos_embedding.unsqueeze(0)
    
    def generate_square_subsequent_mask(self, sz, device):
        """Generate a square causal mask for sequence."""
        return torch.triu(
            torch.full((sz, sz), float("-inf"), device=device),
            diagonal=1
        )
    
    def forward(self, batch):
        # Extract and normalize inputs
        visual_features = batch['visual_features']
        imu_features = batch['imu_features']
        
        # Apply input normalization
        visual_norm, imu_norm = self.input_norm(visual_features, imu_features)
        
        # Combine normalized features
        combined = torch.cat([visual_norm, imu_norm], dim=-1)
        
        batch_size, seq_length, _ = combined.shape
        
        # Apply input projection and positional encoding
        features = self.fc1(combined)
        pos_embedding = self.positional_embedding(seq_length, combined.device)
        features = features + pos_embedding * 0.1  # Scale down positional encoding
        
        # Generate and apply causal mask
        mask = self.generate_square_subsequent_mask(seq_length, combined.device)
        # Pass through transformer with causal mask
        output = self.transformer_encoder(features, mask=mask)
        
        # Project to 7DoF
        output = self.fc2(output)

        # For relative poses between consecutive frames
        # Take only the last prediction (current -> next frame) to match paper
        predictions = output[:, -1:, :]  # shape [B, 1, 7]
        
        # Split and normalize quaternions
        translation = predictions[:, :, :3]
        quaternion = predictions[:, :, 3:]
        
        # Ensure quaternion normalization doesn't produce NaN
        quat_norm = torch.norm(quaternion, p=2, dim=-1, keepdim=True)
        quaternion = quaternion / (quat_norm + 1e-8)
        
        return {
            'poses': torch.cat([translation, quaternion], dim=-1),
            'translation': translation,
            'rotation': quaternion
        }


def robust_geodesic_loss(pred_quat, gt_quat):
    """Plain geodesic rotation loss for proper gradients"""
    # Reshape to [B*N, 4]
    pred_quat = pred_quat.reshape(-1, 4)
    gt_quat = gt_quat.reshape(-1, 4)
    
    # Normalize with epsilon for stability
    pred_quat = F.normalize(pred_quat, p=2, dim=-1, eps=1e-8)
    gt_quat = F.normalize(gt_quat, p=2, dim=-1, eps=1e-8)
    
    # Compute dot product
    dot = (pred_quat * gt_quat).sum(dim=-1)
    
    # Handle double cover with absolute value
    dot = torch.abs(dot)
    
    # Clamp conservatively to avoid numerical issues
    dot = torch.clamp(dot, min=-0.9999, max=0.9999)
    
    # Plain geodesic distance - no approximation
    angle_error = 2.0 * torch.acos(dot)
    
    return angle_error.mean()


def quat_inv(q):
    """Quaternion inverse (conjugate for unit quaternions)"""
    # q = [x, y, z, w] -> q_inv = [-x, -y, -z, w]
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)

def quat_mul(q1, q2):
    """Quaternion multiplication: q1 * q2"""
    # q1 = [x1, y1, z1, w1], q2 = [x2, y2, z2, w2]
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([x, y, z, w], dim=-1)

def quat_apply(q, v):
    """Apply quaternion rotation to vector v"""
    # Convert vector to quaternion form [vx, vy, vz, 0]
    v_quat = torch.cat([v, torch.zeros_like(v[..., :1])], dim=-1)
    
    # Rotate: q * v * q^{-1}
    q_conj = quat_inv(q)
    v_rot = quat_mul(quat_mul(q, v_quat), q_conj)
    
    # Extract vector part
    return v_rot[..., :3]

def compute_stable_loss(predictions, batch, trans_weight=100.0, rot_weight=1.0):
    """Stable loss computation with gradient-friendly formulation"""
    pred_poses = predictions['poses']  # [B, 1, 7] - only last prediction
    
    # Compute relative ground truth from last two frames
    prev_q = batch['poses'][:, -2:-1, 3:]    # [B, 1, 4] q_{t-1}
    prev_t = batch['poses'][:, -2:-1, :3]   # [B, 1, 3] p_{t-1}
    last_q = batch['poses'][:, -1:, 3:]     # [B, 1, 4] q_t
    last_t = batch['poses'][:, -1:, :3]     # [B, 1, 3] p_t
    
    # Relative rotation: q_rel = q_{t-1}^{-1} * q_t
    q_rel = quat_mul(quat_inv(prev_q), last_q)
    
    # Relative translation expressed in the PREVIOUS frame
    world_delta = last_t - prev_t  # Translation in world frame
    rel_trans = quat_apply(quat_inv(prev_q), world_delta)  # Transform to frame t-1
    
    # Combine into relative pose
    gt_poses = torch.cat([rel_trans, q_rel], dim=-1)  # [B, 1, 7]
    
    # Split predictions and ground truth
    pred_trans = pred_poses[:, :, :3]
    pred_rot = pred_poses[:, :, 3:]
    gt_trans = gt_poses[:, :, :3]
    gt_rot = gt_poses[:, :, 3:]
    
    # Use Huber loss for translation (more robust to outliers)
    trans_loss = F.smooth_l1_loss(pred_trans, gt_trans)
    
    # Rotation loss
    rot_loss = robust_geodesic_loss(pred_rot, gt_rot)
    
    # Check for NaN
    if torch.isnan(trans_loss) or torch.isnan(rot_loss) or torch.isinf(trans_loss) or torch.isinf(rot_loss):
        print(f"Warning: Invalid loss - trans: {trans_loss.item() if not torch.isnan(trans_loss) else 'nan'}, "
              f"rot: {rot_loss.item() if not torch.isnan(rot_loss) else 'nan'}")
        return None
    
    # Combined loss with proper SI unit scaling (100x for meters to match radians)
    total_loss = trans_weight * trans_loss + rot_weight * rot_loss
    
    return {
        'total_loss': total_loss,
        'trans_loss': trans_loss,
        'rot_loss': rot_loss
    }


def train_epoch(model, dataloader, optimizer, device, epoch, grad_clip=10.0):
    """Train with gradient monitoring"""
    model.train()
    total_loss = 0
    num_batches = 0
    nan_count = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        batch_gpu = {
            'visual_features': batch['visual_features'].to(device),
            'imu_features': batch['imu_features'].to(device),
            'poses': batch['poses'].to(device)
        }
        
        # Keep raw features and centimeter-scale translations
        # NO per-batch normalization - the dataset already handles normalization globally
        # Debug first batch to verify data scales
        if batch_idx == 0:
            vf = batch_gpu['visual_features']
            imu = batch_gpu['imu_features']
            poses = batch_gpu['poses']
            
            print(f"Visual features - mean: {vf.mean().item():.6f}, std: {vf.std().item():.6f}")
            print(f"IMU features - mean: {imu.mean().item():.6f}, std: {imu.std().item():.6f}")
            print(f"GT translations (m) - mean: {poses[:,:,:3].mean().item():.6f}, std: {poses[:,:,:3].std().item():.6f}")
            print(f"GT quaternions - mean: {poses[:,:,3:].mean().item():.6f}, std: {poses[:,:,3:].std().item():.6f}")
        
        # Forward pass
        predictions = model(batch_gpu)
        # Print predictions for first batch
        if batch_idx == 0:
            pred_window = predictions['poses'][0].cpu().detach().numpy()  # [seq_len-1, 7]
            print("Predicted pose window shape:", pred_window.shape)
            print(pred_window)
        loss_dict = compute_stable_loss(predictions, batch_gpu)
        
        if loss_dict is None:
            nan_count += 1
            continue
        
        loss = loss_dict['total_loss']
        
        # Don't skip high-loss batches - they contain important learning signals
        # Just log them for monitoring
        if loss.item() > 100.0:
            print(f"High loss batch {batch_idx}: {loss.item():.2f} (training continues)")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradient norms before clipping
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # More aggressive gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        # Always apply optimizer step after clipping
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'trans': f"{loss_dict['trans_loss'].item():.4f}",
            'rot': f"{loss_dict['rot_loss'].item():.4f}",
            'grad': f"{total_norm:.2f}",
            'nan': nan_count
        })
        
        # Detailed logging
        if batch_idx % 50 == 0 and batch_idx > 0:
            with torch.no_grad():
                pred_poses = predictions['poses'][0, :5].cpu().numpy()
                gt_poses = batch_gpu['poses'][0, 1:6].cpu().numpy()
                
                print(f"\nBatch {batch_idx} sample relative poses:")
                print("  Predicted relative pose:")
                print(pred_poses)
                print("  Ground truth relative pose:")
                # Compute relative GT for display
                last = batch_gpu['poses'][0, -1, :].cpu().numpy()
                prev = batch_gpu['poses'][0, -2, :].cpu().numpy()
                gt_rel = last.copy()
                gt_rel[:3] = last[:3] - prev[:3]  # relative translation
                print(gt_rel.reshape(1, -1))
    
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
            loss_dict = compute_stable_loss(predictions, batch_gpu)
            
            # Compute translation error and align GT dynamically
            pred_trans = predictions['translation']  # [B, num_preds, 3]
            num_preds = pred_trans.shape[1]
            gt_trans = batch_gpu['poses'][:, :num_preds, :3]  # [B, num_preds, 3]
            trans_error = torch.mean(torch.norm(pred_trans - gt_trans, dim=-1))
            
            # Convert rotation error to degrees
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
    parser = argparse.ArgumentParser(description='Train VIFT with Stable Training')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)  # Smaller batch size
    parser.add_argument('--lr', type=float, default=1e-4)  # Match paper's learning rate
    parser.add_argument('--data-dir', type=str, default='/home/external/VIFT_AEA/aria_latent_full_frames')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_vift_stable')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(args.checkpoint_dir) / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("VIFT Training - Stable Version")
    print(f"{'='*60}")
    print("Key features:")
    print("- Input normalization for visual and IMU features")
    print("- Robust loss functions (Huber + smooth geodesic)")
    print("- Gradient clipping at 10.0 (matching paper)")
    print("- Pre-norm transformer for stability")
    print("- Learning rate 1e-4 (matching paper)")
    print("- Causal masking with single-step prediction")
    print("- No output BatchNorm (fixed architectural issue)")
    print("- Translation loss weight: 100.0 (meters to match radians)")
    print("- RELATIVE pose training (matching evaluation)")
    print(f"{'='*60}\n")
    
    # Load datasets
    train_dataset = AriaLatentDataset(os.path.join(args.data_dir, 'train'))
    val_dataset = AriaLatentDataset(os.path.join(args.data_dir, 'val'))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders with fewer workers to reduce memory pressure
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=2, pin_memory=False)
    
    # Initialize model
    model = VIFTStable().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Optimizer with small learning rate and weight decay
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
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.8f}")
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
        print(f"  Val Trans Error: {val_metrics['trans_error']:.4f} m")
        print(f"  Val Rot Error: {val_metrics['rot_error_deg']:.2f} degrees")
        
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
                    'input_dim': 768,
                    'embedding_dim': 256,
                    'num_layers': 4,
                    'nhead': 8,
                    'dim_feedforward': 1024,
                    'architecture': 'original',
                    'features': 'matching_paper_architecture,single_step_prediction,proper_init'
                }
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            print(f"  ✓ New best model saved")
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()