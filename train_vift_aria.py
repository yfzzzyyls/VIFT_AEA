#!/usr/bin/env python3
"""
Train VIFT with transition-based architecture for Aria dataset.
Key difference: Model outputs embeddings, transitions are computed as differences.
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
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.data.components.aria_latent_dataset import AriaLatentDataset
from src.models.components.pose_transformer import PoseTransformer


class VIFTTransition(nn.Module):
    """VIFT model with transition-based architecture"""
    
    def __init__(self, input_dim=768, embedding_dim=128, num_layers=2, 
                 nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        # Use the existing PoseTransformer but replace output
        self.pose_transformer = PoseTransformer(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # CRITICAL CHANGE: Output embeddings, not poses
        # Replace the pose output with embedding output
        self.pose_transformer.fc2 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )
        
        # NEW: Transition to pose projection
        # This projects transition embeddings to 7DoF poses
        self.transition_to_pose = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 7)  # 3 translation + 4 quaternion
        )
        
        # Initialize projection with reasonable values
        with torch.no_grad():
            # Initialize last layer with normal distribution
            nn.init.normal_(self.transition_to_pose[-1].weight, mean=0.0, std=0.1)
            # Initialize bias to zero for translations, identity for quaternion
            self.transition_to_pose[-1].bias.data[:3] = torch.tensor([0.0, 0.0, 0.0])  # Zero translation bias
            self.transition_to_pose[-1].bias.data[3:] = torch.tensor([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
    
    def forward(self, batch):
        # Combine visual and IMU features
        visual_features = batch['visual_features']
        imu_features = batch['imu_features']
        combined = torch.cat([visual_features, imu_features], dim=-1)
        
        batch_size, seq_len, _ = combined.shape
        
        # Apply positional encoding and projection
        pos_embedding = self.pose_transformer.positional_embedding(seq_len).to(combined.device)
        combined = self.pose_transformer.fc1(combined)
        combined += pos_embedding
        
        # Use causal mask for proper temporal modeling
        mask = self.pose_transformer.generate_square_subsequent_mask(seq_len, combined.device)
        
        # Pass through transformer with causal mask
        transformer_output = self.pose_transformer.transformer_encoder(combined, mask=mask, is_causal=True)
        
        # Get embeddings (not poses!)
        embeddings = self.pose_transformer.fc2(transformer_output)  # [B, seq_len, embedding_dim]
        
        # Add small noise during training to prevent constant embeddings
        if self.training:
            noise_scale = 0.01  # Very small noise (1%) for stability
            noise = torch.randn_like(embeddings) * noise_scale
            embeddings = embeddings + noise
        
        # CRITICAL: Compute transitions as differences between consecutive embeddings
        # This is the KEY difference from our previous implementation
        transitions = embeddings[:, 1:] - embeddings[:, :-1]  # [B, seq_len-1, embedding_dim]
        
        # Remove layer norm - it destroys motion magnitude information!
        # transitions = nn.functional.layer_norm(transitions, transitions.shape[-1:])
        
        # Project transitions to pose space
        pose_predictions = self.transition_to_pose(transitions)  # [B, seq_len-1, 7]
        
        # Scale up predictions to reasonable motion range
        # This helps prevent collapse to near-zero predictions
        translation_scale = 5.0  # Scale factor for translations
        pose_predictions[:, :, :3] = pose_predictions[:, :, :3] * translation_scale
        
        # Split and normalize quaternions
        translation = pose_predictions[:, :, :3]
        quaternion = pose_predictions[:, :, 3:]
        
        # Normalize quaternions
        quaternion = nn.functional.normalize(quaternion, p=2, dim=-1)
        
        return {
            'embeddings': embeddings,
            'transitions': transitions,
            'translation': translation,
            'rotation': quaternion,
            'poses': torch.cat([translation, quaternion], dim=-1)
        }


def compute_loss(predictions, batch, trans_weight=1.0, rot_weight=1.0, 
                smooth_weight=0.1, diversity_weight=0.1, embed_reg_weight=0.001,
                temporal_weight=0.5, transition_weight=0.5):
    """
    Enhanced loss computation for transition-based predictions.
    
    Key difference: We compare predicted transitions with ground truth transitions,
    not absolute poses.
    """
    pred_poses = predictions['poses']  # [B, seq_len-1, 7]
    embeddings = predictions['embeddings']  # [B, seq_len, embed_dim]
    gt_poses = batch['poses']  # [B, seq_len, 7] - already relative poses!
    
    # Since we predict seq_len-1 relative motions, align with GT
    if gt_poses.shape[1] == pred_poses.shape[1] + 1:
        # Skip the first GT pose
        gt_poses = gt_poses[:, 1:, :]
    elif gt_poses.shape[1] != pred_poses.shape[1]:
        raise ValueError(f"Shape mismatch: pred {pred_poses.shape} vs gt {gt_poses.shape}")
    
    # Split translation and rotation
    pred_trans = pred_poses[:, :, :3]
    pred_rot = pred_poses[:, :, 3:]
    gt_trans = gt_poses[:, :, :3]
    gt_rot = gt_poses[:, :, 3:]
    
    # Normalize GT quaternions
    gt_rot = nn.functional.normalize(gt_rot, p=2, dim=-1)
    
    # 1. Adaptive translation loss based on motion magnitude
    trans_magnitude = gt_trans.norm(dim=-1, keepdim=True)
    small_motion_mask = (trans_magnitude < 0.5).float()  # < 0.5cm
    motion_weight = 1.0 + 2.0 * small_motion_mask  # Weight small motions 3x more
    
    trans_diff = (pred_trans - gt_trans).abs()
    trans_loss = (trans_diff * motion_weight).mean()
    
    # 2. Quaternion loss
    dot = (pred_rot * gt_rot).sum(dim=-1, keepdim=True)
    gt_rot_aligned = torch.where(dot < 0, -gt_rot, gt_rot)
    rot_loss = nn.functional.l1_loss(pred_rot, gt_rot_aligned)
    
    # 3. Temporal smoothness loss on predicted transitions
    smooth_loss = torch.tensor(0.0, device=pred_trans.device)
    if pred_trans.shape[1] > 1:
        # Penalize sudden changes in predicted transitions
        transition_diff = pred_poses[:, 1:] - pred_poses[:, :-1]
        smooth_loss = transition_diff.abs().mean()
    
    # 4. Diversity loss - prevent constant predictions (increased weight!)
    pred_variance = pred_trans.var(dim=1).mean()
    gt_variance = gt_trans.var(dim=1).mean()
    diversity_loss = (gt_variance - pred_variance).abs()
    
    # 5. NEW: Embedding regularization - prevent embeddings from growing too large
    embed_reg_loss = embeddings.pow(2).mean()
    
    # 6. NEW: Temporal variation loss - encourage embeddings to change over time
    # Compute temporal differences to ensure embeddings actually change
    embed_diffs = embeddings[:, 1:] - embeddings[:, :-1]
    embed_diff_norms = embed_diffs.norm(dim=-1).mean()
    # We want embedding differences to be meaningful (not too small)
    temporal_loss = torch.relu(1.0 - embed_diff_norms) * 5.0  # Penalize if diff < 1.0
    
    # 7. NEW: Transition magnitude loss - ensure transitions have reasonable magnitude
    transitions = predictions['transitions']
    transition_norms = transitions.norm(dim=-1)
    # Encourage reasonable transition magnitudes (0.5 to 5.0 cm range)
    transition_loss = (torch.relu(0.5 - transition_norms.mean()) + 
                      torch.relu(transition_norms.mean() - 5.0)) * 2.0
    
    # Combined loss
    total_loss = (trans_weight * trans_loss + 
                  rot_weight * rot_loss + 
                  smooth_weight * smooth_loss +
                  diversity_weight * diversity_loss +
                  embed_reg_weight * embed_reg_loss +
                  temporal_weight * temporal_loss +
                  transition_weight * transition_loss)
    
    return {
        'total_loss': total_loss,
        'trans_loss': trans_loss,
        'rot_loss': rot_loss,
        'smooth_loss': smooth_loss,
        'diversity_loss': diversity_loss,
        'embed_reg_loss': embed_reg_loss,
        'temporal_loss': temporal_loss,
        'transition_loss': transition_loss
    }


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, print_freq=20):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        batch_gpu = {
            'visual_features': batch['visual_features'].to(device),
            'imu_features': batch['imu_features'].to(device),
            'poses': batch['poses'].to(device)
        }
        
        # Forward pass
        predictions = model(batch_gpu)
        loss_dict = compute_loss(predictions, batch_gpu)
        loss = loss_dict['total_loss']
        
        # Check for NaN or very large loss
        if torch.isnan(loss) or loss.item() > 100:
            print(f"Invalid loss at batch {batch_idx} (loss={loss.item():.2f}), skipping...")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Step the scheduler after each batch (for OneCycleLR)
        scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'trans': f"{loss_dict['trans_loss'].item():.4f}",
            'temporal': f"{loss_dict['temporal_loss'].item():.4f}",
            'transition': f"{loss_dict['transition_loss'].item():.4f}"
        })
        
        # Detailed logging
        if batch_idx % print_freq == 0 and batch_idx > 0:
            print(f"\nBatch {batch_idx}:")
            print(f"  Total loss: {loss.item():.6f}")
            print(f"  Trans loss: {loss_dict['trans_loss'].item():.6f}")
            print(f"  Rot loss: {loss_dict['rot_loss'].item():.6f}")
            print(f"  Smooth loss: {loss_dict['smooth_loss'].item():.6f}")
            print(f"  Diversity loss: {loss_dict['diversity_loss'].item():.6f}")
            print(f"  Embed reg loss: {loss_dict['embed_reg_loss'].item():.6f}")
            print(f"  Temporal loss: {loss_dict['temporal_loss'].item():.6f}")
            print(f"  Transition loss: {loss_dict['transition_loss'].item():.6f}")
            
            # Sample predictions
            with torch.no_grad():
                pred_poses = predictions['poses'][0, :5].cpu().numpy()
                gt_poses = batch_gpu['poses'][0, 1:6].cpu().numpy()  # Skip first GT, align with predictions
                
                # Check variance to detect constant predictions
                pred_var = predictions['poses'][0].cpu().numpy()[:, :3].var(axis=0)
                gt_var = batch_gpu['poses'][0, 1:].cpu().numpy()[:, :3].var(axis=0)
                print(f"  Prediction variance: {pred_var}")
                print(f"  GT variance: {gt_var}")
                
                # Check embedding statistics
                embeddings = predictions['embeddings'][0].detach().cpu().numpy()  # [seq_len, embed_dim]
                embed_mean = embeddings.mean()
                embed_std = embeddings.std()
                
                # Check temporal variation in embeddings
                temporal_std = embeddings.std(axis=0).mean()  # Std across time for each dimension
                spatial_std = embeddings.std(axis=1).mean()   # Std across dimensions for each timestep
                
                # Check if embeddings are changing over time
                embedding_diffs = np.diff(embeddings, axis=0)  # Differences between consecutive embeddings
                diff_magnitudes = np.linalg.norm(embedding_diffs, axis=1)
                
                print(f"  Embedding stats:")
                print(f"    Overall mean: {embed_mean:.4f}, std: {embed_std:.4f}")
                print(f"    Temporal std (avg across dims): {temporal_std:.4f}")
                print(f"    Spatial std (avg across time): {spatial_std:.4f}")
                print(f"    Embedding diff magnitudes: min={diff_magnitudes.min():.4f}, max={diff_magnitudes.max():.4f}, mean={diff_magnitudes.mean():.4f}")
                
                # Check first and last embeddings to see if they differ
                first_embed = embeddings[0]
                last_embed = embeddings[-1]
                embed_distance = np.linalg.norm(last_embed - first_embed)
                print(f"    Distance between first and last embedding: {embed_distance:.4f}")
                
                # Check transition magnitudes
                trans_mag = predictions['transitions'][0].norm(dim=-1).cpu().numpy()[:5]
                print(f"  Transition magnitudes: {trans_mag}")
                
                print("  Sample predictions (first 5 frames):")
                for i in range(5):
                    print(f"    Frame {i}:")
                    print(f"      GT:   t=[{gt_poses[i,0]:.3f}, {gt_poses[i,1]:.3f}, {gt_poses[i,2]:.3f}], "
                          f"q=[{gt_poses[i,3]:.3f}, {gt_poses[i,4]:.3f}, {gt_poses[i,5]:.3f}, {gt_poses[i,6]:.3f}]")
                    print(f"      Pred: t=[{pred_poses[i,0]:.3f}, {pred_poses[i,1]:.3f}, {pred_poses[i,2]:.3f}], "
                          f"q=[{pred_poses[i,3]:.3f}, {pred_poses[i,4]:.3f}, {pred_poses[i,5]:.3f}, {pred_poses[i,6]:.3f}]")
    
    return total_loss / num_batches if num_batches > 0 else float('inf')


def validate(model, dataloader, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_trans_error = 0
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
            
            if torch.isnan(loss_dict['total_loss']):
                continue
            
            # Compute translation error
            pred_trans = predictions['translation']
            gt_trans = batch_gpu['poses'][:, :pred_trans.shape[1], :3]
            trans_error = torch.mean(torch.norm(pred_trans - gt_trans, dim=-1))
            
            total_loss += loss_dict['total_loss'].item()
            total_trans_error += trans_error.item()
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches if num_batches > 0 else float('inf'),
        'trans_error': total_trans_error / num_batches if num_batches > 0 else float('inf')
    }


def main():
    parser = argparse.ArgumentParser(description='Train VIFT with transition-based architecture')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--data-dir', type=str, default='/home/external/VIFT_AEA/aria_latent_full_frames')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_vift_aria')
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
    print("VIFT with Transition-based Architecture")
    print(f"{'='*60}")
    print("Key architectural changes:")
    print("1. Model outputs embeddings, not poses directly")
    print("2. Transitions computed as embedding differences")
    print("3. Transitions projected to 7DoF pose space")
    print("4. Enhanced diversity loss weight (0.2)")
    print("5. Embedding regularization added")
    print(f"{'='*60}\n")
    
    # Load datasets
    train_dataset = AriaLatentDataset(os.path.join(args.data_dir, 'train'))
    val_dataset = AriaLatentDataset(os.path.join(args.data_dir, 'val'))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    model = VIFTTransition().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Optimizer with normal learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler - OneCycleLR for better training dynamics
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr * 3,  # Peak at 3x base lr (more stable)
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,  # 20% warmup for stability
        anneal_strategy='cos'
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, 
                               epoch + 1, args.print_freq)
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_metrics['loss']:.6f}")
        print(f"  Val Trans Error: {val_metrics['trans_error']:.4f} cm")
        
        # Save checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            print(f"  ✓ New best model saved")
        
        # Scheduler is stepped in train_epoch for OneCycleLR
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()