#!/usr/bin/env python3
"""
Train VIFT from scratch on TUM VI dataset.
This is a separate training script specifically for TUM VI.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, ConcatDataset, DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.components.vsvio import TransformerVIO
from src.data.components.tumvi_dataset import TUMVIDataset
from train_aria_from_scratch import VIFTFromScratch, compute_loss, robust_geodesic_loss_stable


def create_tumvi_data_loaders(data_root, batch_size, num_workers=4):
    """Create TUM VI data loaders with proper directory structure."""
    data_root = Path(data_root)
    
    # Define train, validation and test sequences
    # Room sequences are in 512x512, corridor sequences in 1024x1024
    train_sequences = ['room1', 'room2', 'room3', 'room4', 
                      'corridor1', 'corridor2', 'corridor3']
    val_sequences = ['room5', 'corridor4']
    test_sequences = ['room6', 'corridor5']  # Reserved for testing, not used during training
    
    # Create train datasets
    train_datasets = []
    for seq in train_sequences:
        # Check both possible directory structures
        seq_paths = [
            data_root / seq,  # Direct: /path/tumvi/room1/
            data_root / f"dataset-{seq}_512_16"  # Extracted: /path/tumvi/dataset-room1_512_16/
        ]
        
        for seq_path in seq_paths:
            if seq_path.exists() and (seq_path / 'mav0').exists():
                print(f"Found training sequence at: {seq_path}")
                dataset = TUMVIDataset(
                    root_dir=seq_path.parent,
                    sequence=seq_path.name,
                    sequence_length=11,
                    stride=1  # Dense sampling for training
                )
                train_datasets.append(dataset)
                break
        else:
            print(f"Warning: Training sequence {seq} not found")
    
    if not train_datasets:
        raise RuntimeError("No training sequences found!")
    
    train_dataset = ConcatDataset(train_datasets)
    print(f"Total training samples: {len(train_dataset)}")
    
    # Create validation datasets
    val_datasets = []
    for seq in val_sequences:
        seq_paths = [
            data_root / seq,
            data_root / f"dataset-{seq}_512_16"
        ]
        
        for seq_path in seq_paths:
            if seq_path.exists() and (seq_path / 'mav0').exists():
                print(f"Found validation sequence at: {seq_path}")
                dataset = TUMVIDataset(
                    root_dir=seq_path.parent,
                    sequence=seq_path.name,
                    sequence_length=11,
                    stride=10  # Sparse sampling for validation
                )
                val_datasets.append(dataset)
                break
        else:
            print(f"Warning: Validation sequence {seq} not found")
    
    if val_datasets:
        val_dataset = ConcatDataset(val_datasets)
        print(f"Total validation samples: {len(val_dataset)}")
    else:
        print("Warning: No validation sequences found, using subset of training data")
        # Use 10% of training data for validation
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_epoch(model, dataloader, optimizer, device, epoch, scale_loss_weight=20.0):
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
        loss_dict = compute_loss(predictions, batch, beta=scale_loss_weight)
        
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
            'rot': f"{loss_dict['rot_loss'].item():.4f}",
            'scale': f"{loss_dict['scale_loss'].item():.4f}"
        })
    
    return total_loss / num_batches if num_batches > 0 else float('inf')


def validate(model, dataloader, device, scale_loss_weight=20.0):
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
            loss_dict = compute_loss(predictions, batch, beta=scale_loss_weight)
            
            if loss_dict is None:
                continue
            
            # Compute errors
            pred_poses = predictions['poses']  # [B, 10, 7]
            gt_poses = batch['gt_poses']  # [B, 10, 7]
            
            # Translation error
            pred_trans = pred_poses[:, :, :3]
            gt_trans = gt_poses[:, :, :3]
            trans_error = torch.norm(pred_trans - gt_trans, dim=-1).mean()
            
            # Rotation error
            pred_rot = pred_poses[:, :, 3:]
            gt_rot = gt_poses[:, :, 3:]
            pred_rot = F.normalize(pred_rot, p=2, dim=-1)
            gt_rot = F.normalize(gt_rot, p=2, dim=-1)
            
            # Compute angle between quaternions
            dot_product = torch.sum(pred_rot * gt_rot, dim=-1).abs()
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
    parser = argparse.ArgumentParser(description='Train VIFT on TUM VI dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Root directory containing TUM VI sequences')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_tumvi',
                        help='Directory to save checkpoints')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--scale-loss-weight', type=float, default=20.0,
                        help='Weight for scale consistency loss')
    parser.add_argument('--encoder-type', type=str, default='flownet', choices=['cnn', 'flownet'],
                        help='Type of visual encoder to use (default: flownet)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Training VIFT on TUM VI Dataset")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Scale loss weight: {args.scale_loss_weight}")
    print(f"Encoder type: {args.encoder_type}")
    print("="*60 + "\n")
    
    # Create data loaders
    train_loader, val_loader = create_tumvi_data_loaders(
        args.data_dir,
        args.batch_size,
        args.num_workers
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VIFTFromScratch(encoder_type=args.encoder_type).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {checkpoint['epoch']}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch,
            scale_loss_weight=args.scale_loss_weight
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device, 
                             scale_loss_weight=args.scale_loss_weight)
        
        # Step scheduler
        scheduler.step()
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_metrics['loss']:.6f}")
        print(f"  Val Trans Error: {val_metrics['trans_error']*100:.2f} cm")
        print(f"  Val Rot Error: {val_metrics['rot_error']:.2f}°")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_metrics': val_metrics,
            'config': model.config.__dict__,
            'best_val_loss': best_val_loss
        }
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            print("  ✓ New best model saved")
        
        # Save latest model
        torch.save(checkpoint, checkpoint_dir / 'latest_model.pt')
        
        # Save periodic checkpoints
        if epoch % 5 == 0:
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()