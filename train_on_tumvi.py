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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.components.vsvio import TransformerVIO
from src.data.components.tumvi_dataset import TUMVIDataset
try:
    from src.data.components.tumvi_fast_dataset import TUMVIFastDataset
except ImportError:
    TUMVIFastDataset = None
from train_aria_from_scratch import VIFTFromScratch, compute_loss, robust_geodesic_loss_stable


def setup_distributed(args):
    """Initialize distributed training if requested."""
    if args.distributed:
        # Get local rank from environment if not set
        if args.local_rank == -1:
            args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Set CUDA device before initializing process group
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        
        # Initialize the process group
        dist.init_process_group(
            backend=args.backend,
            init_method=args.init_method,
            world_size=int(os.environ.get('WORLD_SIZE', 1)),
            rank=int(os.environ.get('RANK', 0))
        )
        
        # Get rank and world size
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        print(f"[Rank {rank}/{world_size}] Distributed training initialized on GPU {args.local_rank}")
        return device, rank, world_size
    else:
        # Non-distributed setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, 0, 1


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_tumvi_data_loaders(data_root, batch_size, num_workers=4, stride=10, distributed=False, world_size=1, rank=0):
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
                
                # Check if preprocessed data exists
                use_fast = False
                if TUMVIFastDataset and (seq_path / 'mav0' / 'cam0' / 'data').exists():
                    # Check for .npy files
                    npy_files = list((seq_path / 'mav0' / 'cam0' / 'data').glob('*.npy'))
                    if len(npy_files) > 0:
                        use_fast = True
                        print(f"  Using preprocessed data (found {len(npy_files)} .npy files)")
                
                if use_fast:
                    dataset = TUMVIFastDataset(
                        root_dir=seq_path.parent,
                        sequence=seq_path.name,
                        sequence_length=11,
                        stride=stride,
                        use_preprocessed=True
                    )
                else:
                    dataset = TUMVIDataset(
                        root_dir=seq_path.parent,
                        sequence=seq_path.name,
                        sequence_length=11,
                        stride=stride
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
                
                # Check if preprocessed data exists
                use_fast = False
                if TUMVIFastDataset and (seq_path / 'mav0' / 'cam0' / 'data').exists():
                    # Check for .npy files
                    npy_files = list((seq_path / 'mav0' / 'cam0' / 'data').glob('*.npy'))
                    if len(npy_files) > 0:
                        use_fast = True
                        print(f"  Using preprocessed data (found {len(npy_files)} .npy files)")
                
                if use_fast:
                    dataset = TUMVIFastDataset(
                        root_dir=seq_path.parent,
                        sequence=seq_path.name,
                        sequence_length=11,
                        stride=max(stride, 10),
                        use_preprocessed=True
                    )
                else:
                    dataset = TUMVIDataset(
                        root_dir=seq_path.parent,
                        sequence=seq_path.name,
                        sequence_length=11,
                        stride=max(stride, 10)  # At least stride 10 for validation
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
    
    # Create samplers for distributed training
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler


def train_epoch(model, dataloader, optimizer, device, epoch, scale_loss_weight=20.0, gradient_accumulation_steps=1):
    """Train one epoch with gradient accumulation support."""
    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()
    
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
        
        # Scale loss by gradient accumulation steps
        loss = loss / gradient_accumulation_steps
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"NaN loss detected at batch {batch_idx}, skipping...")
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping and optimizer step only after accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
        
        # Update metrics (unscale loss for accurate reporting)
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
            'trans': f"{loss_dict['trans_loss'].item():.4f}",
            'rot': f"{loss_dict['rot_loss'].item():.4f}",
            'scale': f"{loss_dict['scale_loss'].item():.4f}",
            'acc_step': f"{(batch_idx % gradient_accumulation_steps) + 1}/{gradient_accumulation_steps}"
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
    
    # Clear GPU cache to free memory
    torch.cuda.empty_cache()
    
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
    parser.add_argument('--stride', type=int, default=10,
                        help='Stride for creating training sequences (default: 10, use 1 for dense sampling)')
    parser.add_argument('--encoder-type', type=str, default='flownet', choices=['cnn', 'flownet'],
                        help='Type of visual encoder to use (default: flownet)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Number of gradient accumulation steps (default: 1)')
    
    # Transformer architecture arguments
    parser.add_argument('--transformer-layers', type=int, default=8,
                        help='Number of transformer encoder layers (default: 8)')
    parser.add_argument('--transformer-heads', type=int, default=16,
                        help='Number of attention heads (default: 16)')
    parser.add_argument('--transformer-dim-feedforward', type=int, default=4096,
                        help='Dimension of feedforward network (default: 4096)')
    parser.add_argument('--transformer-dropout', type=float, default=0.1,
                        help='Dropout rate in transformer (default: 0.1)')
    
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Distributed training arguments
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed training with DDP')
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='Local rank for distributed training (set automatically by torchrun)')
    parser.add_argument('--backend', type=str, default='nccl',
                        help='Distributed backend to use (nccl, gloo)')
    parser.add_argument('--init-method', type=str, default='env://',
                        help='URL to set up distributed training')
    
    args = parser.parse_args()
    
    # Setup distributed training
    device, rank, world_size = setup_distributed(args)
    is_main_process = rank == 0
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    if is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if is_main_process:
        print("\n" + "="*60)
        print("Training VIFT on TUM VI Dataset")
        print("="*60)
        print(f"Data directory: {args.data_dir}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Effective batch size per GPU: {args.batch_size * args.gradient_accumulation_steps}")
        print(f"Total effective batch size: {args.batch_size * args.gradient_accumulation_steps * world_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Scale loss weight: {args.scale_loss_weight}")
        print(f"Encoder type: {args.encoder_type}")
        print(f"Stride: {args.stride}")
        print(f"Distributed: {'Yes' if args.distributed else 'No'}")
        print(f"World size: {world_size}")
        print("="*60 + "\n")
    
    # Create data loaders
    train_loader, val_loader, train_sampler = create_tumvi_data_loaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        stride=args.stride,
        distributed=args.distributed,
        world_size=world_size,
        rank=rank
    )
    
    # Create model
    model = VIFTFromScratch(
        encoder_type=args.encoder_type,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_dim_feedforward=args.transformer_dim_feedforward,
        transformer_dropout=args.transformer_dropout
    ).to(device)
    
    # Wrap with DDP if distributed
    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )
        if is_main_process:
            print(f"Using DistributedDataParallel with {world_size} GPUs")
    
    # Count parameters (only on main process)
    if is_main_process:
        base_model = model.module if hasattr(model, 'module') else model
        total_params = sum(p.numel() for p in base_model.parameters())
        trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
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
        # Set epoch for distributed sampler
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if is_main_process:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch,
            scale_loss_weight=args.scale_loss_weight,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device, 
                             scale_loss_weight=args.scale_loss_weight)
        
        # Step scheduler
        scheduler.step()
        
        # Only print and save on main process
        if is_main_process:
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_metrics['loss']:.6f}")
            print(f"  Val Trans Error: {val_metrics['trans_error']*100:.2f} cm")
            print(f"  Val Rot Error: {val_metrics['rot_error']:.2f}°")
        
        # Save checkpoints only on main process
        if is_main_process:
            # Get model state dict (unwrap DDP if needed)
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            base_model = model.module if hasattr(model, 'module') else model
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_metrics': val_metrics,
                'config': base_model.config.__dict__,
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
    
    if is_main_process:
        print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
        print(f"Checkpoints saved to: {checkpoint_dir}")
    
    # Clean up distributed training
    cleanup_distributed()


if __name__ == "__main__":
    main()