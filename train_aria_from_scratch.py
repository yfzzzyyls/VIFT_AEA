#!/usr/bin/env python3
"""
Train VIFT from scratch on Aria Everyday Activities dataset.
This trains all components: Image Encoder + IMU Encoder + Pose Transformer.
No pre-trained weights are used.

Usage:
    # Single GPU training (original mode)
    python train_aria_from_scratch.py --batch-size 4 --epochs 30

    # Multi-GPU training with DistributedDataParallel (recommended for 2+ GPUs)
    # This provides ~2x speedup compared to DataParallel
    torchrun --nproc_per_node=4 train_aria_from_scratch.py --distributed --batch-size 4

    # Alternative launcher (deprecated but works)
    python -m torch.distributed.launch --nproc_per_node=4 train_aria_from_scratch.py --distributed --batch-size 4

    # Custom number of GPUs
    torchrun --nproc_per_node=8 train_aria_from_scratch.py --distributed --batch-size 2

Note: When using distributed training, the effective batch size is batch_size * num_gpus
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.models.components.vsvio import TransformerVIO
from src.data.components.aria_raw_dataset import AriaRawDataModule, collate_variable_length


def robust_geodesic_loss_stable(pred_quat, gt_quat):
    """
    Numerically stable quaternion geodesic loss using atan2.
    This avoids the derivative blow-up of acos near |dot|=1.
    """
    # Reshape to [B*N, 4] for batch processing
    original_shape = pred_quat.shape
    pred_quat = pred_quat.reshape(-1, 4)
    gt_quat = gt_quat.reshape(-1, 4)
    
    # Normalize quaternions with epsilon for stability
    pred_quat = F.normalize(pred_quat, p=2, dim=-1, eps=1e-8)
    gt_quat = F.normalize(gt_quat, p=2, dim=-1, eps=1e-8)
    
    # Compute dot product (handle double cover with abs)
    dot_product = torch.sum(pred_quat * gt_quat, dim=-1)
    abs_dot = dot_product.abs()
    
    # Use atan2(sqrt(1-x²), x) which has finite slope at |x|→1
    # Clamp to avoid numerical issues in sqrt
    abs_dot_clamped = abs_dot.clamp(max=1.0 - 1e-7)
    angle = 2.0 * torch.atan2(
        torch.sqrt(1.0 - abs_dot_clamped**2), 
        abs_dot_clamped
    )
    
    return angle.mean()


class VIFTFromScratch(nn.Module):
    """VIFT model for training from scratch with quaternion output."""
    
    def __init__(self, use_searaft=False):
        super().__init__()
        
        # Model configuration
        class Config:
            # Sequence parameters
            seq_len = 11
            
            # Image parameters
            img_w = 704
            img_h = 704
            
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
        # Add use_searaft after creating the config
        self.config.use_searaft = use_searaft
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
        images = batch['images']  # [B, 11, 3, 704, 704]
        imu = batch['imu']        # [B, 110, 6]
        
        # Get frame IDs if available for multi-frame correlation
        frame_ids = batch.get('frame_ids', None)  # Optional: [B, 11] or None
        
        # Get features from backbone (which creates RGB-RGB pairs internally)
        fv, fi = self.backbone.Feature_net(images, imu, frame_ids)  # fv: [B, 10, 512], fi: [B, 10, 256]
        
        # Concatenate visual and inertial features
        combined_features = torch.cat([fv, fi], dim=-1)  # [B, 10, 768]
        
        # Predict poses for all transitions
        # Each timestep already represents a transition (src->tgt pair)
        transition_features = combined_features  # [B, 10, 768]
        
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


def compute_loss(predictions, batch, alpha=10.0, beta=5.0):
    """Compute loss with quaternion representation for multi-step prediction
    
    Args:
        predictions: Model predictions with 'poses' key
        batch: Input batch with 'gt_poses' key
        alpha: Scale factor for translation loss (default 10.0)
        beta: Scale factor for scale consistency loss (default 5.0)
    """
    pred_poses = predictions['poses']  # [B, 10, 7]
    gt_poses = batch['gt_poses']  # [B, 10, 7]
    
    # Split predictions and ground truth
    pred_trans = pred_poses[:, :, :3]  # [B, 10, 3]
    pred_rot = pred_poses[:, :, 3:]    # [B, 10, 4]
    gt_trans = gt_poses[:, :, :3]      # [B, 10, 3]
    gt_rot = gt_poses[:, :, 3:]        # [B, 10, 4]
    
    # Translation loss
    trans_loss = F.smooth_l1_loss(pred_trans, gt_trans)
    
    # Scale consistency loss - helps prevent scale drift
    pred_scale = pred_trans.norm(dim=-1)  # [B, 10]
    gt_scale = gt_trans.norm(dim=-1)      # [B, 10]
    scale_loss = F.smooth_l1_loss(pred_scale, gt_scale)
    
    # Rotation loss using numerically stable geodesic distance
    rot_loss = robust_geodesic_loss_stable(pred_rot, gt_rot)
    
    if torch.isnan(trans_loss) or torch.isnan(rot_loss) or torch.isnan(scale_loss):
        return None
    
    # Combined loss with proper weighting
    # Translation needs higher weight as it's in meters
    # total_loss = alpha * trans_loss + beta * scale_loss + rot_loss
    total_loss = 10 * trans_loss + beta * scale_loss + 2000 * rot_loss
    
    return {
        'total_loss': total_loss,
        'trans_loss': trans_loss,
        'rot_loss': rot_loss,
        'scale_loss': scale_loss
    }


def train_epoch(model, dataloader, optimizer, device, epoch, warmup_scheduler=None, global_step=0):
    """Train one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    step = global_step  # Use global step counter for gradient clipping
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        batch = {
            'images': batch['images'].to(device),      # [B, 11, 3, H, W]
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
        
        # Gradient clipping schedule
        if step < 300:
            clip_val = 1.0
        elif step < 500:
            clip_val = 2.0
        else:
            clip_val = 3.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
        
        optimizer.step()
        
        # Step warmup scheduler if provided (per batch)
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        
        step += 1  # Increment step counter
        
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
    
    return total_loss / num_batches if num_batches > 0 else float('inf'), step


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
                'images': batch['images'].to(device),      # [B, 11, 3, H, W]
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
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--use-searaft', action='store_true',
                        help='Use SEA-RAFT feature encoder instead of CNN')
    parser.add_argument('--no-multiframe', action='store_true',
                        help='Disable multi-frame correlation in SEA-RAFT encoder')
    
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
    
    # Only print from main process
    if is_main_process:
        if args.distributed:
            print(f"Using Distributed Training with {world_size} GPUs")
        else:
            print(f"Using single GPU training")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if is_main_process:
        print("\n" + "="*60)
        print("Training VIFT from Scratch on Aria Data")
        print("="*60)
        print("Training all components:")
        print(f"- Image Encoder ({'SEA-RAFT Features' if args.use_searaft else '6-layer CNN'})")
        if args.use_searaft and hasattr(args, 'use_multiframe') and args.use_multiframe:
            print("  ✓ Multi-frame correlation enabled")
        print("- IMU Encoder (3-layer 1D CNN)")
        print("- Pose Transformer (4 layers, 8 heads)")
        print("- Quaternion output (3 trans + 4 quat)")
        print("- Multi-step prediction (all 10 transitions)")
        print(f"\nTraining Configuration:")
        print(f"- Distributed: {'Yes (DDP)' if args.distributed else 'No'}")
        print(f"- World size: {world_size}")
        print(f"- Batch size per GPU: {args.batch_size}")
        print(f"- Total batch size: {args.batch_size * world_size}")
        print(f"- Learning rate: {args.lr}")
        print(f"- Window stride: 1 (overlapping windows for more training samples)")
        print("="*60 + "\n")
    
    # Create data module
    # For DDP, each process loads its own subset of data
    data_module = AriaRawDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,  # Per GPU batch size
        num_workers=args.num_workers,
        sequence_length=11,
        stride=1  # Changed from 10 - will create 10x more training samples
    )
    data_module.setup()
    
    # Get datasets
    train_dataset = data_module.train_dataset
    val_dataset = data_module.val_dataset
    
    # Create distributed samplers if using DDP
    if args.distributed:
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
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_variable_length
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_variable_length
    )
    
    if is_main_process:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = VIFTFromScratch(use_searaft=args.use_searaft)
    
    # Configure multi-frame if using SEA-RAFT
    if args.use_searaft and not args.no_multiframe:
        model.config.use_multiframe = True
        if is_main_process:
            print("✓ Multi-frame correlation enabled for SEA-RAFT")
    else:
        model.config.use_multiframe = False
        
    model = model.to(device)
    
    # Wrap with DDP if distributed
    if args.distributed:
        # Set find_unused_parameters=True because the TransformerVIO model
        # has some unused parameters in its architecture
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
    global_step = 0  # Track global step across epochs
    
    for epoch in range(1, args.epochs + 1):
        # Set epoch for distributed sampler
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if is_main_process:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"{'='*60}")
        
        # Determine if we're in warmup phase
        warmup_scheduler_to_use = warmup_scheduler if global_step < warmup_steps else None
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, device, epoch, 
            warmup_scheduler=warmup_scheduler_to_use, global_step=global_step
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Step cosine scheduler after warmup phase (per epoch)
        if global_step >= warmup_steps:
            cosine_scheduler.step()
        
        # Only print and save on main process
        if is_main_process:
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_metrics['loss']:.6f}")
            print(f"  Val Trans Error: {val_metrics['trans_error']*100:.2f} cm")
            print(f"  Val Rot Error: {val_metrics['rot_error']:.2f}°")
        
        # Save checkpoints only on main process
        if is_main_process:
            # Get model state dict (unwrap DDP/DataParallel if needed)
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            base_model = model.module if hasattr(model, 'module') else model
            
            # Save best checkpoint
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': warmup_scheduler.state_dict() if global_step < warmup_steps else cosine_scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_metrics': val_metrics,
                    'config': base_model.config.__dict__,
                    'global_step': global_step
                }
                torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
                print("  ✓ New best model saved")
            
            # Save latest checkpoint (always create fresh dictionary)
            latest_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': warmup_scheduler.state_dict() if global_step < warmup_steps else cosine_scheduler.state_dict(),
                'train_loss': train_loss,
                'val_metrics': val_metrics,
                'config': base_model.config.__dict__,
                'global_step': global_step
            }
            torch.save(latest_checkpoint, checkpoint_dir / 'latest_model.pt')
    
    # Final summary
    if is_main_process:
        print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
        print(f"Checkpoints saved to: {checkpoint_dir}")
    
    # Clean up distributed training
    cleanup_distributed()


if __name__ == "__main__":
    main()