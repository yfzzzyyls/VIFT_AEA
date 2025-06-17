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
import math
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
from src.data.components.aria_raw_dataset import AriaRawDataModule

# -----------------------------------------------
# Aria hardware specifications
VIDEO_FPS = 20           # Aria RGB stream at 20 FPS
IMU_RATE_HZ = 1000       # Aria high-rate IMU at 1kHz
IMU_PER_FRAME = IMU_RATE_HZ // VIDEO_FPS   # 50 samples per video frame
# -----------------------------------------------


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
    
    def __init__(self):
        super().__init__()
        
        # Model configuration
        class Config:
            # Sequence parameters
            seq_len = 21          # 21 RGB frames → 20 pose transitions
            
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
        # Use the standard TransformerVIO model
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
        
        # Learnable uncertainty weights for automatic loss balancing
        self.log_sigma_t = nn.Parameter(torch.zeros(()))  # Translation uncertainty
        self.log_sigma_r = nn.Parameter(torch.zeros(()))  # Rotation uncertainty
        
        # Running statistics for adaptive loss scaling
        self.register_buffer('trans_loss_avg', torch.ones(()))
        self.register_buffer('rot_loss_avg', torch.ones(()))
        self.register_buffer('loss_count', torch.zeros(()))
    
    def _robust_geodesic_loss_stable(self, pred_quat, gt_quat):
        """Numerically stable quaternion geodesic loss using atan2."""
        # Reshape to [B*N, 4] for batch processing
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
    
    def forward(self, batch):
        """Forward pass."""
        seq_len = self.config.seq_len
        num_transitions = seq_len - 1
        
        images = batch['images']  # [B, seq_len, 3, 256, 512]
        imu = batch['imu']        # [B, num_transitions * samples_per_interval, 6]
        
        # Validate input shapes
        B = images.shape[0]
        assert images.shape[1] == seq_len, f"Expected {seq_len} frames, got {images.shape[1]}"
        assert images.shape[2:] == (3, 256, 512), f"Expected image shape (3, 256, 512), got {images.shape[2:]}"
        
        # Flexible IMU validation - detect actual samples per interval
        total_imu_samples = imu.shape[1]
        if total_imu_samples % num_transitions == 0:
            samples_per_interval = total_imu_samples // num_transitions
            # Common configurations: 10 (100Hz), 11 (110Hz), 50 (1kHz)
            if samples_per_interval not in [10, 11, 50]:
                print(f"WARNING: Unusual IMU sampling rate: {samples_per_interval} samples per interval")
        else:
            raise ValueError(f"IMU samples {total_imu_samples} not divisible by {num_transitions} transitions")
        
        assert imu.shape[2] == 6, f"Expected 6 IMU channels (ax,ay,az,gx,gy,gz), got {imu.shape[2]}"
        
        # Validate IMU data format: (ax, ay, az, gx, gy, gz)
        # Check that gyroscope values are reasonable (typically < 10 rad/s)
        gyro_magnitude = torch.norm(imu[:, :, 3:], dim=-1)
        assert gyro_magnitude.max() < 20.0, f"Gyroscope values too large (max: {gyro_magnitude.max():.2f}), check if IMU format is (ax,ay,az,gx,gy,gz)"
        
        # Get features from backbone's Feature_net (which creates RGB-RGB pairs internally)
        num_transitions = seq_len - 1
        fv, fi = self.backbone.Feature_net(images, imu)  # fv: [B, num_transitions, 512], fi: [B, num_transitions, 256]
        
        # Validate feature shapes
        assert fv.shape == (B, num_transitions, 512), f"Expected visual features shape (B, {num_transitions}, 512), got {fv.shape}"
        assert fi.shape == (B, num_transitions, 256), f"Expected IMU features shape (B, {num_transitions}, 256), got {fi.shape}"
        
        # Check for dimension swaps - features should have reasonable statistics
        fv_mean = fv.mean()
        fv_batch_mean = fv.mean(dim=(1, 2))  # Mean per batch sample
        # Commented out - this assertion may be too strict during early training
        # assert not torch.allclose(fv_mean.expand_as(fv_batch_mean), fv_batch_mean, rtol=0.1), \
        #     "Visual features have suspiciously uniform statistics across batch dimension"
        
        # Concatenate visual and inertial features
        combined_features = torch.cat([fv, fi], dim=-1)  # [B, 20, 768]
        
        # Predict poses for all transitions
        # Each timestep already represents a transition (src->tgt pair)
        transition_features = combined_features  # [B, 20, 768]
        
        # Apply pose predictor to each timestep
        batch_size, seq_len, feat_dim = transition_features.shape
        transition_features_flat = transition_features.reshape(-1, feat_dim)  # [B*20, 768]
        poses_flat = self.pose_predictor(transition_features_flat)  # [B*20, 7]
        poses = poses_flat.reshape(batch_size, seq_len, 7)  # [B, 20, 7]
        
        # Normalize quaternion part for each pose
        trans = poses[:, :, :3]  # [B, 20, 3]
        quat = poses[:, :, 3:]   # [B, 20, 4]
        quat = F.normalize(quat, p=2, dim=-1)
        
        # Combine and return
        normalized_poses = torch.cat([trans, quat], dim=-1)  # [B, 20, 7]
        
        # If ground truth is provided, compute loss components
        if 'gt_poses' in batch:
            gt_poses = batch['gt_poses']  # [B, 20, 7]
            
            # Split predictions and ground truth
            pred_trans = normalized_poses[:, :, :3]  # [B, 20, 3]
            pred_rot = normalized_poses[:, :, 3:]    # [B, 20, 4]
            gt_trans = gt_poses[:, :, :3]      # [B, 20, 3]
            gt_rot = gt_poses[:, :, 3:]        # [B, 20, 4]
            
            # Raw losses
            trans_loss_raw = F.smooth_l1_loss(pred_trans, gt_trans, reduction='mean')
            rot_loss_raw = self._robust_geodesic_loss_stable(pred_rot, gt_rot)
            
            # Adaptive scaling
            trans_scale = (1.0 / (self.trans_loss_avg + 1e-8)).detach()
            rot_scale = (1.0 / (self.rot_loss_avg + 1e-8)).detach()
            
            trans_loss = trans_loss_raw * trans_scale
            rot_loss = rot_loss_raw * rot_scale
            
            # Compute weighted losses with learnable uncertainty
            precision_t = torch.exp(-2 * self.log_sigma_t)
            precision_r = torch.exp(-2 * self.log_sigma_r)
            
            weighted_trans_loss = 0.5 * precision_t * trans_loss + self.log_sigma_t
            weighted_rot_loss = 0.5 * precision_r * rot_loss + self.log_sigma_r
            
            total_loss = weighted_trans_loss + weighted_rot_loss
            
            return {
                'poses': normalized_poses,
                'total_loss': total_loss,
                'trans_loss_raw': trans_loss_raw,
                'rot_loss_raw': rot_loss_raw,
                'trans_scale': trans_scale,
                'rot_scale': rot_scale
            }
        else:
            return {
                'poses': normalized_poses  # [B, 20, 7]
            }


def compute_loss(model, predictions, batch, is_training=True):
    """Process loss computation results and update running statistics
    
    Args:
        model: Model instance (to access running stats)
        predictions: Model predictions with loss components
        batch: Input batch (not used now since loss is computed in forward)
        is_training: Whether we're in training mode (to update running stats)
    """
    # Loss is already computed in the forward pass
    if 'total_loss' not in predictions:
        raise ValueError("Loss not computed in forward pass. Make sure gt_poses is in batch.")
    
    total_loss = predictions['total_loss']
    trans_loss_raw = predictions['trans_loss_raw']
    rot_loss_raw = predictions['rot_loss_raw']
    
    if torch.isnan(trans_loss_raw) or torch.isnan(rot_loss_raw):
        return None
    
    # Get the actual model (unwrap DDP if needed)
    base_model = model.module if hasattr(model, 'module') else model
    
    # Update running statistics (exponential moving average)
    if is_training and base_model.training:
        with torch.no_grad():
            if base_model.loss_count < 100:  # Collect initial statistics
                # Simple average for first 100 batches
                base_model.trans_loss_avg = (base_model.trans_loss_avg * base_model.loss_count + trans_loss_raw) / (base_model.loss_count + 1)
                base_model.rot_loss_avg = (base_model.rot_loss_avg * base_model.loss_count + rot_loss_raw) / (base_model.loss_count + 1)
                base_model.loss_count += 1
            else:
                # Exponential moving average after initial period
                alpha = 0.01
                base_model.trans_loss_avg = (1 - alpha) * base_model.trans_loss_avg + alpha * trans_loss_raw
                base_model.rot_loss_avg = (1 - alpha) * base_model.rot_loss_avg + alpha * rot_loss_raw
    
    return {
        'total_loss': total_loss,
        'trans_loss': trans_loss_raw,  # Report raw loss for monitoring
        'rot_loss': rot_loss_raw,      # Report raw loss for monitoring
        'trans_scale': predictions['trans_scale'].item(),
        'rot_scale': predictions['rot_scale'].item(),
        'sigma_t': torch.exp(base_model.log_sigma_t).detach().item(),
        'sigma_r': torch.exp(base_model.log_sigma_r).detach().item()
    }


def train_epoch(model, dataloader, optimizers, device, epoch, warmup_schedulers=None, global_step=0, use_amp=False):
    """Train one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    step = global_step  # Use global step counter for gradient clipping
    
    # Create gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        batch = {
            'images': batch['images'].to(device),      # [B, 21, 3, H, W]
            'imu': batch['imu'].to(device).float(),
            'gt_poses': batch['gt_poses'].to(device)
        }
        
        # Forward pass with automatic mixed precision
        if use_amp:
            with torch.cuda.amp.autocast():
                predictions = model(batch)
        else:
            predictions = model(batch)
        
        # Compute loss
        loss_dict = compute_loss(model, predictions, batch)
        
        if loss_dict is None:
            print(f"NaN loss detected at batch {batch_idx}, skipping...")
            continue
        
        loss = loss_dict['total_loss']
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"NaN loss detected at batch {batch_idx}, skipping...")
            continue
        
        # Backward pass
        for opt in optimizers:
            opt.zero_grad()
        
        if use_amp:
            # Scale loss and backward pass
            scaler.scale(loss).backward()
            
            # Unscale gradients before clipping
            for opt in optimizers:
                scaler.unscale_(opt)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # Step optimizers and update scaler
            for opt in optimizers:
                scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            
            # Gradient clipping (simplified to single value)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            for opt in optimizers:
                opt.step()
        
        # Step warmup schedulers if provided (per batch)
        if warmup_schedulers is not None:
            for scheduler in warmup_schedulers:
                scheduler.step()
        
        step += 1  # Increment step counter
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'trans': f"{loss_dict['trans_loss'].item():.4f}",
            'rot': f"{loss_dict['rot_loss'].item():.4f}",
            'σt': f"{loss_dict['sigma_t']:.2f}",
            'σr': f"{loss_dict['sigma_r']:.2f}"
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
                'images': batch['images'].to(device),      # [B, 21, 3, H, W]
                'imu': batch['imu'].to(device).float(),
                'gt_poses': batch['gt_poses'].to(device)
            }
            
            # Forward pass
            predictions = model(batch)
            
            # Compute loss
            loss_dict = compute_loss(model, predictions, batch, is_training=False)
            
            if loss_dict is None:
                continue
            
            # Compute errors for multi-step prediction
            pred_poses = predictions['poses']  # [B, 20, 7]
            gt_poses = batch['gt_poses']  # [B, 20, 7]
            
            # Translation error
            pred_trans = pred_poses[:, :, :3]  # [B, 20, 3]
            gt_trans = gt_poses[:, :, :3]      # [B, 20, 3]
            trans_error = torch.norm(pred_trans - gt_trans, dim=-1).mean()
            
            # Rotation error
            pred_rot = pred_poses[:, :, 3:]  # [B, 20, 4]
            gt_rot = gt_poses[:, :, 3:]      # [B, 20, 4]
            pred_rot = F.normalize(pred_rot, p=2, dim=-1)
            gt_rot = F.normalize(gt_rot, p=2, dim=-1)
            
            # Compute angle between quaternions
            dot_product = torch.sum(pred_rot * gt_rot, dim=-1).abs()  # [B, num_transitions]
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
    parser.add_argument('--lr-cnn', type=float, default=None,
                        help='CNN learning rate (defaults to --lr)')
    parser.add_argument('--lr-trf', type=float, default=None,
                        help='Transformer learning rate (defaults to --lr)')
    parser.add_argument('--opt-cnn', type=str, default='adamw', choices=['adamw', 'sgd'],
                        help='Optimizer for CNN')
    parser.add_argument('--opt-trf', type=str, default='adamw', choices=['adamw', 'sgd'],
                        help='Optimizer for Transformer')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_from_scratch',
                        help='Directory to save checkpoints')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision training (saves ~40% memory)')
    
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
        print("- Image Encoder (6-layer CNN)")
        print("- IMU Encoder (3-layer 1D CNN)")
        print("- Pose Transformer (4 layers, 8 heads)")
        print("- Quaternion output (3 trans + 4 quat)")
        print("- Multi-step prediction (all 20 transitions)")
        print(f"\nTraining Configuration:")
        print(f"- Distributed: {'Yes (DDP)' if args.distributed else 'No'}")
        print(f"- World size: {world_size}")
        print(f"- Batch size per GPU: {args.batch_size}")
        print(f"- Total batch size: {args.batch_size * world_size}")
        print(f"- Learning rate: {args.lr}")
        print(f"- Window stride: 2 (overlapping sequences for better temporal consistency)")
        print(f"- Mixed precision (AMP): {'Enabled' if args.amp else 'Disabled'}")
        print("="*60 + "\n")
    
    # Create data module
    # For DDP, each process loads its own subset of data
    data_module = AriaRawDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,  # Per GPU batch size
        num_workers=args.num_workers,
        sequence_length=21,   # keep stride=2
        stride=2  # Better temporal overlap (was 10)
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
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    if is_main_process:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = VIFTFromScratch()
    model = model.to(device)
    
    # Wrap with DDP if distributed
    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True  # Set to True since we're only using Feature_net from TransformerVIO
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
    
    # Split parameters between CNN and Transformer
    base_model = model.module if hasattr(model, 'module') else model
    cnn_params = []
    trf_params = []
    
    # Visual encoder parameters (CNN)
    # Also include IMU encoder in CNN params since it's also convolutional
    for name, param in base_model.named_parameters():
        if 'backbone.visual_encoder' in name or 'backbone.inertial_encoder' in name:
            cnn_params.append(param)
        elif 'backbone' in name or 'pose_predictor' in name or 'log_sigma' in name:
            trf_params.append(param)
        else:
            # Fail fast if we encounter unexpected parameters
            # This guard ensures that any new model components (e.g., new stem blocks, auxiliary heads)
            # are explicitly classified into either CNN or Transformer groups, preventing silent
            # misclassification that could lead to suboptimal learning rates or optimizer settings.
            raise RuntimeError(f"Unclassified parameter: {name}. Please update parameter splitting logic.")
    
    # Count parameters in each group
    cnn_count = sum(p.numel() for p in cnn_params)
    trf_count = sum(p.numel() for p in trf_params)
    
    if is_main_process:
        print(f"\nParameter Split:")
        print(f"  CNN parameters: {cnn_count:,}")
        print(f"  Transformer parameters: {trf_count:,}")
        
        # Verify all parameters are accounted for
        assert cnn_count + trf_count == trainable_params, \
            f"Parameter count mismatch: {cnn_count} + {trf_count} != {trainable_params}"
    
    # Set learning rates
    lr_cnn = args.lr_cnn if args.lr_cnn is not None else args.lr
    lr_trf = args.lr_trf if args.lr_trf is not None else args.lr
    
    # Create optimizers based on command line arguments
    if args.opt_cnn == 'sgd' and args.opt_trf == 'adamw' and len(cnn_params) > 0 and len(trf_params) > 0:
        # Different optimizers for CNN and Transformer
        optimizer_cnn = torch.optim.SGD([
            {'params': cnn_params, 'lr': lr_cnn, 'momentum': 0.9}
        ], lr=lr_cnn)
        
        # Create second optimizer for transformer
        optimizer_trf = torch.optim.AdamW([
            {'params': trf_params, 'lr': lr_trf}
        ], lr=lr_trf, weight_decay=1e-4)
        
        # Use canonical PyTorch pattern: separate optimizers and schedulers
        optimizers = [optimizer_cnn, optimizer_trf]
        use_dual_optimizers = True
        
        if is_main_process:
            print(f"  CNN LR: {lr_cnn:.2e}, Optimizer: sgd")
            print(f"  Transformer LR: {lr_trf:.2e}, Optimizer: adamw")
    else:
        # Single optimizer for all parameters
        if args.opt_cnn == 'sgd':
            optimizer = torch.optim.SGD([
                {'params': cnn_params, 'lr': lr_cnn},
                {'params': trf_params, 'lr': lr_trf}
            ], lr=args.lr, momentum=0.9)
        else:
            optimizer = torch.optim.AdamW([
                {'params': cnn_params, 'lr': lr_cnn},
                {'params': trf_params, 'lr': lr_trf}
            ], lr=args.lr, weight_decay=1e-4)
        
        optimizers = [optimizer]
        use_dual_optimizers = False
        
        if is_main_process:
            print(f"  CNN LR: {lr_cnn:.2e}, Optimizer: {args.opt_cnn}")
            print(f"  Transformer LR: {lr_trf:.2e}, Optimizer: {args.opt_trf}")
    
    # Create schedulers - one per optimizer (canonical PyTorch pattern)
    warmup_steps = int(len(train_loader) * 1.5)  # ~1.5 epochs for longer windows
    warmup_epochs = math.ceil(warmup_steps / len(train_loader))
    
    warmup_schedulers = []
    cosine_schedulers = []
    
    for opt in optimizers:
        warmup_schedulers.append(torch.optim.lr_scheduler.LinearLR(
            opt, 
            start_factor=0.1, 
            total_iters=warmup_steps
        ))
        
        cosine_schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs - warmup_epochs, eta_min=1e-6
        ))
    
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
            print(f"Learning rates: " + ", ".join([f"{opt.param_groups[0]['lr']:.2e}" for opt in optimizers]))
            print(f"{'='*60}")
        
        # Determine if we're in warmup phase
        warmup_schedulers_to_use = warmup_schedulers if global_step < warmup_steps else None
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizers, device, epoch, 
            warmup_schedulers=warmup_schedulers_to_use, global_step=global_step,
            use_amp=args.amp
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Step cosine schedulers after warmup phase (per epoch)
        if global_step >= warmup_steps:
            for scheduler in cosine_schedulers:
                scheduler.step()
        
        # Only print and save on main process
        if is_main_process:
            base_model = model.module if hasattr(model, 'module') else model
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_metrics['loss']:.6f}")
            print(f"  Val Trans Error: {val_metrics['trans_error']*100:.2f} cm")
            print(f"  Val Rot Error: {val_metrics['rot_error']:.2f}°")
            print(f"  Adaptive Scales - Trans: {1.0/(base_model.trans_loss_avg.item()+1e-8):.2f}, Rot: {1.0/(base_model.rot_loss_avg.item()+1e-8):.2f}")
            print(f"  Learned Sigmas - σ_t: {torch.exp(base_model.log_sigma_t).item():.3f}, σ_r: {torch.exp(base_model.log_sigma_r).item():.3f}")
        
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
                    'optimizer_state_dicts': [opt.state_dict() for opt in optimizers],
                    'warmup_scheduler_state_dicts': [s.state_dict() for s in warmup_schedulers],
                    'cosine_scheduler_state_dicts': [s.state_dict() for s in cosine_schedulers],
                    'train_loss': train_loss,
                    'val_metrics': val_metrics,
                    'config': base_model.config.__dict__,
                    'global_step': global_step,
                    'use_dual_optimizers': use_dual_optimizers
                }
                torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
                print("  ✓ New best model saved")
            
            # Save latest checkpoint (always create fresh dictionary)
            latest_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dicts': [opt.state_dict() for opt in optimizers],
                'warmup_scheduler_state_dicts': [s.state_dict() for s in warmup_schedulers],
                'cosine_scheduler_state_dicts': [s.state_dict() for s in cosine_schedulers],
                'train_loss': train_loss,
                'val_metrics': val_metrics,
                'config': base_model.config.__dict__,
                'global_step': global_step,
                'use_dual_optimizers': use_dual_optimizers
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