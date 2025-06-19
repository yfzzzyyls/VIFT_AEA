#!/usr/bin/env python3
"""
Train IMU-only VIFT on Aria Everyday Activities dataset.
This trains only IMU Encoder + Pose Transformer (no visual encoder).

Usage:
    # Single GPU training
    python train_aria_from_scratch_imu_only.py --batch-size 8 --epochs 60

    # Multi-GPU training with DistributedDataParallel
    torchrun --nproc_per_node=4 train_aria_from_scratch_imu_only.py --distributed --batch-size 8

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
import warnings

# Silence NetworkX warning
try:
    import networkx as nx
    nx.config.use_numpy = True
except:
    pass

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.models.components.vsvio import TransformerVIO
from src.data.components.aria_raw_dataset import AriaRawDataModule

# -----------------------------------------------
# Aria hardware specifications
VIDEO_FPS = 20           # Aria RGB stream at 20 FPS
IMU_RATE_HZ = 1000       # Aria high-rate IMU at 1kHz
IMU_PER_FRAME = 11       # 11 samples per video frame (as processed by process_aria.py)
# Note: process_aria.py downsamples to 11 samples per interval
# -----------------------------------------------


class IMUOnlyVIO(nn.Module):
    """IMU-only version of VIFT for training from scratch.
    
    Architecture:
    - IMU encoder (1D CNN) -> 256-dim features
    - Pose transformer on 256-dim tokens
    - Separate translation and rotation heads
    """
    
    def __init__(self):
        super().__init__()
        
        # Model configuration
        class Config:
            # Sequence parameters
            seq_len = 21          # 21 frames → 20 pose transitions
            
            # Feature dimensions - IMU only
            i_f_len = 256         # IMU feature dimension
            embedding_dim = 256   # Same as i_f_len (no visual features)
            
            # IMU encoder parameters
            imu_dropout = 0.2
            
            # Transformer parameters
            num_layers = 4
            nhead = 8
            dim_feedforward = 2048
            dropout = 0.1
            
            # For compatibility with TransformerVIO
            img_w = 512
            img_h = 256
            v_f_len = 512
            rnn_hidden_size = 512
            rnn_dropout_between = 0.1
            rnn_dropout_out = 0.1
            fuse_method = 'cat'
        
        self.config = Config()
        
        # Use TransformerVIO backbone for IMU encoder
        self.backbone = TransformerVIO(self.config)
        
        # Create pose transformer that works on 256-dim IMU features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.embedding_dim,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation='relu',
            batch_first=True
        )
        self.pose_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers
        )
        
        # Prediction heads
        hidden_dim = self.config.embedding_dim // 2  # 128
        
        # Shared first layer
        self.shared_layer = nn.Sequential(
            nn.Linear(self.config.embedding_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Translation head
        self.trans_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)
        )
        
        # Rotation head (quaternion)
        self.rot_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4)
        )
        
        # Initialize quaternion head to favor identity rotation
        with torch.no_grad():
            self.rot_head[-1].bias[:3].fill_(0.0)  # qx, qy, qz = 0
            self.rot_head[-1].bias[3].fill_(1.0)    # qw = 1
            self.rot_head[-1].weight.data *= 0.01   # Small weights
        
        # Learnable uncertainty weights for automatic loss balancing
        self.s_t = nn.Parameter(torch.zeros(()))  # Translation log-variance
        self.s_r = nn.Parameter(torch.zeros(()))  # Rotation log-variance
    
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
        abs_dot_clamped = abs_dot.clamp(max=1.0 - 1e-7)
        angle = 2.0 * torch.atan2(
            torch.sqrt(1.0 - abs_dot_clamped**2), 
            abs_dot_clamped
        )
        
        return angle.mean()
    
    def forward(self, batch, epoch=0, batch_idx=0):
        """Forward pass using only IMU data."""
        seq_len = self.config.seq_len
        num_transitions = seq_len - 1
        
        # Only use IMU data (ignore images)
        imu = batch['imu']  # [B, num_transitions * samples_per_interval, 6]
        
        # Validate input shapes
        B = imu.shape[0]
        
        # Flexible IMU validation
        total_imu_samples = imu.shape[1]
        if total_imu_samples % num_transitions == 0:
            samples_per_interval = total_imu_samples // num_transitions
            if samples_per_interval not in [10, 11, 50]:
                print(f"WARNING: Unusual IMU sampling rate: {samples_per_interval} samples per interval")
        else:
            raise ValueError(f"IMU samples {total_imu_samples} not divisible by {num_transitions} transitions")
        
        assert imu.shape[2] == 6, f"Expected 6 IMU channels (ax,ay,az,gx,gy,gz), got {imu.shape[2]}"
        
        # Validate IMU data format on first batch
        if batch_idx == 0:
            gyro_magnitude = torch.norm(imu[:, :, 3:], dim=-1)
            max_gyro = gyro_magnitude.max().item()
            if max_gyro > 40.0:
                warnings.warn(f"Large gyroscope values detected (max: {max_gyro:.2f} rad/s). Normal range is < 30 rad/s.")
        
        # Get IMU features using backbone's Feature_net.inertial_encoder
        # We need to format the IMU data as windows for the encoder
        imu_windows = []
        for i in range(num_transitions):
            # Each transition has exactly samples_per_interval IMU samples
            start_idx = i * samples_per_interval
            end_idx = start_idx + samples_per_interval
            imu_window = imu[:, start_idx:end_idx, :]
            imu_windows.append(imu_window.unsqueeze(1))
        
        imu_formatted = torch.cat(imu_windows, dim=1)  # [B, num_transitions, samples_per_interval, 6]
        fi = self.backbone.Feature_net.inertial_encoder(imu_formatted)  # [B, 20, 256]
        
        # Validate feature shape
        assert fi.shape == (B, num_transitions, 256), f"Expected IMU features shape (B, {num_transitions}, 256), got {fi.shape}"
        
        # Apply pose transformer to IMU features
        transformer_out = self.pose_transformer(fi)  # [B, 20, 256]
        
        # Apply prediction heads
        batch_size, seq_len, feat_dim = transformer_out.shape
        features_flat = transformer_out.reshape(-1, feat_dim)  # [B*20, 256]
        
        # Shared layer
        shared_features = self.shared_layer(features_flat)  # [B*20, 128]
        
        # Separate predictions
        trans_flat = self.trans_head(shared_features)  # [B*20, 3]
        quat_flat = self.rot_head(shared_features)    # [B*20, 4]
        
        # Reshape
        trans = trans_flat.reshape(batch_size, seq_len, 3)  # [B, 20, 3]
        quat = quat_flat.reshape(batch_size, seq_len, 4)   # [B, 20, 4]
        
        # Normalize quaternion
        quat = F.normalize(quat, p=2, dim=-1)
        
        # Combine
        normalized_poses = torch.cat([trans, quat], dim=-1)  # [B, 20, 7]
        
        # If ground truth is provided, compute loss
        if 'gt_poses' in batch:
            gt_poses = batch['gt_poses']  # [B, 20, 7]
            
            # Split predictions and ground truth
            pred_trans = normalized_poses[:, :, :3]  # [B, 20, 3]
            pred_rot = normalized_poses[:, :, 3:]    # [B, 20, 4]
            gt_trans = gt_poses[:, :, :3]      # [B, 20, 3]
            gt_rot = gt_poses[:, :, 3:]        # [B, 20, 4]
            
            # Adaptive scale normalization
            scale_norm = gt_trans.reshape(-1, 3).norm(dim=-1).mean().clamp(min=1.0)
            trans_loss_raw = F.smooth_l1_loss(pred_trans / scale_norm, gt_trans / scale_norm, reduction='mean')
            rot_loss_raw = self._robust_geodesic_loss_stable(pred_rot, gt_rot)
            
            # Scale rotation loss
            ROT_SCALE = 10.0
            rot_loss_raw = rot_loss_raw * ROT_SCALE
            
            # Clamp and regularize log-variances
            s_t_c = torch.clamp(self.s_t, -2., 2.)
            s_r_c = torch.clamp(self.s_r, -2., 2.)
            
            # Homoscedastic uncertainty loss
            total_loss = (torch.exp(-s_t_c) * trans_loss_raw 
                          + torch.exp(-s_r_c) * rot_loss_raw 
                          + s_t_c + s_r_c 
                          + 1e-4 * (s_t_c**2 + s_r_c**2))
            
            # Path length prior
            pred_path_length = torch.sum(torch.norm(pred_trans, dim=-1), dim=1) / num_transitions  # [B]
            gt_path_length = torch.sum(torch.norm(gt_trans, dim=-1), dim=1) / num_transitions      # [B]
            path_loss = F.mse_loss(pred_path_length, gt_path_length)
            
            # Curriculum learning for path weight
            path_weight = min(0.02 + (0.08 * epoch / 5.0), 0.1)
            total_loss = total_loss + path_weight * path_loss
            
            return {
                'poses': normalized_poses,
                'total_loss': total_loss,
                'trans_loss_raw': trans_loss_raw,
                'rot_loss_raw': rot_loss_raw,
                'path_loss': path_loss,
                's_t': self.s_t.detach(),
                's_r': self.s_r.detach(),
                's_t_c': s_t_c.detach(),
                's_r_c': s_r_c.detach()
            }
        else:
            return {
                'poses': normalized_poses  # [B, 20, 7]
            }


def compute_loss(model, predictions, batch, is_training=True):
    """Process loss computation results."""
    if 'total_loss' not in predictions:
        raise ValueError("Loss not computed in forward pass. Make sure gt_poses is in batch.")
    
    total_loss = predictions['total_loss']
    trans_loss_raw = predictions['trans_loss_raw']
    rot_loss_raw = predictions['rot_loss_raw']
    
    if torch.isnan(trans_loss_raw) or torch.isnan(rot_loss_raw):
        return None
    
    return {
        'total_loss': total_loss,
        'trans_loss': trans_loss_raw,
        'rot_loss': rot_loss_raw,
        'path_loss': predictions.get('path_loss', 0.0),
        's_t': predictions.get('s_t', 0.0),
        's_r': predictions.get('s_r', 0.0),
        's_t_c': predictions.get('s_t_c', 0.0),
        's_r_c': predictions.get('s_r_c', 0.0)
    }


def train_epoch(model, dataloader, optimizers, device, epoch, warmup_steps=0, global_step=0, use_amp=False, initial_lrs=None):
    """Train one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    step = global_step
    
    # Create gradient scaler for mixed precision training
    if use_amp:
        if hasattr(torch.amp, 'GradScaler'):
            scaler = torch.amp.GradScaler(enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device - ONLY IMU and poses (no images)
        batch_cuda = {
            'imu': batch['imu'].to(device).float(),
            'gt_poses': batch['gt_poses'].to(device)
        }
        
        # Runtime check for IMU data
        imu_mag = torch.norm(batch_cuda['imu'][:, :, :3], dim=-1).max()
        if imu_mag > 150.0:  # ~15g threshold
            print(f"\n⚠️ WARNING: Large IMU acceleration detected in batch {batch_idx}: {imu_mag:.1f} m/s²")
            continue
        
        # Forward pass with automatic mixed precision
        if use_amp:
            if hasattr(torch.amp, 'autocast'):
                with torch.amp.autocast('cuda', enabled=True):
                    predictions = model(batch_cuda, epoch=epoch, batch_idx=batch_idx)
            else:
                with torch.cuda.amp.autocast():
                    predictions = model(batch_cuda, epoch=epoch, batch_idx=batch_idx)
        else:
            predictions = model(batch_cuda, epoch=epoch, batch_idx=batch_idx)
        
        # Compute loss
        loss_dict = compute_loss(model, predictions, batch_cuda)
        
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
            scaler.scale(loss).backward()
            
            for opt in optimizers:
                scaler.unscale_(opt)
            
            # Gradient clipping
            clip_value = 1.0 if epoch == 1 else 5.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
            
            for opt in optimizers:
                scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            
            clip_value = 1.0 if epoch == 1 else 5.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
            
            for opt in optimizers:
                opt.step()
        
        # Manual warmup learning rate adjustment
        if global_step < warmup_steps and initial_lrs is not None:
            warmup_factor = min(1.0, (global_step + 1) / warmup_steps)
            for opt_idx, opt in enumerate(optimizers):
                for param_group_idx, param_group in enumerate(opt.param_groups):
                    param_group['lr'] = initial_lrs[opt_idx][param_group_idx] * warmup_factor
        
        step += 1
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Calculate uncertainty weights
        exp_neg_s_t = torch.exp(-loss_dict['s_t_c']).item()
        exp_neg_s_r = torch.exp(-loss_dict['s_r_c']).item()
        
        # Warn if weights are diverging
        if exp_neg_s_t > 1000 or exp_neg_s_r > 1000:
            print(f"\n⚠️  WARNING: Uncertainty weights diverging! exp(-s_t)={exp_neg_s_t:.1f}, exp(-s_r)={exp_neg_s_r:.1f}")
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'trans': f"{loss_dict['trans_loss'].item():.4f}",
            'rot': f"{loss_dict['rot_loss'].item():.4f}",
            'path': f"{loss_dict['path_loss']:.4f}",
            'w_t': f"{exp_neg_s_t:.1f}",
            'w_r': f"{exp_neg_s_r:.1f}"
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
            # Move to device - ONLY IMU and poses
            batch_cuda = {
                'imu': batch['imu'].to(device).float(),
                'gt_poses': batch['gt_poses'].to(device)
            }
            
            # Forward pass
            predictions = model(batch_cuda, epoch=0)
            
            # Compute loss
            loss_dict = compute_loss(model, predictions, batch_cuda, is_training=False)
            
            if loss_dict is None:
                continue
            
            # Compute errors
            pred_poses = predictions['poses']  # [B, 20, 7]
            gt_poses = batch_cuda['gt_poses']  # [B, 20, 7]
            
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


def setup_distributed(args):
    """Initialize distributed training if requested."""
    if args.distributed:
        if args.local_rank == -1:
            args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        
        dist.init_process_group(
            backend=args.backend,
            init_method=args.init_method,
            world_size=int(os.environ.get('WORLD_SIZE', 1)),
            rank=int(os.environ.get('RANK', 0))
        )
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        print(f"[Rank {rank}/{world_size}] Distributed training initialized on GPU {args.local_rank}")
        return device, rank, world_size
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, 0, 1


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Train IMU-only VIFT on Aria data')
    parser.add_argument('--data-dir', type=str, default='aria_processed',
                        help='Directory with processed Aria data')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr-imu', type=float, default=None,
                        help='IMU encoder learning rate (defaults to --lr)')
    parser.add_argument('--lr-trf', type=float, default=None,
                        help='Transformer learning rate (defaults to --lr)')
    parser.add_argument('--opt-imu', type=str, default='adamw', choices=['adamw', 'sgd'],
                        help='Optimizer for IMU encoder')
    parser.add_argument('--opt-trf', type=str, default='adamw', choices=['adamw', 'sgd'],
                        help='Optimizer for Transformer')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_imu_only',
                        help='Directory to save checkpoints')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision training')
    
    # Distributed training arguments
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed training with DDP')
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='Local rank for distributed training')
    parser.add_argument('--backend', type=str, default='nccl',
                        help='Distributed backend to use')
    parser.add_argument('--init-method', type=str, default='env://',
                        help='URL to set up distributed training')
    
    args = parser.parse_args()
    
    # Setup distributed training
    device, rank, world_size = setup_distributed(args)
    is_main_process = rank == 0
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if is_main_process:
        print("\n" + "="*60)
        print("Training IMU-only VIFT on Aria Data")
        print("="*60)
        print("Model components:")
        print("- IMU Encoder (3-layer 1D CNN from TransformerVIO)")
        print("- Pose Transformer (4 layers, 8 heads)")
        print("- 256-dim IMU features only (no visual features)")
        print("- Quaternion output (3 trans + 4 quat)")
        print("- Multi-step prediction (all 20 transitions)")
        print(f"- IMU sampling: 11 samples per interval (220 Hz effective rate)")
        print(f"\nTraining Configuration:")
        print(f"- Distributed: {'Yes (DDP)' if args.distributed else 'No'}")
        print(f"- World size: {world_size}")
        print(f"- Batch size per GPU: {args.batch_size}")
        print(f"- Total batch size: {args.batch_size * world_size}")
        print(f"- Learning rate: {args.lr}")
        print(f"- Window stride: 1 (maximal overlap, causal prediction)")
        print(f"- Mixed precision (AMP): {'Enabled' if args.amp else 'Disabled'}")
        print("="*60 + "\n")
    
    # Create data module (same as full model)
    data_module = AriaRawDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequence_length=21,
        stride=1  # Maximum overlap
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
    model = IMUOnlyVIO()
    model = model.to(device)
    
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
    
    # Count parameters
    if is_main_process:
        base_model = model.module if hasattr(model, 'module') else model
        total_params = sum(p.numel() for p in base_model.parameters())
        trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        print(f"\nModel Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
    
    # Split parameters
    base_model = model.module if hasattr(model, 'module') else model
    imu_params = []
    trf_params = []
    uncert_params = []
    
    for name, param in base_model.named_parameters():
        if 'backbone.Feature_net.inertial_encoder' in name:
            imu_params.append(param)
        elif name in ['s_t', 's_r']:
            uncert_params.append(param)
        elif 'pose_transformer' in name or 'shared_layer' in name or 'trans_head' in name or 'rot_head' in name:
            trf_params.append(param)
        elif 'backbone.Feature_net.conv' in name or 'backbone.Feature_net.visual_head' in name:
            # Skip visual encoder params - they exist but won't be used
            continue
        elif 'backbone.Pose_net' in name:
            # Skip the backbone's pose network - we use our own
            continue
        elif 'backbone' in name:
            # Other backbone params - skip them
            continue
        else:
            # Warn about truly unclassified parameters
            print(f"WARNING: Unclassified parameter: {name}")
            # Add to transformer params as fallback
            trf_params.append(param)
    
    # Count parameters in each group
    imu_count = sum(p.numel() for p in imu_params)
    trf_count = sum(p.numel() for p in trf_params)
    uncert_count = sum(p.numel() for p in uncert_params)
    
    if is_main_process:
        print(f"\nParameter Split:")
        print(f"  IMU encoder parameters: {imu_count:,}")
        print(f"  Transformer parameters: {trf_count:,}")
        print(f"  Uncertainty parameters: {uncert_count:,}")
    
    # Learning rate scaling
    baseline_batch_size = 4
    baseline_world_size = 1
    
    global_batch_size = args.batch_size * world_size
    baseline_global_batch_size = baseline_batch_size * baseline_world_size
    global_batch_ratio = global_batch_size / baseline_global_batch_size
    
    # Different scaling strategies
    if args.opt_imu == 'sgd':
        lr_scale_imu = global_batch_ratio
    else:
        lr_scale_imu = math.sqrt(global_batch_ratio)
    
    if args.opt_trf == 'sgd':
        lr_scale_trf = global_batch_ratio
    else:
        lr_scale_trf = math.sqrt(global_batch_ratio)
    
    # Max LR ceiling
    max_lr_adamw = 1e-3
    max_lr_sgd = 5e-3
    
    # Apply scaling
    if args.lr_imu is not None:
        lr_imu = args.lr_imu
    else:
        base_lr_imu = 1e-4
        lr_imu = base_lr_imu * lr_scale_imu
        if args.opt_imu == 'adamw':
            lr_imu = min(lr_imu, max_lr_adamw)
        else:
            lr_imu = min(lr_imu, max_lr_sgd)
    
    if args.lr_trf is not None:
        lr_trf = args.lr_trf
    else:
        base_lr_trf = 5e-4
        lr_trf = base_lr_trf * lr_scale_trf
        if args.opt_trf == 'adamw':
            lr_trf = min(lr_trf, max_lr_adamw)
        else:
            lr_trf = min(lr_trf, max_lr_sgd)
    
    if is_main_process:
        print(f"\nLearning Rate Configuration:")
        print(f"  Batch size per GPU: {args.batch_size} (baseline: {baseline_batch_size})")
        print(f"  Global batch size: {global_batch_size} (baseline: {baseline_global_batch_size})")
        print(f"  Global batch ratio: {global_batch_ratio:.1f}x")
        print(f"  IMU LR: {lr_imu:.2e}")
        print(f"  Transformer LR: {lr_trf:.2e}")
        print(f"  Uncertainty params LR: {lr_trf * 0.1:.2e} (10% of transformer LR)")
    
    # Create optimizers
    if args.opt_imu == 'sgd' and args.opt_trf == 'adamw' and len(imu_params) > 0 and len(trf_params) > 0:
        optimizer_imu = torch.optim.SGD([
            {'params': imu_params, 'lr': lr_imu, 'momentum': 0.9}
        ], momentum=0.9)
        
        optimizer_trf = torch.optim.AdamW([
            {'params': trf_params, 'lr': lr_trf},
            {'params': uncert_params, 'lr': lr_trf * 0.1, 'weight_decay': 0.0}
        ], weight_decay=1e-4)
        
        optimizers = [optimizer_imu, optimizer_trf]
        use_dual_optimizers = True
        
        if is_main_process:
            print(f"  IMU encoder: {lr_imu:.2e}, Optimizer: sgd")
            print(f"  Transformer: {lr_trf:.2e}, Optimizer: adamw")
    else:
        if args.opt_imu == 'sgd':
            optimizer = torch.optim.SGD([
                {'params': imu_params, 'lr': lr_imu},
                {'params': trf_params, 'lr': lr_trf},
                {'params': uncert_params, 'lr': lr_trf * 0.1, 'weight_decay': 0.0}
            ], momentum=0.9)
        else:
            optimizer = torch.optim.AdamW([
                {'params': imu_params, 'lr': lr_imu},
                {'params': trf_params, 'lr': lr_trf},
                {'params': uncert_params, 'lr': lr_trf * 0.1, 'weight_decay': 0.0}
            ], weight_decay=1e-4)
        
        optimizers = [optimizer]
        use_dual_optimizers = False
    
    # Create schedulers
    batch_scale = args.batch_size / 4.0
    warmup_steps = int(len(train_loader) * 1.5 * batch_scale)
    warmup_epochs = math.ceil(warmup_steps / len(train_loader))
    
    # Store initial learning rates for manual warmup
    initial_lrs = []
    for opt in optimizers:
        initial_lrs.append([group['lr'] for group in opt.param_groups])
        # Start with low learning rate for warmup
        for group in opt.param_groups:
            group['lr'] = group['lr'] * 0.1
    
    cosine_schedulers = []
    for opt in optimizers:
        cosine_schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs - warmup_epochs, eta_min=1e-6
        ))
    
    # Training loop
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if is_main_process:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"Learning rates: " + ", ".join([f"{opt.param_groups[0]['lr']:.2e}" for opt in optimizers]))
            print(f"{'='*60}")
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizers, device, epoch, 
            warmup_steps=warmup_steps, global_step=global_step,
            use_amp=args.amp, initial_lrs=initial_lrs
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Step cosine schedulers after warmup
        if global_step >= warmup_steps:
            # First time transitioning from warmup to cosine, restore full LR
            if global_step - len(train_loader) < warmup_steps:
                for opt_idx, opt in enumerate(optimizers):
                    for param_group_idx, param_group in enumerate(opt.param_groups):
                        param_group['lr'] = initial_lrs[opt_idx][param_group_idx]
            
            for scheduler in cosine_schedulers:
                scheduler.step()
        
        # Print and save
        if is_main_process:
            base_model = model.module if hasattr(model, 'module') else model
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_metrics['loss']:.6f}")
            print(f"  Val Trans Error: {val_metrics['trans_error']*100:.2f} cm")
            print(f"  Val Rot Error: {val_metrics['rot_error']:.2f}°")
            print(f"  Adaptive Weights - Trans: {torch.exp(-base_model.s_t).item():.3f}, Rot: {torch.exp(-base_model.s_r).item():.3f}")
        
        # Save checkpoints
        if is_main_process:
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            base_model = model.module if hasattr(model, 'module') else model
            
            # Save best
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
            
            # Save latest
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
    
    cleanup_distributed()


if __name__ == "__main__":
    main()