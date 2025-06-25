#!/usr/bin/env python3
"""
Training script with true shared memory using torch.multiprocessing.spawn
"""

import os
import sys
import time
import math
import random
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from models.flownet_lstm_transformer import FlowNetLSTMTransformer
from data.aria_dataset_mmap_shared import AriaDatasetMMapShared, collate_mmap_shared
from configs.flownet_lstm_transformer_config import get_config, Config
from utils.losses import compute_pose_loss, quaternion_geodesic_loss
from utils.metrics import compute_trajectory_metrics


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    print(f"[Rank {rank}/{world_size}] Distributed training initialized")
    return device


def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()


def create_model(config: Config, device: torch.device, pretrained_path: Optional[str] = None, 
                 train_transformer_only: bool = False) -> nn.Module:
    """Create and initialize the model."""
    model = FlowNetLSTMTransformer(config.model)
    
    # Load pretrained encoders if specified
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained encoders from {pretrained_path}")
        pretrained_state = torch.load(pretrained_path, map_location='cpu')
        
        # Extract encoder weights from pretrained model
        visual_encoder_dict = {}
        imu_encoder_dict = {}
        
        for k, v in pretrained_state.items():
            if k.startswith('visual_encoder.'):
                # Map old visual encoder names to new model
                new_key = k.replace('visual_encoder.', '')
                visual_encoder_dict[new_key] = v
            elif k.startswith('imu_encoder.'):
                # Map old IMU encoder names to new model
                new_key = k.replace('imu_encoder.', '')
                imu_encoder_dict[new_key] = v
        
        # Load visual encoder weights
        if visual_encoder_dict:
            try:
                model.visual_encoder.load_state_dict(visual_encoder_dict, strict=False)
                print(f"Loaded {len(visual_encoder_dict)} visual encoder parameters")
            except Exception as e:
                print(f"Warning: Could not load all visual encoder weights: {e}")
        
        # Load IMU encoder weights
        if imu_encoder_dict:
            try:
                model.imu_encoder.load_state_dict(imu_encoder_dict, strict=False)
                print(f"Loaded {len(imu_encoder_dict)} IMU encoder parameters")
            except Exception as e:
                print(f"Warning: Could not load all IMU encoder weights: {e}")
        
        # Freeze encoders if only training transformer
        if train_transformer_only:
            print("Freezing visual and IMU encoders - only training pose transformer")
            for param in model.visual_encoder.parameters():
                param.requires_grad = False
            for param in model.imu_encoder.parameters():
                param.requires_grad = False
    
    model = model.to(device)
    
    # Initialize weights for unfrozen parts
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Set forget gate bias to 1
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1)
    
    # Only initialize pose predictor and output heads if using pretrained encoders
    if pretrained_path and train_transformer_only:
        model.pose_predictor.apply(init_weights)
        model.translation_head.apply(init_weights)
        model.rotation_head.apply(init_weights)
    else:
        model.apply(init_weights)
    
    return model


def create_dataloaders(config: Config, rank: int, world_size: int):
    """Create training and validation dataloaders with shared memory."""
    # Training dataset
    train_dataset = AriaDatasetMMapShared(
        data_dir=config.data.data_dir,
        split='train',
        sequence_length=config.data.sequence_length,
        stride=config.data.stride,
        image_size=config.data.image_size,
        rank=rank,
        world_size=world_size
    )
    
    # Validation dataset
    val_dataset = AriaDatasetMMapShared(
        data_dir=config.data.data_dir,
        split='val',
        sequence_length=config.data.sequence_length,
        stride=config.data.stride * 5,  # Less dense sampling
        image_size=config.data.image_size,
        rank=rank,
        world_size=world_size
    )
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_mmap_shared
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_mmap_shared
    )
    
    return train_loader, val_loader, train_sampler, train_dataset, val_dataset


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Config,
    device: torch.device,
    epoch: int,
    scaler: Optional[GradScaler] = None,
    rank: int = 0
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    trans_loss_sum = 0
    rot_loss_sum = 0
    scale_loss_sum = 0
    num_batches = 0
    
    # Progress bar only on main process
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to GPU
        images = batch['images'].to(device)
        poses_gt = batch['poses'].to(device)
        sequence_lengths = batch['sequence_lengths'].to(device)
        
        # Handle IMU data
        if 'imu_padded' in batch:
            imu_padded = batch['imu_padded'].to(device)
            imu_lengths = batch['imu_lengths'].to(device)
            # Convert to list format for model
            imu_sequences = []
            for b in range(imu_padded.shape[0]):
                seq_list = []
                for t in range(imu_padded.shape[1]):
                    actual_len = imu_lengths[b, t]
                    if actual_len > 0:
                        seq_list.append(imu_padded[b, t, :actual_len])
                imu_sequences.append(seq_list)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if config.training.use_amp and scaler is not None:
            with autocast('cuda'):
                outputs = model(images, imu_sequences)
                loss_dict = compute_pose_loss(
                    outputs['poses'],
                    poses_gt,
                    sequence_lengths,
                    translation_weight=config.training.translation_weight,
                    rotation_weight=config.training.rotation_weight,
                    scale_weight=config.training.scale_weight
                )
                loss = loss_dict['total_loss']
        else:
            outputs = model(images, imu_sequences)
            loss_dict = compute_pose_loss(
                outputs['poses'],
                poses_gt,
                sequence_lengths,
                translation_weight=config.training.translation_weight,
                rotation_weight=config.training.rotation_weight,
                scale_weight=config.training.scale_weight
            )
            loss = loss_dict['total_loss']
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"NaN loss detected at batch {batch_idx}, skipping...")
            continue
        
        # Backward pass
        if config.training.use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        trans_loss_sum += loss_dict['translation_loss'].item()
        rot_loss_sum += loss_dict['rotation_loss'].item()
        scale_loss_sum += loss_dict['scale_loss'].item()
        num_batches += 1
        
        # Update progress bar (convert rotation from radians to degrees)
        if rank == 0 and batch_idx % config.training.log_every == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'trans(m)': f"{loss_dict['translation_loss'].item():.4f}",
                'rot(°)': f"{loss_dict['rotation_loss'].item() * 180.0 / math.pi:.2f}",
                'scale': f"{loss_dict['scale_loss'].item():.4f}"
            })
    
    # Average losses
    avg_losses = {
        'total_loss': total_loss / num_batches,
        'translation_loss': trans_loss_sum / num_batches,
        'rotation_loss': rot_loss_sum / num_batches,
        'scale_loss': scale_loss_sum / num_batches
    }
    
    return avg_losses


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    config: Config,
    device: torch.device,
    rank: int = 0
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    trans_loss_sum = 0
    rot_loss_sum = 0
    scale_loss_sum = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", disable=(rank != 0)):
            # Move data to GPU
            images = batch['images'].to(device)
            poses_gt = batch['poses'].to(device)
            sequence_lengths = batch['sequence_lengths'].to(device)
            
            # Handle IMU data
            if 'imu_padded' in batch:
                imu_padded = batch['imu_padded'].to(device)
                imu_lengths = batch['imu_lengths'].to(device)
                # Convert to list format
                imu_sequences = []
                for b in range(imu_padded.shape[0]):
                    seq_list = []
                    for t in range(imu_padded.shape[1]):
                        actual_len = imu_lengths[b, t]
                        if actual_len > 0:
                            seq_list.append(imu_padded[b, t, :actual_len])
                    imu_sequences.append(seq_list)
            
            # Forward pass
            outputs = model(images, imu_sequences)
            loss_dict = compute_pose_loss(
                outputs['poses'],
                poses_gt,
                sequence_lengths,
                translation_weight=config.training.translation_weight,
                rotation_weight=config.training.rotation_weight,
                scale_weight=config.training.scale_weight
            )
            
            # Update statistics
            total_loss += loss_dict['total_loss'].item()
            trans_loss_sum += loss_dict['translation_loss'].item()
            rot_loss_sum += loss_dict['rotation_loss'].item()
            scale_loss_sum += loss_dict['scale_loss'].item()
            num_batches += 1
    
    # Average losses
    avg_losses = {
        'total_loss': total_loss / num_batches,
        'translation_loss': trans_loss_sum / num_batches,
        'rotation_loss': rot_loss_sum / num_batches,
        'scale_loss': scale_loss_sum / num_batches
    }
    
    return avg_losses


def get_memory_usage():
    """Get current memory usage in GB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024**3


def train_worker(rank: int, world_size: int, config: Config):
    """Training worker function for each process."""
    # Setup distributed training
    device = setup_distributed(rank, world_size)
    is_main_process = rank == 0
    
    # Set random seed
    set_seed(config.seed + rank)
    
    print(f"[Rank {rank}] Initial memory: {get_memory_usage():.2f} GB")
    
    # Create output directories
    if is_main_process:
        checkpoint_dir = Path(config.training.checkpoint_dir) / config.experiment_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        import json
        with open(checkpoint_dir / 'config.json', 'w') as f:
            json.dump(config.__dict__, f, indent=2, default=str)
    
    # Print configuration
    if is_main_process:
        print("\n" + "="*80)
        print("FlowNet-LSTM-Transformer Training (Shared Memory)")
        print("="*80)
        print(f"Experiment: {config.experiment_name}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Using memory-mapped shared data")
        print(f"Batch size per GPU: {config.data.batch_size}")
        print(f"Total batch size: {config.data.batch_size * world_size}")
        print("="*80 + "\n")
    
    # Create model with optional pretrained encoders
    pretrained_path = getattr(config.training, 'pretrained_path', None)
    train_transformer_only = getattr(config.training, 'train_transformer_only', False)
    
    model = create_model(config, device, pretrained_path, train_transformer_only)
    print(f"[Rank {rank}] After model creation: {get_memory_usage():.2f} GB")
    
    # Print trainable parameters
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        if train_transformer_only:
            print(f"Training pose transformer only ({trainable_params/total_params*100:.1f}% of model)")
    
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    print(f"[Rank {rank}] After DDP wrap: {get_memory_usage():.2f} GB")
    
    # Create optimizer - only for parameters that require gradients
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    print(f"[Rank {rank}] After optimizer: {get_memory_usage():.2f} GB")
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.num_epochs,
        eta_min=1e-6
    )
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler('cuda') if config.training.use_amp else None
    
    # Create dataloaders
    train_loader, val_loader, train_sampler, train_dataset, val_dataset = create_dataloaders(
        config, rank, world_size
    )
    print(f"[Rank {rank}] After dataloaders: {get_memory_usage():.2f} GB")
    
    # Training loop
    best_val_loss = float('inf')
    
    try:
        for epoch in range(1, config.training.num_epochs + 1):
            # Update sampler epoch
            train_sampler.set_epoch(epoch)
            
            # Train
            train_losses = train_epoch(
                model, train_loader, optimizer, config, device, epoch, scaler, rank
            )
            
            # Validate
            if epoch % config.training.validate_every == 0:
                val_losses = validate(model, val_loader, config, device, rank)
                
                if is_main_process:
                    print(f"\nEpoch {epoch} Summary:")
                    print(f"Train Loss: {train_losses['total_loss']:.4f}")
                    print(f"  - Translation (m): {train_losses['translation_loss']:.4f}")
                    print(f"  - Rotation (°): {train_losses['rotation_loss'] * 180.0 / math.pi:.2f}")
                    print(f"  - Scale: {train_losses['scale_loss']:.4f}")
                    print(f"Val Loss: {val_losses['total_loss']:.4f}")
                    print(f"  - Translation (m): {val_losses['translation_loss']:.4f}")
                    print(f"  - Rotation (°): {val_losses['rotation_loss'] * 180.0 / math.pi:.2f}")
                    print(f"  - Scale: {val_losses['scale_loss']:.4f}")
                    
                    # Save best model
                    if val_losses['total_loss'] < best_val_loss:
                        best_val_loss = val_losses['total_loss']
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'train_losses': train_losses,
                            'val_losses': val_losses,
                            'config': config
                        }
                        torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
                        print(f"Saved best model (val_loss: {best_val_loss:.4f})")
            
            # Save latest checkpoint
            if is_main_process:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_losses': train_losses,
                    'config': config,
                    'best_val_loss': best_val_loss
                }
                torch.save(checkpoint, checkpoint_dir / 'latest_checkpoint.pt')
            
            # Update learning rate
            scheduler.step()
    
    finally:
        # Clean up
        if rank == 0:
            # Optionally clean up memory-mapped files
            # train_dataset.cleanup()
            # val_dataset.cleanup()
            pass
        
        cleanup()
    
    if is_main_process:
        print("\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")


def main():
    """Main function to launch distributed training."""
    # Get configuration
    config = get_config()
    
    # Determine number of GPUs
    if config.training.distributed:
        world_size = torch.cuda.device_count()
        print(f"Starting distributed training with {world_size} GPUs")
        
        # Spawn processes
        mp.spawn(
            train_worker,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU training
        print("Starting single GPU training")
        train_worker(0, 1, config)


if __name__ == "__main__":
    # Required for multiprocessing
    mp.set_start_method('spawn', force=True)
    main()