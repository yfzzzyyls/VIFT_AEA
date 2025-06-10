#!/usr/bin/env python3
"""
Improved training script with model architecture selection
Supports both MultiHeadVIOModelSeparate and original VIFT PoseTransformer
"""

import os
import sys
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    RichProgressBar,
    LearningRateMonitor
)
from rich.console import Console
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
from typing import Dict, Optional, Tuple

# Add src to path
sys.path.append('src')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.multihead_vio_separate import MultiHeadVIOModelSeparate
from src.models.multihead_vio_separate_fixed import MultiHeadVIOModelSeparate as MultiHeadVIOModelSeparateFixed
from src.models.multihead_vio import MultiHeadVIOModel
from src.models.components.pose_transformer import PoseTransformer
from src.models.vift_quaternion_module import VIFTQuaternionModel
from src.metrics.arvr_loss_wrapper import ARVRLossWrapper
from torchmetrics import MeanAbsoluteError

console = Console()


class SeparateFeatureDataset(Dataset):
    """Dataset for pre-extracted visual and IMU features"""
    
    def __init__(self, data_dir: str, augment: bool = False):
        self.data_dir = Path(data_dir)
        self.augment = augment
        
        # Get all sample IDs
        all_files = list(self.data_dir.glob("*_visual.npy"))
        self.sample_ids = sorted([int(f.stem.split('_')[0]) for f in all_files])
        
        console.print(f"Found {len(self.sample_ids)} samples in {data_dir}")
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # Load features
        visual_features = np.load(self.data_dir / f"{sample_id}_visual.npy")
        imu_features = np.load(self.data_dir / f"{sample_id}_imu.npy")
        poses = np.load(self.data_dir / f"{sample_id}_gt.npy")
        
        # Convert to tensors
        visual_features = torch.from_numpy(visual_features).float()
        imu_features = torch.from_numpy(imu_features).float()
        poses = torch.from_numpy(poses).float()
        
        # CRITICAL FIX: The pretrained data is in METERS, not centimeters!
        # Scale translations from meters to centimeters
        poses[:, :3] *= 100.0
        
        # Apply augmentation if requested
        if self.augment:
            # Add small noise to features
            visual_features += torch.randn_like(visual_features) * 0.01
            imu_features += torch.randn_like(imu_features) * 0.01
        
        return visual_features, imu_features, poses


class VIFTLightningWrapper(L.LightningModule):
    """
    Lightning wrapper for the original VIFT PoseTransformer
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        embedding_dim: int = 128,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        rotation_weight: float = 1.0,
        translation_weight: float = 1.0,
        optimizer_type: str = 'adamw',
        scheduler_type: str = 'onecycle',
        warmup_steps: int = 500
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Create the original VIFT model
        self.model = PoseTransformer(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Loss function
        self.arvr_loss = ARVRLossWrapper(use_log_scale=True, use_weighted_loss=False)
        
        # Metrics
        self.train_rot_mae = MeanAbsoluteError()
        self.train_trans_mae = MeanAbsoluteError()
        self.val_rot_mae = MeanAbsoluteError()
        self.val_trans_mae = MeanAbsoluteError()
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass compatible with the original VIFT
        """
        # Extract features
        visual_features = batch['visual_features']
        imu_features = batch['imu_features']
        poses = batch['poses']
        
        # Concatenate visual and IMU features as expected by VIFT
        B, seq_len, _ = visual_features.shape
        combined_features = torch.cat([visual_features, imu_features], dim=-1)  # [B, seq_len, 768]
        
        # Create the batch tuple expected by VIFT forward method
        batch_tuple = (combined_features, None, None)
        
        # Forward through VIFT (it expects gt but doesn't use it in forward)
        output = self.model(batch_tuple, None)  # [B, seq_len, 6]
        
        # Convert output format: VIFT outputs [tx, ty, tz, rx, ry, rz]
        # We need to convert to quaternions for consistency
        translation = output[:, :, :3]
        
        # Convert VIFT's Euler angles to quaternions
        euler_angles = output[:, :, 3:6]  # [B, seq_len, 3]
        B, seq_len, _ = euler_angles.shape
        
        # Reshape for batch processing
        euler_flat = euler_angles.reshape(-1, 3)
        
        # Convert Euler to quaternion using scipy
        from scipy.spatial.transform import Rotation
        quaternions = []
        for i in range(euler_flat.shape[0]):
            r = Rotation.from_euler('xyz', euler_flat[i].detach().cpu().numpy())
            q = r.as_quat()  # Returns [x, y, z, w]
            quaternions.append(q)
        
        quaternions = torch.tensor(quaternions, device=euler_angles.device, dtype=euler_angles.dtype)
        rotation = quaternions.reshape(B, seq_len, 4)
        
        return {
            'translation': translation,
            'rotation': rotation,
            'vift_output': output
        }
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                     batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute loss for VIFT model"""
        # Get predictions
        vift_output = predictions['vift_output']  # [B, seq_len, 6]
        
        # Get targets
        poses = batch['poses']  # [B, seq_len, 7]
        target_translation = poses[:, :, :3]
        target_rotation = poses[:, :, 3:7]
        
        # VIFT outputs translation and euler angles, so we compute translation loss
        B, seq_len, _ = vift_output.shape
        
        pred_trans = vift_output[:, :, :3].reshape(-1, 3).contiguous()
        target_trans = target_translation.reshape(-1, 3).contiguous()
        
        # For rotation, use the quaternion predictions
        pred_rot = predictions['rotation'].reshape(-1, 4).contiguous()
        target_rot = target_rotation.reshape(-1, 4).contiguous()
        
        # Compute losses
        loss_dict = self.arvr_loss(
            pred_rotation=pred_rot,
            target_rotation=target_rot,
            pred_translation=pred_trans,
            target_translation=target_trans
        )
        
        # Weight losses
        loss_dict['rotation_loss'] *= self.hparams.rotation_weight
        loss_dict['translation_loss'] *= self.hparams.translation_weight
        
        return loss_dict
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        loss_dict = self.compute_loss(predictions, batch)
        
        total_loss = loss_dict.get('total_loss', sum(v for k, v in loss_dict.items() if k != 'total_loss'))
        
        # Log losses
        self.log('train/total_loss', total_loss, prog_bar=True)
        for key, value in loss_dict.items():
            if key != 'total_loss':
                self.log(f'train/{key}', value)
        
        # Update metrics
        with torch.no_grad():
            vift_output = predictions['vift_output']
            poses = batch['poses']
            
            pred_trans = vift_output[:, :, :3].reshape(-1, 3).contiguous()
            target_trans = poses[:, :, :3].reshape(-1, 3).contiguous()
            pred_rot = predictions['rotation'].reshape(-1, 4).contiguous()
            target_rot = poses[:, :, 3:7].reshape(-1, 4).contiguous()
            
            self.train_trans_mae(pred_trans, target_trans)
            self.train_rot_mae(pred_rot, target_rot)
            
            self.log('train/trans_mae', self.train_trans_mae, prog_bar=True)
            self.log('train/rot_mae', self.train_rot_mae, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        loss_dict = self.compute_loss(predictions, batch)
        
        total_loss = loss_dict.get('total_loss', sum(v for k, v in loss_dict.items() if k != 'total_loss'))
        
        # Log losses
        self.log('val/total_loss', total_loss, prog_bar=True)
        for key, value in loss_dict.items():
            if key != 'total_loss':
                self.log(f'val/{key}', value)
        
        # Update metrics
        with torch.no_grad():
            vift_output = predictions['vift_output']
            poses = batch['poses']
            
            pred_trans = vift_output[:, :, :3].reshape(-1, 3).contiguous()
            target_trans = poses[:, :, :3].reshape(-1, 3).contiguous()
            pred_rot = predictions['rotation'].reshape(-1, 4).contiguous()
            target_rot = poses[:, :, 3:7].reshape(-1, 4).contiguous()
            
            self.val_trans_mae(pred_trans, target_trans)
            self.val_rot_mae(pred_rot, target_rot)
            
            self.log('val/trans_mae', self.val_trans_mae, prog_bar=True)
            self.log('val/rot_mae', self.val_rot_mae, prog_bar=True)
        
        return total_loss
    
    def configure_optimizers(self):
        # Select optimizer
        if hasattr(self.hparams, 'optimizer_type'):
            opt_type = self.hparams.optimizer_type
        else:
            opt_type = 'adamw'
            
        if opt_type == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-7
            )
        elif opt_type == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9
            )
        else:  # adamw
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-7
            )
        
        # Select scheduler
        if hasattr(self.hparams, 'scheduler_type'):
            sched_type = self.hparams.scheduler_type
        else:
            sched_type = 'onecycle'
            
        if sched_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.estimated_stepping_batches,
                eta_min=1e-6
            )
        elif sched_type == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=self.trainer.estimated_stepping_batches
            )
        elif sched_type == 'constant':
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1.0
            )
        else:  # onecycle
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate * 10,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=10,
                final_div_factor=100
            )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    visual_features, imu_features, poses = zip(*batch)
    return {
        'visual_features': torch.stack(visual_features),
        'imu_features': torch.stack(imu_features),
        'poses': torch.stack(poses)
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Train VIO models with different architectures')
    
    # Model selection
    parser.add_argument('--model', type=str, choices=['multihead', 'multihead_fixed', 'vift_original', 'vift_quaternion'], 
                        default='multihead', help='Model architecture to use')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Loss weights
    parser.add_argument('--rotation-weight', type=float, default=0.1, help='Weight for rotation loss')
    parser.add_argument('--translation-weight', type=float, default=10.0, help='Weight for translation loss')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='aria_latent_data_pretrained', 
                        help='Directory containing the data')
    
    # Additional hyperparameters for optimization
    parser.add_argument('--optimizer', type=str, default='adamw', 
                        choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='onecycle',
                        choices=['cosine', 'linear', 'constant', 'onecycle'], help='Learning rate scheduler')
    parser.add_argument('--gradient-accumulation', type=int, default=1, 
                        help='Gradient accumulation steps')
    parser.add_argument('--warmup-steps', type=int, default=500, help='Warmup steps for scheduler')
    
    # Experiment tracking
    parser.add_argument('--checkpoint-dir', type=str, default=None, 
                        help='Directory to save checkpoints')
    parser.add_argument('--experiment-name', type=str, default=None, 
                        help='Name for this experiment')
    
    # Training configuration
    parser.add_argument('--log-every-n-steps', type=int, default=5,
                        help='Log metrics every n steps (default: 5)')
    parser.add_argument('--val-check-interval', type=float, default=1.0,
                        help='Run validation every n epochs (default: 1.0)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    console.rule(f"[bold cyan]🚀 VIO Training - {args.model.upper()} Model[/bold cyan]")
    
    # Check if data directory exists
    if not Path(args.data_dir).exists():
        console.print(f"[red]Error: Data directory not found: {args.data_dir}[/red]")
        console.print("\n[yellow]Did you run feature extraction first?[/yellow]")
        console.print("Run this command first:")
        console.print(f"   python generate_all_pretrained_latents_fixed.py \\")
        console.print(f"       --processed-dir /mnt/ssd_ext/incSeg-data/aria_processed \\")
        console.print(f"       --output-dir {args.data_dir}")
        return
    
    # Configuration
    data_dir = args.data_dir
    learning_rate = args.lr
    batch_size = args.batch_size
    max_epochs = args.epochs
    
    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  Model: {args.model}")
    console.print(f"  Epochs: {max_epochs}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Learning rate: {learning_rate}")
    console.print(f"  Hidden dim: {args.hidden_dim}")
    console.print(f"  Dropout: {args.dropout}")
    console.print(f"  Optimizer: {args.optimizer}")
    console.print(f"  Scheduler: {args.scheduler}")
    console.print(f"  Gradient accumulation: {args.gradient_accumulation}")
    
    # Set seed
    L.seed_everything(42, workers=True)
    
    # Create model based on selection
    if args.model == 'multihead':
        console.print("\n[bold]Using MultiHeadVIOModelSeparate[/bold]")
        model = MultiHeadVIOModelSeparate(
            visual_dim=512,
            imu_dim=256,
            hidden_dim=args.hidden_dim,
            num_shared_layers=2,
            num_specialized_layers=2,
            num_heads=args.num_heads,
            dropout=args.dropout,
            learning_rate=learning_rate,
            weight_decay=args.weight_decay,
            sequence_length=10,
            rotation_weight=args.rotation_weight,
            translation_weight=args.translation_weight,
            optimizer_type=args.optimizer,
            scheduler_type=args.scheduler,
            warmup_steps=args.warmup_steps
        )
    elif args.model == 'multihead_fixed':
        console.print("\n[bold]Using MultiHeadVIOModel[/bold]")
        console.print("[green]✓ Fixed rotation head (no ReLU)[/green]")
        console.print("[green]✓ Proper weight initialization[/green]")
        console.print("[green]✓ Geodesic rotation metric[/green]")
        console.print("[green]✓ Specialized heads for rotation and translation[/green]")
        model = MultiHeadVIOModel(
            visual_dim=512,
            imu_dim=256,
            hidden_dim=args.hidden_dim,
            num_shared_layers=2,
            num_specialized_layers=2,
            num_heads=args.num_heads,
            dropout=args.dropout,
            learning_rate=learning_rate,
            weight_decay=args.weight_decay,
            sequence_length=10,
            rotation_weight=args.rotation_weight,
            translation_weight=args.translation_weight,
            optimizer_type=args.optimizer,
            scheduler_type=args.scheduler,
            warmup_steps=args.warmup_steps
        )
    elif args.model == 'vift_original':
        console.print("\n[bold]Using Original VIFT PoseTransformer[/bold]")
        model = VIFTLightningWrapper(
            input_dim=768,  # 512 visual + 256 IMU
            embedding_dim=args.hidden_dim,
            num_layers=2,
            nhead=args.num_heads,
            dim_feedforward=args.hidden_dim * 4,
            dropout=args.dropout,
            learning_rate=learning_rate,
            weight_decay=args.weight_decay,
            rotation_weight=args.rotation_weight,
            translation_weight=args.translation_weight,
            optimizer_type=args.optimizer,
            scheduler_type=args.scheduler,
            warmup_steps=args.warmup_steps
        )
    else:  # vift_quaternion
        console.print("\n[bold]Using VIFT Quaternion Model[/bold]")
        console.print("[green]✓ Outputs quaternions instead of Euler angles[/green]")
        console.print("[green]✓ Avoids gimbal lock and discontinuities[/green]")
        console.print("[green]✓ Better rotation representation[/green]")
        model = VIFTQuaternionModel(
            input_dim=768,  # 512 visual + 256 IMU
            embedding_dim=args.hidden_dim,
            num_layers=2,
            nhead=args.num_heads,
            dim_feedforward=args.hidden_dim * 4,
            dropout=args.dropout,
            learning_rate=learning_rate,
            weight_decay=args.weight_decay,
            rotation_weight=args.rotation_weight,
            translation_weight=args.translation_weight,
            optimizer_type=args.optimizer,
            scheduler_type=args.scheduler,
            warmup_steps=args.warmup_steps
        )
    
    # Create datasets with augmentation
    console.print("\n[bold]Loading data...[/bold]")
    train_dataset = SeparateFeatureDataset(f"{data_dir}/train", augment=True)
    val_dataset = SeparateFeatureDataset(f"{data_dir}/val", augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    console.print(f"  Training samples: {len(train_dataset):,}")
    console.print(f"  Validation samples: {len(val_dataset):,}")
    
    # Check model size
    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"\n[bold]Model size:[/bold] {total_params/1e6:.1f}M parameters")
    
    # Setup checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = Path("logs") / f"checkpoints_{args.model}"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Logger
    logger = TensorBoardLogger(
        save_dir="logs" if not args.experiment_name else str(Path(args.checkpoint_dir).parent.parent / "logs"),
        name=args.experiment_name or f"{args.model}_training",
        version=args.experiment_name or f"lr_{learning_rate}_hd_{args.hidden_dim}"
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val/total_loss",
            mode="min",
            save_top_k=3,
            dirpath=str(checkpoint_dir),
            filename="epoch_{epoch:03d}_val_loss_{val/total_loss:.6f}",
            save_last=True,
            verbose=True
        ),
        EarlyStopping(
            monitor="val/total_loss",
            patience=20,  # More patience for proper convergence
            mode="min",
            verbose=True,
            min_delta=0.0001
        ),
        LearningRateMonitor(logging_interval='step'),
        RichProgressBar(refresh_rate=10)
    ]
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,  # Use single GPU for now
        strategy="auto",  # Let Lightning choose the strategy
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.gradient_accumulation,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,  # Check validation every full epoch
        deterministic=True
    )
    
    # Train
    console.rule(f"[bold cyan]🏃 Starting Training - {args.model.upper()}[/bold cyan]")
    console.print("\n[bold yellow]Expected behavior:[/bold yellow]")
    console.print("  • Loss should start around 1-10 range")
    console.print("  • Model should predict meaningful motions")
    console.print("  • Training should take 20-40 epochs to converge")
    console.print("  • Validation loss should decrease gradually")
    console.print()
    
    trainer.fit(model, train_loader, val_loader)
    
    console.rule("[bold green]✅ Training Completed![/bold green]")
    
    if trainer.checkpoint_callback:
        best_path = trainer.checkpoint_callback.best_model_path
        best_loss = trainer.checkpoint_callback.best_model_score
        
        console.print(f"\n[bold]Best checkpoint:[/bold] {best_path}")
        console.print(f"[bold]Best validation loss:[/bold] {best_loss:.6f}")
        
        console.print("\n[bold cyan]Next steps:[/bold cyan]")
        console.print("1. Check TensorBoard for loss curves:")
        console.print(f"   tensorboard --logdir logs/{args.model}_training")
        
        console.print("\n2. Evaluate on test sequences:")
        console.print(f"   python inference_full_sequence.py \\")
        console.print(f"       --sequence-id all \\")
        console.print(f"       --checkpoint {best_path} \\")
        console.print(f"       --processed-dir /mnt/ssd_ext/incSeg-data/aria_processed")
        
        console.print("\n3. Or evaluate a single sequence:")
        console.print(f"   python inference_full_sequence.py \\")
        console.print(f"       --sequence-id 123 \\")
        console.print(f"       --checkpoint {best_path} \\")
        console.print(f"       --processed-dir /mnt/ssd_ext/incSeg-data/aria_processed")


if __name__ == "__main__":
    main()
