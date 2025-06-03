#!/usr/bin/env python3
"""
Training script for separate visual and IMU features
This properly uses the 256-dimensional IMU features from the pretrained encoder
"""

import os
import sys
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    RichProgressBar,
    LearningRateMonitor
)
from rich.console import Console
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.append('src')

from src.models.multihead_vio_separate import MultiHeadVIOModelSeparate

console = Console()


class SeparateFeatureDataset(Dataset):
    """Dataset that loads separate visual and IMU features"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
        # Find all samples
        self.samples = []
        i = 0
        consecutive_misses = 0
        
        while consecutive_misses < 100:
            visual_path = os.path.join(data_dir, f"{i}_visual.npy")
            imu_path = os.path.join(data_dir, f"{i}_imu.npy")
            gt_path = os.path.join(data_dir, f"{i}_gt.npy")
            
            if os.path.exists(visual_path) and os.path.exists(imu_path) and os.path.exists(gt_path):
                self.samples.append(i)
                consecutive_misses = 0
            else:
                consecutive_misses += 1
            
            i += 1
        
        console.print(f"  Found {len(self.samples)} samples in {data_dir}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        
        # Load separate visual and IMU features
        visual_features = np.load(os.path.join(self.data_dir, f"{sample_id}_visual.npy"))  # [10, 512]
        imu_features = np.load(os.path.join(self.data_dir, f"{sample_id}_imu.npy"))        # [10, 256]
        visual_features = torch.from_numpy(visual_features).float()
        imu_features = torch.from_numpy(imu_features).float()
        
        # Load ground truth poses
        poses = np.load(os.path.join(self.data_dir, f"{sample_id}_gt.npy"))
        poses = torch.from_numpy(poses).float()
        
        return visual_features, imu_features, poses


def train_with_separate_features(
    data_dir: str = "aria_latent_data_pretrained",
    learning_rate: float = 5e-4,
    batch_size: int = 32,
    max_epochs: int = 100,
    use_wandb: bool = True
):
    """Train with separate visual and IMU features"""
    
    console.rule(f"[bold cyan]🚀 Training with Separate Visual & IMU Features[/bold cyan]")
    console.print(f"\nConfiguration:")
    console.print(f"  Visual features: 512-dimensional")
    console.print(f"  IMU features: 256-dimensional (actual encoded features)")
    console.print(f"  Sequence length: 10 transitions")
    console.print(f"  Learning rate: {learning_rate}")
    console.print(f"  Batch size: {batch_size}")
    
    # Set seed
    L.seed_everything(42, workers=True)
    
    # Create model
    model = MultiHeadVIOModelSeparate(
        visual_dim=512,
        imu_dim=256,
        hidden_dim=256,
        num_shared_layers=4,
        num_specialized_layers=3,
        num_heads=8,
        dropout=0.1,
        learning_rate=learning_rate,
        weight_decay=1e-5,
        sequence_length=10
    )
    
    # Create datasets
    console.print("\n[bold]Loading separate visual and IMU features...[/bold]")
    train_dataset = SeparateFeatureDataset(f"{data_dir}/train")
    val_dataset = SeparateFeatureDataset(f"{data_dir}/val")
    
    # Create dataloaders
    def collate_fn(batch):
        visual_features, imu_features, poses = zip(*batch)
        return {
            'visual_features': torch.stack(visual_features),
            'imu_features': torch.stack(imu_features),
            'poses': torch.stack(poses)
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn
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
    
    # Verify data
    console.print("\n[bold]Verifying feature shapes...[/bold]")
    sample_batch = next(iter(train_loader))
    console.print(f"  Visual features: {sample_batch['visual_features'].shape}")
    console.print(f"  IMU features: {sample_batch['imu_features'].shape}")
    console.print(f"  Poses: {sample_batch['poses'].shape}")
    
    # Setup loggers
    loggers = []
    
    tb_logger = TensorBoardLogger(
        save_dir="logs",
        name="separate_features",
        version=f"lr_{learning_rate}"
    )
    loggers.append(tb_logger)
    
    if use_wandb:
        wandb_logger = WandbLogger(
            project="vift-aea-separate",
            name=f"separate_features_lr_{learning_rate}",
            tags=["separate-features", "pretrained", f"lr_{learning_rate}"]
        )
        loggers.append(wandb_logger)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val/total_loss",
            mode="min",
            save_top_k=3,
            dirpath=f"logs/checkpoints_separate",
            filename="epoch_{epoch:03d}_{val_total_loss:.4f}",
            save_last=True,
            verbose=True
        ),
        EarlyStopping(
            monitor="val/total_loss",
            patience=10,
            mode="min",
            verbose=True
        ),
        LearningRateMonitor(logging_interval='step'),
        RichProgressBar(refresh_rate=10)
    ]
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=20,
        val_check_interval=0.5
    )
    
    # Train
    console.rule("[bold cyan]🏃 Starting Training[/bold cyan]")
    console.print("\n[bold yellow]Key features:[/bold yellow]")
    console.print("  • Separate visual (512D) and IMU (256D) processing")
    console.print("  • Multi-modal transformer fusion")
    console.print("  • Specialized rotation and translation heads")
    console.print("  • 10 transition predictions from 11 frames")
    console.print()
    
    try:
        trainer.fit(model, train_loader, val_loader)
        
        console.rule("[bold green]✅ Training Completed![/bold green]")
        
        if trainer.checkpoint_callback:
            best_path = trainer.checkpoint_callback.best_model_path
            best_loss = trainer.checkpoint_callback.best_model_score
            
            console.print(f"\n[bold]Best checkpoint:[/bold] {best_path}")
            console.print(f"[bold]Best validation loss:[/bold] {best_loss:.6f}")
            
            # Show model size
            total_params = sum(p.numel() for p in model.parameters())
            console.print(f"\n[bold]Model size:[/bold] {total_params/1e6:.1f}M parameters")
            
    except Exception as e:
        console.print(f"\n[bold red]Training failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train with separate visual and IMU features')
    parser.add_argument('--data_dir', type=str, default='aria_latent_data_pretrained',
                       help='Data directory')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Max epochs')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable W&B')
    
    args = parser.parse_args()
    
    console.print("[bold magenta]Training with Separate Visual & IMU Features[/bold magenta]")
    console.print("Using actual 256D IMU features from pretrained encoder\n")
    
    train_with_separate_features(
        data_dir=args.data_dir,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        use_wandb=not args.no_wandb
    )