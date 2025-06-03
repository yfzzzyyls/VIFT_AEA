#!/usr/bin/env python3
"""
Improved training script with fixes for loss underflow and overfitting
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

# Add src to path
sys.path.append('src')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_separate_features import SeparateFeatureDataset
from src.models.multihead_vio_separate import MultiHeadVIOModelSeparate

console = Console()


def main():
    console.rule("[bold cyan]🚀 Improved VIO Training[/bold cyan]")
    
    # Configuration
    data_dir = "aria_latent_data_pretrained"
    learning_rate = 1e-3
    batch_size = 32
    max_epochs = 100
    
    console.print("\n[bold]Key Improvements:[/bold]")
    console.print("  ✅ Fixed loss function with SmoothL1 + log scale")
    console.print("  ✅ Removed excessive small motion weighting")
    console.print("  ✅ Added regularization to prevent trivial solutions")
    console.print("  ✅ Reduced model capacity (2.8M params)")
    console.print("  ✅ Added data augmentation")
    console.print("  ✅ Better learning rate schedule (OneCycleLR)")
    console.print("  ✅ Improved logging and monitoring")
    
    # Set seed
    L.seed_everything(42, workers=True)
    
    # Create improved model
    model = MultiHeadVIOModelSeparate(
        visual_dim=512,
        imu_dim=256,
        hidden_dim=128,  # Reduced
        num_shared_layers=2,  # Reduced
        num_specialized_layers=2,  # Reduced
        num_heads=4,  # Reduced
        dropout=0.2,  # Increased
        learning_rate=learning_rate,
        weight_decay=1e-4,  # Increased
        sequence_length=10,
        rotation_weight=1.0,  # Balanced weights
        translation_weight=1.0
    )
    
    # Create datasets with augmentation
    console.print("\n[bold]Loading data with augmentation...[/bold]")
    train_dataset = SeparateFeatureDataset(f"{data_dir}/train", augment=True)
    val_dataset = SeparateFeatureDataset(f"{data_dir}/val", augment=False)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = torch.utils.data.DataLoader(
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
    
    # Logger
    logger = TensorBoardLogger(
        save_dir="logs",
        name="improved_training",
        version=f"lr_{learning_rate}_fixed_loss"
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val/total_loss",
            mode="min",
            save_top_k=3,
            dirpath="logs/checkpoints_improved",
            filename="epoch_{epoch:03d}_val_loss_{val_total_loss:.6f}",
            save_last=True,
            verbose=True
        ),
        EarlyStopping(
            monitor="val/total_loss",
            patience=25,
            mode="min",
            verbose=True,
            min_delta=0.00001
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
        accumulate_grad_batches=2,  # Effective batch size of 64
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=20,
        val_check_interval=0.5,
        deterministic=True
    )
    
    # Train
    console.rule("[bold cyan]🏃 Starting Improved Training[/bold cyan]")
    console.print("\n[bold yellow]Expected behavior:[/bold yellow]")
    console.print("  • Loss should start around 1-10 range (not 0.0000)")
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
        console.print("   tensorboard --logdir logs/improved_training")
        console.print("\n2. Run inference on test set:")
        console.print(f"   python inference_full_sequence.py --sequence-id all \\")
        console.print(f"     --checkpoint {best_path}")


def collate_fn(batch):
    """Collate function for DataLoader"""
    visual_features, imu_features, poses = zip(*batch)
    return {
        'visual_features': torch.stack(visual_features),
        'imu_features': torch.stack(imu_features),
        'poses': torch.stack(poses)
    }


if __name__ == "__main__":
    main()