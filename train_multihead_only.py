#!/usr/bin/env python3
"""
Train the multi-head model with all-frames prediction.
Modified to predict poses for all frames in the sequence, similar to original VIFT.
"""

import os
import sys
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Add src to path
sys.path.append('src')

from src.data.simple_aria_datamodule import SimpleAriaDataModule
from src.models.multihead_vio import MultiHeadVIOModel
from trajectory_validation_callback import TrajectoryValidationCallback


def train_multihead_model():
    """Train the multi-head VIO model."""
    print("🚀 Training Multi-Head AR/VR VIO Model")
    
    # Set seed
    L.seed_everything(42, workers=True)
    
    # Create model
    model = MultiHeadVIOModel(
        feature_dim=768,
        hidden_dim=256,
        num_shared_layers=4,
        num_specialized_layers=3,
        num_heads=8,
        dropout=0.1,
        learning_rate=1e-4,
        weight_decay=1e-5,
        rotation_weight=1.0,
        translation_weight=1.0,
        velocity_weight=0.3,
        sequence_length=11
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create data module - using ALL available data
    # NOTE: Current data has 50/25/25 split instead of 80/10/10
    # Train: 3920 samples, Val: 1960 samples, Test: 1960 samples
    datamodule = SimpleAriaDataModule(
        train_data_dir="aria_latent_data/train",
        val_data_dir="aria_latent_data/val",
        test_data_dir="aria_latent_data/test",
        batch_size=32,  # Increased batch size for 4 GPUs (8 per GPU)
        num_workers=8,   # More workers for faster data loading
        max_train_samples=None,  # Use all 3920 training samples
        max_val_samples=None     # Use all 1960 validation samples
    )
    
    # Setup logger
    logger = TensorBoardLogger("logs", name="arvr_multihead_vio")
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val/total_loss",
            mode="min",
            save_top_k=3,
            filename="multihead_{epoch:02d}_{val_total_loss:.4f}",
            save_last=True
        ),
        EarlyStopping(
            monitor="val/total_loss",
            patience=25,  # Increased patience for 50-epoch training
            mode="min",
            verbose=True,
            min_delta=1e-6  # Add minimum delta to avoid stopping on tiny improvements
        ),
        TrajectoryValidationCallback(
            log_every_n_epochs=5  # Compute trajectory metrics every 5 epochs
        )
    ]
    
    # Create trainer (auto-detect GPUs)
    trainer = L.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices="auto",  # Automatically detect and use all available GPUs
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",  # DDP with unused params
        precision="16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False  # Sync BN across GPUs
    )
    
    # Print GPU information
    num_gpus = torch.cuda.device_count()
    print(f"🖥️ Detected {num_gpus} GPU(s)")
    if num_gpus > 1:
        print(f"📊 Using DDP strategy across {num_gpus} GPUs")
    
    print("🏋️ Starting training with trajectory-aware validation...")
    trainer.fit(model, datamodule)
    
    print(f"✅ Training completed with trajectory metrics!")
    print(f"Best model: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best loss: {trainer.checkpoint_callback.best_model_score:.6f}")
    print(f"🎯 For full trajectory evaluation, run: python evaluate_trajectory_based.py")
    
    return trainer.checkpoint_callback.best_model_path


if __name__ == "__main__":
    train_multihead_model()