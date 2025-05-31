#!/usr/bin/env python3
"""
Train just the multi-head model.
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
        sequence_length=11,
        feature_dim=768,
        hidden_dim=256,
        num_transformer_layers=4,
        num_attention_heads=8,
        head_layers=3,
        dropout=0.1,
        use_auxiliary_tasks=True,
        lr=1e-4,
        weight_decay=1e-5,
        rotation_weight=1.0,
        translation_weight=1.0,
        velocity_weight=0.3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create data module
    datamodule = SimpleAriaDataModule(
        train_data_dir="aria_latent_data/train",
        val_data_dir="aria_latent_data/val",
        test_data_dir="aria_latent_data/test",
        batch_size=16,
        num_workers=4,
        max_train_samples=1000,
        max_val_samples=100
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
            patience=10,
            mode="min",
            verbose=True
        ),
        TrajectoryValidationCallback(
            log_every_n_epochs=5  # Compute trajectory metrics every 5 epochs
        )
    ]
    
    # Create trainer (single GPU)
    trainer = L.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        logger=logger,
        callbacks=callbacks,
        deterministic=True
    )
    
    print("🏋️ Starting training with trajectory-aware validation...")
    trainer.fit(model, datamodule)
    
    print(f"✅ Training completed with trajectory metrics!")
    print(f"Best model: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best loss: {trainer.checkpoint_callback.best_model_score:.6f}")
    print(f"🎯 For full trajectory evaluation, run: python evaluate_trajectory_based.py")
    
    return trainer.checkpoint_callback.best_model_path


if __name__ == "__main__":
    train_multihead_model()