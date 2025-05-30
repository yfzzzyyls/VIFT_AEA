#!/usr/bin/env python3
"""
Simplified training script for AR/VR optimized VIO models.
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
from src.models.multiscale_vio import MultiScaleTemporalVIO
from src.models.multihead_vio import MultiHeadVIOModel


def train_multiscale_model():
    """Train the multi-scale temporal VIO model."""
    print("🚀 Training Multi-Scale AR/VR VIO Model")
    
    # Set seed
    L.seed_everything(42, workers=True)
    
    # Create model
    model = MultiScaleTemporalVIO(
        sequence_lengths=[7, 11, 15],
        feature_dim=768,
        hidden_dim=256,
        num_transformer_layers=4,
        num_attention_heads=8,
        dropout=0.1,
        use_scale_weights=True,
        lr=1e-4,
        weight_decay=1e-5
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create data module
    datamodule = SimpleAriaDataModule(
        batch_size=16,
        num_workers=4,
        max_train_samples=1000,
        max_val_samples=100
    )
    
    # Setup logger
    logger = TensorBoardLogger("logs", name="arvr_multiscale_vio")
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val/total_loss",
            mode="min",
            save_top_k=3,
            filename="multiscale_{epoch:02d}_{val_total_loss:.4f}",
            save_last=True
        ),
        EarlyStopping(
            monitor="val/total_loss",
            patience=10,
            mode="min",
            verbose=True
        )
    ]
    
    # Create trainer
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
    
    print("🏋️ Starting training...")
    trainer.fit(model, datamodule)
    
    print(f"✅ Training completed!")
    print(f"Best model: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best loss: {trainer.checkpoint_callback.best_model_score:.6f}")
    
    return trainer.checkpoint_callback.best_model_path


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
        )
    ]
    
    # Create trainer
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
    
    print("🏋️ Starting training...")
    trainer.fit(model, datamodule)
    
    print(f"✅ Training completed!")
    print(f"Best model: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best loss: {trainer.checkpoint_callback.best_model_score:.6f}")
    
    return trainer.checkpoint_callback.best_model_path


def main():
    """Main training function."""
    print("🎯 AR/VR VIO Training Suite")
    print("="*50)
    
    # Train multi-scale model
    print("\n1️⃣ Training Multi-Scale Model...")
    multiscale_path = train_multiscale_model()
    
    print("\n" + "="*50)
    
    # Train multi-head model
    print("\n2️⃣ Training Multi-Head Model...")
    multihead_path = train_multihead_model()
    
    print("\n" + "="*50)
    print("🎉 ALL TRAINING COMPLETED!")
    print("="*50)
    print(f"Multi-Scale Model: {multiscale_path}")
    print(f"Multi-Head Model: {multihead_path}")
    print("\n📋 Next Steps:")
    print("1. Evaluate models with: python evaluation_auto.py --checkpoint <path>")
    print("2. Compare performance with baseline VIFT")
    print("3. Analyze AR/VR specific improvements")


if __name__ == "__main__":
    main()