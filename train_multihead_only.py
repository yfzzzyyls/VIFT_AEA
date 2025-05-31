#!/usr/bin/env python3
"""
Train the multi-head model with all-frames prediction.
Modified to predict poses for all frames in the sequence, similar to original VIFT.
Includes all visualization and monitoring tools from original VIFT.
"""

import os
import sys
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    RichModelSummary,
    RichProgressBar,
    LearningRateMonitor
)
from torchinfo import summary
from rich.console import Console
from rich.tree import Tree
from rich import print as rprint
import logging

# Add src to path
sys.path.append('src')

from src.data.simple_aria_datamodule import SimpleAriaDataModule
from src.models.multihead_vio import MultiHeadVIOModel
from trajectory_validation_callback import TrajectoryValidationCallback

# Setup console for rich output
console = Console()


def log_hyperparameters(model, datamodule, trainer, logger_list):
    """Log hyperparameters to all loggers, mimicking original VIFT."""
    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    hparams = {
        "model": model.__class__.__name__,
        "model/params/total": total_params,
        "model/params/trainable": trainable_params,
        "model/params/non_trainable": non_trainable_params,
        "data/batch_size": datamodule.batch_size,
        "data/num_workers": datamodule.num_workers,
        "trainer/max_epochs": trainer.max_epochs,
        "trainer/accumulate_grad_batches": trainer.accumulate_grad_batches,
        "trainer/gradient_clip_val": trainer.gradient_clip_val,
        "trainer/precision": trainer.precision,
        "model/learning_rate": model.hparams.learning_rate,
        "model/weight_decay": model.hparams.weight_decay,
        "model/feature_dim": model.hparams.feature_dim,
        "model/hidden_dim": model.hparams.hidden_dim,
        "model/num_shared_layers": model.hparams.num_shared_layers,
        "model/num_specialized_layers": model.hparams.num_specialized_layers,
        "model/num_heads": model.hparams.num_heads,
        "model/dropout": model.hparams.dropout,
        "model/rotation_weight": model.hparams.rotation_weight,
        "model/translation_weight": model.hparams.translation_weight,
        "model/velocity_weight": model.hparams.velocity_weight,
        "seed": 42,
        "task_name": "multihead_vio_all_frames"
    }
    
    # Log to each logger
    for logger in logger_list:
        logger.log_hyperparams(hparams)


def print_config_tree(config_dict, save_path="logs/config_tree.txt"):
    """Print configuration as a tree structure, similar to original VIFT."""
    tree = Tree("⚙️ Configuration")
    
    # Model config
    model_branch = tree.add("📦 Model")
    model_branch.add(f"Type: MultiHeadVIOModel")
    model_branch.add(f"Feature dim: {config_dict['feature_dim']}")
    model_branch.add(f"Hidden dim: {config_dict['hidden_dim']}")
    model_branch.add(f"Shared layers: {config_dict['num_shared_layers']}")
    model_branch.add(f"Specialized layers: {config_dict['num_specialized_layers']}")
    model_branch.add(f"Attention heads: {config_dict['num_heads']}")
    model_branch.add(f"Dropout: {config_dict['dropout']}")
    model_branch.add(f"Learning rate: {config_dict['learning_rate']}")
    
    # Data config
    data_branch = tree.add("📊 Data")
    data_branch.add(f"Batch size: {config_dict['batch_size']}")
    data_branch.add(f"Workers: {config_dict['num_workers']}")
    data_branch.add(f"Dataset: AriaEveryday Activities")
    
    # Training config
    train_branch = tree.add("🏃 Training")
    train_branch.add(f"Epochs: {config_dict['max_epochs']}")
    train_branch.add(f"GPUs: {config_dict['num_gpus']}")
    train_branch.add(f"Precision: 16-mixed")
    train_branch.add(f"Gradient clip: 1.0")
    train_branch.add(f"Accumulate batches: 2")
    
    # Print to console
    rprint(tree)
    
    # Save to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(str(tree))
    console.print(f"💾 Config tree saved to: {save_path}", style="green")


def train_multihead_model(use_wandb=True, use_tensorboard=True, project_name="vift-aea", 
                         experiment_name=None, tags=None, offline=False):
    """Train the multi-head VIO model with full logging and visualization."""
    console.rule("[bold cyan]🚀 Training Multi-Head AR/VR VIO Model[/bold cyan]")
    
    # Set seed
    L.seed_everything(42, workers=True)
    console.print("✅ Random seed set to 42", style="green")
    
    # Model configuration
    model_config = {
        'feature_dim': 768,
        'hidden_dim': 256,
        'num_shared_layers': 4,
        'num_specialized_layers': 3,
        'num_heads': 8,
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'rotation_weight': 1.0,
        'translation_weight': 1.0,
        'velocity_weight': 0.3,
        'sequence_length': 11
    }
    
    # Create model
    console.print("\n[bold]Creating Multi-Head VIO Model...[/bold]")
    model = MultiHeadVIOModel(**model_config)
    
    # Calculate and display parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    console.print(f"\n[bold cyan]Model Statistics:[/bold cyan]")
    console.print(f"  Total parameters: {total_params:,}")
    console.print(f"  Trainable parameters: {trainable_params:,}")
    console.print(f"  Non-trainable parameters: {non_trainable_params:,}")
    
    # Print model summary using torchinfo
    console.print("\n[bold]Model Architecture Summary:[/bold]")
    dummy_batch = {
        'images': torch.randn(1, 11, 768),
        'imus': torch.randn(1, 11, 6),
        'poses': torch.randn(1, 11, 7)
    }
    summary(model, input_data=[dummy_batch], verbose=1, depth=3, 
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20, row_settings=["var_names"])
    
    # Data configuration
    data_config = {
        'batch_size': 32,  # Increased batch size for 4 GPUs (8 per GPU)
        'num_workers': 8   # More workers for faster data loading
    }
    
    # Create data module - using ALL available data with proper 80/10/10 split
    console.print("\n[bold]Creating Data Module...[/bold]")
    datamodule = SimpleAriaDataModule(
        train_data_dir="aria_latent_data/train",
        val_data_dir="aria_latent_data/val",
        test_data_dir="aria_latent_data/test",
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        max_train_samples=None,  # Use all training samples
        max_val_samples=None     # Use all validation samples
    )
    
    # Setup datamodule to get dataset info
    datamodule.setup("fit")
    console.print(f"\n[bold cyan]Dataset Statistics:[/bold cyan]")
    console.print(f"  Training samples: {len(datamodule.train_dataset):,}")
    console.print(f"  Validation samples: {len(datamodule.val_dataset):,}")
    console.print(f"  Batch size: {data_config['batch_size']}")
    console.print(f"  Workers: {data_config['num_workers']}")
    
    # Setup loggers (matching original VIFT's approach)
    console.print("\n[bold]Setting up Loggers...[/bold]")
    loggers = []
    
    if use_tensorboard:
        tb_logger = TensorBoardLogger(
            save_dir="logs",
            name="multihead_vio",
            version=experiment_name,
            log_graph=True,  # Log model graph
            default_hp_metric=False
        )
        loggers.append(tb_logger)
        console.print("  ✅ TensorBoard logger enabled", style="green")
    
    if use_wandb:
        # Default tags if none provided
        if tags is None:
            tags = ["multihead", "all-frames", "aria-everyday", "vift-aea"]
        
        wandb_logger = WandbLogger(
            project=project_name,
            name=experiment_name or "multihead_all_frames",
            save_dir="logs",
            offline=offline,  # Support offline mode
            log_model=True,  # Log model checkpoints
            tags=tags,
            group="multihead_vio",
            job_type="train",
            config={
                **model_config,
                **data_config,
                "model_type": "MultiHeadVIO",
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "dataset": "AriaEveryday",
                "max_epochs": 50,
                "seed": 42
            }
        )
        loggers.append(wandb_logger)
        mode = "offline" if offline else "online"
        console.print(f"  ✅ Weights & Biases logger enabled ({mode} mode)", style="green")
    
    if not loggers:
        console.print("  ⚠️  No loggers enabled. Training without logging.", style="yellow")
    
    # Setup callbacks (matching original VIFT's approach)
    console.print("\n[bold]Setting up Callbacks...[/bold]")
    callbacks = [
        # Model checkpoint - save best and last
        ModelCheckpoint(
            monitor="val/total_loss",
            mode="min",
            save_top_k=3,
            dirpath="logs/checkpoints",
            filename="epoch_{epoch:03d}_{val_total_loss:.4f}",
            save_last=True,
            verbose=True,
            auto_insert_metric_name=False
        ),
        # Early stopping with patience
        EarlyStopping(
            monitor="val/total_loss",
            patience=25,  # Increased patience for 50-epoch training
            mode="min",
            verbose=True,
            min_delta=1e-6,  # Minimum change to qualify as improvement
            check_finite=True,
            strict=True
        ),
        # Learning rate monitor
        LearningRateMonitor(
            logging_interval='step',
            log_momentum=False
        ),
        # Rich progress bar for better visualization
        RichProgressBar(
            refresh_rate=10,
            leave=True
        ),
        # Rich model summary
        RichModelSummary(
            max_depth=3
        ),
        # Custom trajectory validation
        TrajectoryValidationCallback(
            log_every_n_epochs=5  # Compute trajectory metrics every 5 epochs
        )
    ]
    
    console.print("  ✅ ModelCheckpoint callback", style="green")
    console.print("  ✅ EarlyStopping callback (patience=25)", style="green")
    console.print("  ✅ LearningRateMonitor callback", style="green")
    console.print("  ✅ RichProgressBar callback", style="green")
    console.print("  ✅ RichModelSummary callback", style="green")
    console.print("  ✅ TrajectoryValidation callback", style="green")
    
    # Training configuration
    training_config = {
        'max_epochs': 50,
        'num_gpus': torch.cuda.device_count()
    }
    
    # Create trainer (matching original VIFT's configuration)
    console.print("\n[bold]Creating PyTorch Lightning Trainer...[/bold]")
    trainer = L.Trainer(
        max_epochs=training_config['max_epochs'],
        accelerator="gpu",
        devices="auto",  # Automatically detect and use all available GPUs
        strategy="ddp_find_unused_parameters_true" if training_config['num_gpus'] > 1 else "auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=2,
        logger=loggers if loggers else False,  # Use loggers or disable
        callbacks=callbacks,
        deterministic=True,
        benchmark=False,  # Disable for deterministic training
        sync_batchnorm=True if training_config['num_gpus'] > 1 else False,
        enable_model_summary=False,  # We use RichModelSummary callback instead
        enable_checkpointing=True,
        enable_progress_bar=True,
        log_every_n_steps=50,
        val_check_interval=1.0,  # Validate every epoch
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,  # Run 2 validation batches before training
        detect_anomaly=False,  # Can enable for debugging
        profiler=None  # Can add "simple" or "advanced" for profiling
    )
    
    # Print GPU information
    console.print(f"\n[bold cyan]Hardware Configuration:[/bold cyan]")
    console.print(f"  GPUs detected: {training_config['num_gpus']}")
    if training_config['num_gpus'] > 0:
        for i in range(training_config['num_gpus']):
            gpu_name = torch.cuda.get_device_name(i)
            console.print(f"  GPU {i}: {gpu_name}")
    if training_config['num_gpus'] > 1:
        console.print(f"  Strategy: DDP with unused parameters", style="yellow")
    
    # Print configuration tree
    config_summary = {
        **model_config,
        **data_config,
        **training_config
    }
    print_config_tree(config_summary)
    
    # Log hyperparameters to all loggers
    if loggers:
        console.print("\n[bold]Logging Hyperparameters...[/bold]")
        log_hyperparameters(model, datamodule, trainer, loggers)
        console.print("  ✅ Hyperparameters logged", style="green")
    
    # Start training
    console.rule("[bold cyan]🏃 Starting Training[/bold cyan]")
    console.print("\nTraining with trajectory-aware validation...\n")
    
    try:
        trainer.fit(model, datamodule)
    except KeyboardInterrupt:
        console.print("\n⚠️  Training interrupted by user", style="yellow")
    except Exception as e:
        console.print(f"\n❌ Training failed with error: {str(e)}", style="red")
        raise
    
    # Training completed
    console.rule("[bold green]✅ Training Completed[/bold green]")
    
    # Print final results
    if trainer.checkpoint_callback:
        console.print("\n[bold]Training Results:[/bold]")
        console.print(f"  Best model path: {trainer.checkpoint_callback.best_model_path}")
        console.print(f"  Best validation loss: {trainer.checkpoint_callback.best_model_score:.6f}")
        console.print(f"  Last epoch: {trainer.current_epoch}")
    
    # Print logger information
    if loggers:
        console.print("\n[bold]Logger Information:[/bold]")
        for logger in loggers:
            if isinstance(logger, TensorBoardLogger):
                console.print(f"  TensorBoard logs: {logger.log_dir}")
                console.print(f"    Run: tensorboard --logdir {logger.log_dir}")
            elif isinstance(logger, WandbLogger):
                console.print(f"  W&B run: {logger.experiment.url if hasattr(logger.experiment, 'url') else 'Check wandb.ai'}")
    
    # Final instructions
    console.print("\n[bold cyan]Next Steps:[/bold cyan]")
    console.print("  1. For trajectory evaluation: python evaluate_trajectory_based.py")
    console.print("  2. View TensorBoard: tensorboard --logdir logs")
    if use_wandb:
        console.print("  3. View W&B dashboard: https://wandb.ai")
    
    return trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Train Multi-Head VIO Model with Full Logging and Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with both TensorBoard and W&B (default)
  python train_multihead_only.py
  
  # Train with TensorBoard only (disable W&B)
  python train_multihead_only.py --no-wandb
  
  # Train with W&B only (disable TensorBoard)
  python train_multihead_only.py --no-tensorboard
  
  # Train with W&B in offline mode
  python train_multihead_only.py --offline
  
  # Train with custom experiment name and tags
  python train_multihead_only.py --name "experiment_v2" --tags "baseline" "full-data"
  
  # Train without any logging
  python train_multihead_only.py --no-wandb --no-tensorboard
        """
    )
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--no-tensorboard', action='store_true', help='Disable TensorBoard logging')
    parser.add_argument('--project', type=str, default='vift-aea', help='W&B project name')
    parser.add_argument('--name', type=str, default=None, help='Experiment name for loggers')
    parser.add_argument('--tags', type=str, nargs='+', help='Tags for W&B run')
    parser.add_argument('--offline', action='store_true', help='Run W&B in offline mode')
    
    args = parser.parse_args()
    
    # Print startup banner
    console = Console()
    console.rule("[bold magenta]VIFT-AEA Multi-Head VIO Training[/bold magenta]")
    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  TensorBoard: {'Disabled' if args.no_tensorboard else 'Enabled'}")
    console.print(f"  Weights & Biases: {'Disabled' if args.no_wandb else 'Enabled'}")
    if not args.no_wandb:
        console.print(f"    Project: {args.project}")
        console.print(f"    Mode: {'Offline' if args.offline else 'Online'}")
    console.print()
    
    # Start training
    train_multihead_model(
        use_wandb=not args.no_wandb,
        use_tensorboard=not args.no_tensorboard,
        project_name=args.project,
        experiment_name=args.name,
        tags=args.tags,
        offline=args.offline
    )