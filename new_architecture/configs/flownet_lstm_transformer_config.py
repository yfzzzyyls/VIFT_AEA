"""
Configuration for FlowNet-LSTM-Transformer Architecture
"""

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Visual encoder (FlowNet)
    visual_feature_dim: int = 256
    
    # IMU encoder (LSTM)
    imu_hidden_dim: int = 128
    imu_lstm_layers: int = 3
    imu_feature_dim: int = 256
    imu_bidirectional: bool = True
    
    # Transformer
    transformer_dim: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 6
    transformer_feedforward: int = 2048
    
    # General
    dropout: float = 0.1


@dataclass
class DataConfig:
    """Dataset configuration."""
    data_dir: str = "aria_processed"
    image_size: tuple = (704, 704)  # Default: 2x2 binned from Aria's 1408x1408
    
    # Sequence settings
    variable_length: bool = True
    sequence_length: int = 11  # Fixed length if variable_length=False
    min_seq_len: int = 10  # Start with 10 frames
    max_seq_len: int = 50
    stride: int = 5
    
    
    # Data loading
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    scheduler: str = "cosine"  # "cosine", "step", "exponential"
    warmup_epochs: int = 5
    
    # Training duration
    num_epochs: int = 100
    
    # Loss weights
    translation_weight: float = 1.0
    rotation_weight: float = 10.0
    scale_weight: float = 20.0
    
    # Checkpointing
    checkpoint_dir: str = "new_architecture/checkpoints"
    save_every: int = 5
    validate_every: int = 1
    
    # Logging
    log_every: int = 100
    use_wandb: bool = False
    project_name: str = "flownet-lstm-transformer"
    
    # Distributed training
    distributed: bool = False
    backend: str = "nccl"
    
    # Mixed precision
    use_amp: bool = True
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_start_len: int = 10  # Start with 10 frames
    curriculum_step: int = 10  # Increase sequence length every 10 epochs
    curriculum_increment: int = 5  # How many frames to add each step


@dataclass
class Config:
    """Complete configuration."""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    
    # Experiment settings
    experiment_name: str = "flownet_lstm_transformer_baseline"
    seed: int = 42
    device: str = "cuda"
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Create config from command line arguments."""
        model_config = ModelConfig(
            visual_feature_dim=args.visual_feature_dim,
            imu_hidden_dim=args.imu_hidden_dim,
            imu_lstm_layers=args.imu_lstm_layers,
            imu_feature_dim=args.imu_feature_dim,
            imu_bidirectional=args.imu_bidirectional,
            transformer_dim=args.transformer_dim,
            transformer_heads=args.transformer_heads,
            transformer_layers=args.transformer_layers,
            transformer_feedforward=args.transformer_feedforward,
            dropout=args.dropout
        )
        
        data_config = DataConfig(
            data_dir=args.data_dir,
            image_size=(args.image_height, args.image_width),
            variable_length=args.variable_length,
            sequence_length=args.sequence_length,
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len,
            stride=args.stride,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        training_config = TrainingConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            scheduler=args.scheduler,
            warmup_epochs=args.warmup_epochs,
            num_epochs=args.num_epochs,
            translation_weight=args.translation_weight,
            rotation_weight=args.rotation_weight,
            scale_weight=args.scale_weight,
            checkpoint_dir=args.checkpoint_dir,
            save_every=args.save_every,
            validate_every=args.validate_every,
            log_every=args.log_every,
            use_wandb=args.use_wandb,
            distributed=args.distributed,
            use_amp=args.use_amp,
            use_curriculum=args.use_curriculum,
            curriculum_step=args.curriculum_step if hasattr(args, 'curriculum_step') else 10,
            curriculum_increment=args.curriculum_increment if hasattr(args, 'curriculum_increment') else 5
        )
        
        return cls(
            model=model_config,
            data=data_config,
            training=training_config,
            experiment_name=args.experiment_name,
            seed=args.seed
        )


def get_parser() -> argparse.ArgumentParser:
    """Get command line argument parser."""
    parser = argparse.ArgumentParser(
        description="FlowNet-LSTM-Transformer for Visual-Inertial Odometry"
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--visual-feature-dim", type=int, default=256,
                            help="Visual encoder output dimension")
    model_group.add_argument("--imu-hidden-dim", type=int, default=128,
                            help="IMU LSTM hidden dimension")
    model_group.add_argument("--imu-lstm-layers", type=int, default=3,
                            help="Number of LSTM layers")
    model_group.add_argument("--imu-feature-dim", type=int, default=256,
                            help="IMU encoder output dimension")
    model_group.add_argument("--imu-bidirectional", action="store_true",
                            help="Use bidirectional LSTM")
    model_group.add_argument("--transformer-dim", type=int, default=512,
                            help="Transformer hidden dimension")
    model_group.add_argument("--transformer-heads", type=int, default=8,
                            help="Number of attention heads")
    model_group.add_argument("--transformer-layers", type=int, default=6,
                            help="Number of transformer layers")
    model_group.add_argument("--transformer-feedforward", type=int, default=2048,
                            help="Transformer feedforward dimension")
    model_group.add_argument("--dropout", type=float, default=0.1,
                            help="Dropout rate")
    
    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data-dir", type=str, default="aria_processed",
                           help="Path to Aria dataset")
    data_group.add_argument("--image-height", type=int, default=704,
                           help="Image height")
    data_group.add_argument("--image-width", type=int, default=704,
                           help="Image width")
    data_group.add_argument("--variable-length", action="store_true",
                           help="Use variable sequence lengths")
    data_group.add_argument("--sequence-length", type=int, default=11,
                           help="Fixed sequence length")
    data_group.add_argument("--min-seq-len", type=int, default=10,
                           help="Minimum sequence length")
    data_group.add_argument("--max-seq-len", type=int, default=50,
                           help="Maximum sequence length")
    data_group.add_argument("--stride", type=int, default=5,
                           help="Stride for sliding window")
    data_group.add_argument("--batch-size", type=int, default=8,
                           help="Batch size per GPU")
    data_group.add_argument("--num-workers", type=int, default=4,
                           help="Number of data loading workers")
    
    # Training arguments
    training_group = parser.add_argument_group("Training")
    training_group.add_argument("--learning-rate", type=float, default=1e-4,
                               help="Learning rate")
    training_group.add_argument("--weight-decay", type=float, default=1e-4,
                               help="Weight decay")
    training_group.add_argument("--max-grad-norm", type=float, default=1.0,
                               help="Maximum gradient norm")
    training_group.add_argument("--scheduler", type=str, default="cosine",
                               choices=["cosine", "step", "exponential"],
                               help="Learning rate scheduler")
    training_group.add_argument("--warmup-epochs", type=int, default=5,
                               help="Number of warmup epochs")
    training_group.add_argument("--num-epochs", type=int, default=100,
                               help="Number of training epochs")
    training_group.add_argument("--translation-weight", type=float, default=1.0,
                               help="Translation loss weight")
    training_group.add_argument("--rotation-weight", type=float, default=10.0,
                               help="Rotation loss weight")
    training_group.add_argument("--scale-weight", type=float, default=20.0,
                               help="Scale consistency loss weight")
    training_group.add_argument("--checkpoint-dir", type=str, 
                               default="new_architecture/checkpoints",
                               help="Checkpoint directory")
    training_group.add_argument("--save-every", type=int, default=5,
                               help="Save checkpoint every N epochs")
    training_group.add_argument("--validate-every", type=int, default=1,
                               help="Validate every N epochs")
    training_group.add_argument("--log-every", type=int, default=100,
                               help="Log every N iterations")
    training_group.add_argument("--use-wandb", action="store_true",
                               help="Use Weights & Biases logging")
    training_group.add_argument("--distributed", action="store_true",
                               help="Use distributed training")
    training_group.add_argument("--use-amp", action="store_true",
                               help="Use automatic mixed precision")
    training_group.add_argument("--use-curriculum", action="store_true",
                               help="Use curriculum learning for sequence length")
    training_group.add_argument("--curriculum-step", type=int, default=10,
                               help="Increase sequence length every N epochs")
    training_group.add_argument("--curriculum-increment", type=int, default=5,
                               help="How many frames to add at each step")
    
    # Experiment arguments
    exp_group = parser.add_argument_group("Experiment")
    exp_group.add_argument("--experiment-name", type=str, 
                          default="flownet_lstm_transformer_baseline",
                          help="Experiment name")
    exp_group.add_argument("--seed", type=int, default=42,
                          help="Random seed")
    
    # Distributed training arguments
    dist_group = parser.add_argument_group("Distributed")
    dist_group.add_argument("--local-rank", type=int, default=-1,
                           help="Local rank for distributed training")
    
    return parser


def get_config() -> Config:
    """Get configuration from command line arguments."""
    parser = get_parser()
    args = parser.parse_args()
    return Config.from_args(args)