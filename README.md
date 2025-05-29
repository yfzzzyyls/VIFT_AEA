# VIFT-AEA: Visual-Inertial Feature Transformer for AriaEveryday

<p align="center">
  <a href="https://pytorch.org/get-started/locally/">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white">
  </a>
  <a href="https://pytorchlightning.ai/">
    <img alt="Lightning" src="https://img.shields.io/badge/Lightning-792ee5?logo=pytorchlightning&logoColor=white">
  </a>
  <a href="https://hydra.cc/">
    <img alt="Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd">
  </a>
</p>

An adaptation of VIFT (Visual-Inertial Fused Transformer) for Meta's AriaEveryday dataset, enabling visual-inertial odometry training on real-world egocentric AR/VR data.

> **Original Paper**: Causal Transformer for Fusion and Pose Estimation in Deep Visual Inertial Odometry  
> Yunus Bilge Kurt, Ahmet Akman, Aydın Alatan  
> *ECCV 2024 VCAD Workshop* [[Paper](https://arxiv.org/abs/2409.08769)] [[Original Repo](https://github.com/ybkurt/VIFT)]

## Key Features

✅ **Verified Working Pipeline** - Complete end-to-end training on AriaEveryday dataset  
✅ **Simplified Architecture** - Uses dummy metrics for easier training without KITTI dependencies  
✅ **Cross-Platform Support** - Seamless operation on CUDA and Apple Silicon (MPS)  
✅ **Flexible Data Splitting** - Automatic train/val/test split generation

## Quick Start

### 1. Environment Setup

```bash
# Clone and setup
git clone <repository-url>
cd VIFT_AEA

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Auto-install dependencies (detects CUDA/MPS/CPU)
python scripts/setup_env.py
```

### 2. Download Pretrained Models

```bash
mkdir -p pretrained_models
# Place vf_512_if_256_3e-05.model in pretrained_models/
```

### 3. Dataset Preparation

#### Step 1: Clean AriaEveryday Dataset (if needed)

```bash
# Check for corrupted sequences
python scripts/check_aria_data.py --data_dir data/aria_everyday

# Remove corrupted sequences (moves them to backup)
python scripts/check_aria_data.py --data_dir data/aria_everyday --remove
```

#### Step 2: Process a Subset of AriaEveryday

```bash
# Process a subset (e.g., 10 sequences) from the full dataset
python scripts/process_aria_to_vift.py \
  --input-dir data/aria_everyday \
  --output-dir data/aria_processed \
  --start-index 0 \
  --max-sequences 10
```

#### Step 3: Split the Processed Data

```bash
# Split into train/val/test (e.g., 5/3/2 for 10 sequences)
python scripts/create_dataset_splits.py \
    --data_dir data/aria_processed \
    --output_dir data/aria_split \
    --train_ratio 0.5 \
    --val_ratio 0.3 \
    --test_ratio 0.2

# This creates separate folders:
# - data/aria_split/train/ (5 sequences)
# - data/aria_split/val/ (3 sequences)
# - data/aria_split/test/ (2 sequences)
```

#### Step 4: Generate Latent Features

```bash
# Cache features for each split (processes ALL sequences in folder)

# Training features
python data/latent_caching_aria.py \
    --data_dir data/aria_split/train \
    --save_dir aria_latent_data/train \
    --mode train \
    --device mps  # or cuda for NVIDIA GPUs

# Validation features  
python data/latent_caching_aria.py \
    --data_dir data/aria_split/val \
    --save_dir aria_latent_data/val \
    --mode val \
    --device mps

# Test features
python data/latent_caching_aria.py \
    --data_dir data/aria_split/test \
    --save_dir aria_latent_data/test \
    --mode test \
    --device mps
```

### 4. Training

```bash
# Create project root marker
touch .project-root

# Train using the verified command (all on one line):
python src/train.py --config-name=train_aria data=aria_latent model=aria_vio_simple data.train_loader.root_dir=aria_latent_data/train data.val_loader.root_dir=aria_latent_data/val data.test_loader.root_dir=aria_latent_data/test data.batch_size=32 trainer.max_epochs=50 trainer.accelerator=mps

# For CUDA GPUs, replace trainer.accelerator=mps with trainer.accelerator=gpu
```

**Expected Training Output:**
- Model: PoseTransformer with ~512K parameters
- Training batches: ~19 (for 5 sequences with batch_size=32)
- Initial loss: ~100-200 (will decrease over epochs)
- Training speed: ~30-40 it/s on Apple Silicon

### 5. Evaluation

```bash
python scripts/evaluate_unbiased.py \
    --checkpoint logs/train/runs/<timestamp>/checkpoints/epoch_000.ckpt \
    --test_data aria_latent_data/test \
    --batch_size 32 \
    --device cuda
```

## Data Pipeline

### Processing Flow

```
AriaEveryday Dataset (98 sequences)
         ↓ (check_aria_data.py - remove corrupted)
Clean Dataset (92 sequences)  
         ↓ (process_aria_to_vift.py - select subset)
Processed Data (e.g., 10 sequences)
         ↓ (create_dataset_splits.py - split data)
Train/Val/Test Folders (5/3/2 sequences)
         ↓ (latent_caching_aria.py - extract features)
Latent Features (.npy files)
         ↓ (train.py - train model)
Trained Model (.ckpt)
```

### Data Format

| Component | Format | Dimensions | Description |
|-----------|--------|------------|-------------|
| Visual | RGB frames | [500, 3, 480, 640] | Original Aria camera frames |
| Visual (processed) | RGB frames | [500, 3, 256, 512] | Resized for model input |
| IMU | 6-DOF | [500, 6] | Averaged from 33 samples/frame |
| Poses | Translation + Quaternion | [500, 7] | [tx,ty,tz,qx,qy,qz,qw] |
| Latent Features | Visual + IMU | [seq_len, 768] | 512 (visual) + 256 (IMU) |

### Dataset Organization

```
data/
├── aria_everyday/          # Original dataset (92 valid sequences)
├── aria_processed/         # Processed subset (e.g., 10 sequences)
│   ├── 00/
│   │   ├── visual_data.pt  # [500, 3, 480, 640]
│   │   ├── imu_data.pt     # [500, 6]
│   │   └── poses.json      # Ground truth poses
│   └── ...
├── aria_split/            # Split into train/val/test
│   ├── train/            # 5 sequences
│   ├── val/              # 3 sequences
│   └── test/             # 2 sequences
└── aria_latent_data/      # Cached features
    ├── train/            # Training features
    ├── val/              # Validation features
    └── test/             # Test features
```

## Configuration

### Key Config Files

- **Data Configs**:
  - `configs/data/aria_custom.yaml` - Flexible paths for custom datasets
  - `configs/data/aria_latent.yaml` - Default Aria latent data
  
- **Model Configs**:
  - `configs/model/aria_vio_simple.yaml` - Simplified model (recommended)
  - `configs/model/aria_vio.yaml` - Full model with KITTI tester
  
- **Experiment Configs**:
  - `configs/experiment/my_aria_experiment.yaml` - Example custom setup

### Creating Custom Experiments

```yaml
# configs/experiment/my_custom_experiment.yaml
# @package _global_
defaults:
  - override /data: aria_custom
  - override /model: aria_vio_simple
  - override /trainer: default

# Dataset paths
train_dir: aria_latent_data/my_train
val_dir: aria_latent_data/my_val
test_dir: aria_latent_data/my_test

# Training parameters
trainer:
  max_epochs: 100
  check_val_every_n_epoch: 5

model:
  optimizer:
    lr: 0.0001
```

## Platform Support

- **CUDA (Linux)**: NVIDIA GPUs with CUDA 11.8+
- **Apple Silicon (macOS)**: M1/M2/M3 with MPS backend
- **CPU**: Fallback for any platform

Verify installation:
```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')
"
```

## Troubleshooting

### Common Issues and Solutions

1. **"Can't instantiate abstract class DummyTester"**
   - The DummyTester class needs a `save_results` method
   - This has been fixed in the current version

2. **"FileNotFoundError: No such file or directory"**
   - Ensure you've run all preprocessing steps in order
   - Check that your paths match exactly (no extra spaces or typos)

3. **Command line errors like "command not found"**
   - Use the training command as ONE line, or use backslashes (`\`) for line breaks
   - No spaces after backslashes if using multiple lines

4. **Corrupted video files**
   - Run `check_aria_data.py` to detect and remove corrupted sequences
   - 6 sequences in the original dataset are known to be corrupted

5. **Wrong data paths in training**
   - Use `data.train_loader.root_dir=path` syntax, not just `train_dir=path`
   - The paths must be specified at the loader level

## New Features in This Fork

- ✅ **check_aria_data.py** - Detect and remove corrupted sequences
- ✅ **create_dataset_splits.py** - Automatically split data into train/val/test folders
- ✅ **aria_vio_simple model** - Simplified model without KITTI dependencies
- ✅ **DummyTester/DummyMetricsCalculator** - Enable training without complex evaluation
- ✅ **Verified MPS support** - Tested on Apple Silicon Macs

## License

This project follows the same license as the original VIFT implementation.