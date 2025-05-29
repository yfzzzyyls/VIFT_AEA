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

#### Step 1: Process a Subset of AriaEveryday

```bash
# Process a subset (e.g., 40 sequences) from the full AriaEveryday dataset
python scripts/process_aria_to_vift.py \
  --input-dir data/aria_everyday \
  --output-dir data/aria_processed \
  --start-index 0 \
  --max-sequences 40
```

#### Step 2: Split the Processed Subset

**Option 1: Automatic Split (Recommended)**
```bash
# Split the 40 processed sequences into train/val/test (70/15/15)
# This will create separate folders for each split
python scripts/create_dataset_splits.py \
    --data_dir data/aria_processed \
    --output_dir data/aria_split \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15

# This creates:
# - data/aria_split/train/ (28 sequences)
# - data/aria_split/val/ (6 sequences)
# - data/aria_split/test/ (6 sequences)
```

**Option 2: Manual Split**
```bash
# Manually organize sequences into folders
mkdir -p data/aria_split/{train,val,test}
# Move sequences 00-27 to train/
# Move sequences 28-33 to val/
# Move sequences 34-39 to test/
```

#### Step 3: Generate Latent Features

```bash
# Cache ALL sequences in each split folder (no need to specify sequences)

# Training features
python data/latent_caching_aria.py \
    --data_dir data/aria_split/train \
    --save_dir aria_latent_data/train \
    --mode train \
    --device cuda  # or mps for Apple Silicon

# Validation features  
python data/latent_caching_aria.py \
    --data_dir data/aria_split/val \
    --save_dir aria_latent_data/val \
    --mode val \
    --device cuda

# Test features
python data/latent_caching_aria.py \
    --data_dir data/aria_split/test \
    --save_dir aria_latent_data/test \
    --mode test \
    --device cuda
```

### 4. Training

```bash
# Create project root marker
touch .project-root

# Train with custom paths
python src/train.py data=aria_custom model=aria_vio_simple \
    train_dir=aria_latent_data/train \
    val_dir=aria_latent_data/val \
    test_dir=aria_latent_data/test \
    data.batch_size=32 \
    trainer.max_epochs=50

# Or use experiment configuration
python src/train.py experiment=my_aria_experiment
```

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

1. **Raw Data** → `process_aria_to_vift.py` → **Processed Data** (visual_data.pt, imu_data.pt, poses.json)
2. **Processed Data** → `latent_caching_aria.py` → **Latent Features** (.npy files)
3. **Latent Features** → `train.py` → **Trained Model**

### Data Format

| Component | Format | Dimensions | Description |
|-----------|--------|------------|-------------|
| Visual | RGB frames | [T, 3, 256, 512] | Resized camera images |
| IMU | 6-DOF | [T, 6] | Averaged from 33 samples/frame |
| Poses | Translation + Quaternion | [T, 7] | [tx,ty,tz,qx,qy,qz,qw] |
| Latent Features | Visual + IMU | [T, 768] | 512 (visual) + 256 (IMU) |

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

## License

This project follows the same license as the original VIFT implementation.