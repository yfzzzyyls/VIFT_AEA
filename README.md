<div align="center">

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

</div>

This is an adaptation of the VIFT implementation for the **AriaEveryday dataset**, optimized for training and inference on NVIDIA H100 GPUs and Apple Silicon.

> **Original Paper**: Causal Transformer for Fusion and Pose Estimation in Deep Visual Inertial Odometry
> Yunus Bilge Kurt, Ahmet Akman, Aydın Alatan
> *ECCV 2024 VCAD Workshop*
> [[Paper](https://arxiv.org/abs/2409.08769)] [[Original Repo](https://github.com/ybkurt/VIFT)]

## Overview

VIFT-AEA is a Visual-Inertial Odometry (VIO) system that leverages transformer architectures for robust pose estimation. This implementation extends the original VIFT framework to support the AriaEveryday dataset, enabling training and inference on real-world AR/VR data captured with Meta's Aria glasses.

## Features

- **Cross-Platform Support**: Seamless operation on both CUDA-enabled Linux systems and Apple Silicon (M1/M2) macOS
- **AriaEveryday Dataset Integration**: Native support for processing and training on Meta's AriaEveryday dataset
- **Transformer-Based Architecture**: State-of-the-art visual-inertial fusion using attention mechanisms
- **Automated Environment Setup**: Intelligent platform detection and dependency installation

## System Requirements

### Supported Platforms

- **Linux**: CUDA 11.8+ compatible GPUs (RTX 20/30/40 series, Tesla, etc.)
- **macOS**: Apple Silicon (M1/M2/M3) with Metal Performance Shaders (MPS)
- **Fallback**: CPU-only execution on any platform

### Dependencies

- Python 3.8+
- PyTorch 2.3.0+
- OpenCV 4.8.0+
- NumPy 2.0.0+

## Quick Start

### 1. Environment Setup

Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd VIFT_AEA
```

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
```

Run the automated environment setup script:

```bash
python scripts/setup_env.py
```

This script will:

- Automatically detect your platform (CUDA/Apple Silicon/CPU)
- Install the appropriate PyTorch version with hardware acceleration
- Install all required dependencies from `requirements.txt`
- Verify the installation

### 2. Dataset Preparation

#### KITTI Dataset (Optional: for baseline comparison)

```bash
cd data
chmod +x data_prep.sh
./data_prep.sh
```

#### AriaEveryday Dataset

Place your AriaEveryday dataset in the `data/aria_everyday/` directory, then process it:

### Process it with sub-set of sequences

```
python scripts/process_aria_to_vift.py \
  --input-dir data/aria_everyday \
  --output-dir data/aria_real_train \
  --start-index 0 \
  --max-sequences 10
```

```
python scripts/process_aria_to_vift.py \
  --input-dir data/aria_everyday \
  --output-dir data/aria_real_test \
  --start-index 10 \
  --max-sequences 5
```

### Latent Caching

**Trainig dataset**

```
python data/latent_caching_aria.py --data_dir data/aria_real_train --save_dir aria_latent_data/train_13 --mode train --device mps
```

**Validation dataset**

```
python data/latent_caching_aria.py --data_dir data/aria_real_test --save_dir aria_latent_data/val_3 --mode val --val_sequences"20,22,24" --device mps
```

**Test/Inference dataset**

```
python data/latent_caching_aria.py --data_dir data/aria_real_test --save_dir aria_latent_data/test_3 --mode test
```

### 3. Training

Before training, create the project root marker file (required by rootutils):

```bash
touch .project-root
```

Train the model on your dataset:

```bash
# Train on AriaEveryday
python src/train.py data=aria_latent model=aria_vio
```

### 4. Evaluation

Evaluate the trained model:

```bash
python scripts/detailed_evaluation.py --checkpoint logs/train/runs/2025-05-27_11-26-57/checkpoints/epoch_000.ckpt --test_data aria_latent_data/test_3 --device mps
```

## Platform-Specific Features

### CUDA (Linux)

- Utilizes CUDA 11.8 for GPU acceleration
- Automatic mixed precision training
- Multi-GPU support for large-scale training

### Apple Silicon (macOS)

- Leverages Metal Performance Shaders (MPS) backend
- Optimized for M1/M2/M3 processors
- Native ARM64 binaries for maximum performance

### CPU Fallback

- Full functionality on any x86_64 or ARM64 system
- Automatic fallback when GPU acceleration is unavailable

## Project Structure

```
VIFT_AEA/
├── src/                    # Source code
│   ├── models/            # Model architectures
│   ├── data/              # Dataset loaders
│   └── train.py           # Training script
├── scripts/               # Utility scripts
│   ├── setup_env.py       # Environment setup
│   └── process_aria_to_vift.py  # AriaEveryday processing
├── data/                  # Datasets
├── configs/               # Hydra configurations
└── requirements.txt       # Dependencies
```

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/data/kitti_vio.yaml` - KITTI dataset configuration
- `configs/data/aria_vio.yaml` - AriaEveryday dataset configuration
- `configs/model/vift.yaml` - Model architecture configuration
- `configs/trainer/default.yaml` - Training configuration

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed by re-running `python scripts/setup_env.py`
2. **CUDA out of memory**: Reduce batch size in the configuration files
3. **Apple Silicon performance**: Ensure you're using the MPS backend by checking `torch.backends.mps.is_available()`

### Platform Detection

The setup script automatically detects your platform. You can verify detection by running:

```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')
"
```

# VIFT Aria Integration

This repository extends the Visual-Inertial Fused Transformer (VIFT) to work with Meta's AriaEveryday dataset, enabling visual-inertial odometry training on real-world egocentric data.

## Overview

VIFT-AEA (VIFT + AriaEveryday) adapts the original VIFT architecture to process Aria's RGB camera streams and IMU data, following the exact same pipeline as KITTI to ensure compatibility with the pretrained models.

## Setup

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd VIFT_AEA

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pretrained Models

```bash
# Download the pretrained VSVIO encoder (same as used for KITTI)
mkdir -p pretrained_models
# Place vf_512_if_256_3e-05.model in pretrained_models/
```

### 3. Prepare Aria Data

Ensure your processed Aria data is in the following structure:

```
data/aria_real_train/
├── 00/
│   ├── visual_data.pt      # [T, 3, H, W] RGB frames
│   ├── imu_data.pt         # [T, 33, 6] IMU measurements  
│   └── poses.json          # Ground truth poses
├── 01/
└── ...

data/aria_real_test/
├── 08/
├── 09/
└── ...
```

## Training Pipeline

### Step 1: Latent Caching (Required)

Extract visual-inertial features using a ResNet-based encoder compatible with VIFT:

**Training Data:**

```bash
python data/latent_caching_aria.py \
    --data_dir data/aria_real_train \
    --save_dir aria_latent_data/train_10 \
    --mode train \
    --device mps
```

**Validation Data:**

```bash
python data/latent_caching_aria.py \
    --data_dir data/aria_real_train \
    --save_dir aria_latent_data/val_10 \
    --mode val \
    --val_sequences "8,9" \
    --device mps
```

This step:

- Uses ResNet18 for visual feature extraction (512 dims)
- Processes RGB images (256×512) and IMU data (6-DOF, padded to 256 dims)
- Outputs 768-dim features: Visual(512) + IMU(256)
- Saves in KITTI-compatible `.npy` format
- Single script handles both training and validation data

### Step 2: Training

**Option A: Using Cached Latent Features (Recommended)**

```bash
python src/train.py data=aria_latent model=aria_vio
```

**Option B: Using Raw Data (Slower)**

```bash
python src/train.py --config-name=train_aria
```

### Step 3: Evaluation

```bash
python src/eval.py --config-name=eval_aria ckpt_path=path/to/checkpoint.ckpt
```

## Data Format Compatibility

### Aria → KITTI Pipeline Alignment

| Component          | KITTI Format           | Aria Format                  | Processing                 |
| ------------------ | ---------------------- | ---------------------------- | -------------------------- |
| **Visual**   | RGB stereo images      | RGB camera frames            | Same transforms (256×512) |
| **IMU**      | 6-DOF IMU              | 6-DOF IMU (33 samples/frame) | Average to 6 values/frame  |
| **Poses**    | [tx,ty,tz,qx,qy,qz,qw] | [tx,ty,tz,qx,qy,qz,qw]       | Same format                |
| **Features** | [seq_len, 768]         | [seq_len, 768]               | Identical (512+256)        |

### Key Features

✅ **ResNet-based Encoder**: Uses ResNet18 for reliable visual feature extraction
✅ **Same Dimensions**: 768-dim features (Visual: 512, IMU: 256)
✅ **Same Transforms**: Image preprocessing (256×512, normalization)
✅ **Same Format**: Output `.npy` files compatible with VIFT
✅ **Unified Script**: Single script handles both training and validation data
✅ **No Dependencies**: Works without VIFT pretrained models

## Configuration Files

### Data Configurations

- `configs/data/aria_vio.yaml` - Raw Aria data loading
- `configs/data/aria_latent.yaml` - Cached latent features

### Model Configurations

- `configs/model/aria_vio.yaml` - Aria-compatible VIO model
- `configs/trainer/default.yaml` - Training parameters

### Training Configurations

- `configs/train_aria.yaml` - Main training config

## File Structure

```
VIFT_AEA/
├── src/
│   ├── data/
│   │   └── aria_datamodule.py          # Aria data loading
│   └── models/
│       └── aria_vio_module.py          # Aria-compatible Lightning module
├── data/
│   └── latent_caching_aria.py          # Unified train/val latent caching
├── configs/
│   ├── data/
│   │   ├── aria_latent.yaml            # Latent data config
│   │   └── aria_vio.yaml               # Raw data config
│   ├── model/
│   │   └── aria_vio.yaml               # Model config
│   └── train_aria.yaml                 # Training config
└── README.md
```

## Performance Notes

### Training Speed

- **With Latent Caching**: ~5-10x faster training
- **Without Caching**: Real-time visual encoding (slower)

### Hardware Requirements

- **GPU**: Recommended for latent caching (MPS/CUDA)
- **RAM**: 16GB+ for large sequences
- **Storage**: ~1GB per 1000 cached samples

## Troubleshooting

### Common Issues

**1. Pretrained Model Missing**

```bash
# No longer needed - script uses ResNet18 encoder
# Works out of the box without external dependencies
```

**2. Data Format Issues**

```bash
# Check your data format
python -c "
import torch
from pathlib import Path
data = torch.load('data/aria_real_train/00/visual_data.pt')
print(f'Visual data shape: {data.shape}')
"
```

**3. Memory Issues**

```bash
# Reduce batch size in config
batch_size: 16  # Default: 32
```

**4. Device Issues**

```bash
# Force CPU if GPU issues
python data/latent_caching_aria.py --device cpu --mode train
```

**5. Validation Sequences**

```bash
# Customize validation sequences
python data/latent_caching_aria.py \
    --mode val --val_sequences "7,8,9" \
    --save_dir aria_latent_data/val_10
```

## Citation

If you use this work, please cite:

```bibtex
@article{vift_aria_2024,
  title={VIFT Aria Integration: Visual-Inertial Odometry with AriaEveryday Dataset},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Acknowledgments

- Original VIFT paper and implementation
- Meta's AriaEveryday dataset
- KITTI dataset for reference implementation

## License

This project follows the same license as the original VIFT implementation.
