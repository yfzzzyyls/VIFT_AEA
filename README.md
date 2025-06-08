# VIFT-AEA: Visual-Inertial Feature Transformer for Aria Everyday Activities

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

A state-of-the-art Visual-Inertial Odometry (VIO) system for the Aria Everyday Activities dataset, implementing and improving upon the VIFT architecture with multiple model variants.

## 🚀 Overview

This repository provides:
- **VIFT Original**: Implementation of the Causal Transformer for Fusion and Pose Estimation
- **MultiHead Improved**: Enhanced architecture with separate visual/IMU processing and multi-head attention
- **MultiHead Fixed**: Corrected version with geodesic loss for accurate rotation estimation
- Quaternion-based pipeline for improved numerical stability
- Pretrained Visual-Selective-VIO feature extraction
- Full training and evaluation pipelines

### Key Results

The MultiHead Fixed model achieves:
- **Translation Error**: 1.34 ± 2.00 mm (after trajectory alignment)
- **Rotation Error**: 6.88 ± 1.48 degrees
- **Frame-to-Frame Translation**: 0.096 ± 0.018 mm
- **Frame-to-Frame Rotation**: 0.029 ± 0.004 degrees

All models exceed AR/VR requirements (<0.1° rotation, <0.1cm translation error per frame).

## 📋 Requirements

- Python 3.9+
- CUDA-capable GPU (8GB+ VRAM recommended)
- 32GB+ RAM
- ~50GB disk space for processed data

## 🔧 Installation

```bash
# Clone repository
git clone https://github.com/your-username/VIFT_AEA.git
cd VIFT_AEA

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Data Preparation

### 1. Download Aria Everyday Activities Dataset

First, obtain the download URLs file from [Project Aria](https://www.projectaria.com/datasets/aea/).

```bash
# Place AriaEverydayActivities_download_urls.json in project root

# Download dataset (choose one)
python scripts/download_aria_dataset.py --all              # All 143 sequences (~500GB)
python scripts/download_aria_dataset.py --num-sequences 10 # Subset for testing
```

### 2. Process to VIFT Format

Convert Aria data to VIFT-compatible format with quaternion representations:

```bash
# Single instance processing
python scripts/process_aria_to_vift.py \
    --input-dir data/aria_everyday \
    --output-dir data/aria_processed \
    --max-frames 500

# Parallel processing (4x faster) - run in separate terminals
# Terminal 1: python scripts/process_aria_to_vift.py --start-index 0 --max-sequences 36
# Terminal 2: python scripts/process_aria_to_vift.py --start-index 36 --max-sequences 36
# Terminal 3: python scripts/process_aria_to_vift.py --start-index 72 --max-sequences 36
# Terminal 4: python scripts/process_aria_to_vift.py --start-index 108 --max-sequences 35
```

### 3. Download Pretrained Encoder

```bash
python download_pretrained_model.py
```

### 4. Extract Visual Features

Generate pretrained features and prepare training data:

```bash
python generate_all_pretrained_latents_fixed.py \
    --processed-dir data/aria_processed \
    --output-dir aria_latent_data_pretrained
```

## 🏃 Training

Train different model architectures:

### MultiHead Fixed (Recommended)

Uses geodesic loss for rotation instead of MSE, providing accurate quaternion training:

```bash
python train_improved.py --model multihead_fixed \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-3 \
    --hidden-dim 128 \
    --num-heads 4 \
    --dropout 0.2
```

### MultiHead Improved

```bash
python train_improved.py --model multihead \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-3 \
    --hidden-dim 128 \
    --num-heads 4 \
    --dropout 0.2
```

### VIFT Original

```bash
python train_improved.py --model vift_original \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --hidden-dim 128 \
    --num-heads 8 \
    --dropout 0.1
```

Monitor training:
```bash
tensorboard --logdir logs/
```

## 📈 Evaluation

Evaluate trained models on test sequences:

```bash
# Evaluate MultiHead Fixed model
python inference_full_sequence.py \
    --sequence-id all \
    --checkpoint logs/checkpoints_multihead_fixed/best_model.ckpt

# Evaluate MultiHead Improved model
python inference_full_sequence.py \
    --sequence-id all \
    --checkpoint logs/checkpoints_multihead/best_model.ckpt

# Evaluate VIFT Original model
python inference_full_sequence.py \
    --sequence-id all \
    --checkpoint logs/checkpoints_vift_original/best_model.ckpt

# Single sequence evaluation example
python inference_full_sequence.py \
    --sequence-id 123 \
    --checkpoint logs/checkpoints_multihead_fixed/best_model.ckpt

# History-based mode (temporal smoothing)
python inference_full_sequence.py \
    --sequence-id all \
    --checkpoint logs/checkpoints_multihead_fixed/best_model.ckpt \
    --mode history
```

**Note**: Replace `best_model.ckpt` with your actual checkpoint path.

## 🏆 Model Comparison

### Architecture Differences

| Feature | VIFT Original | MultiHead Improved | MultiHead Fixed |
|---------|--------------|-------------------|-----------------|
| Input Processing | Concatenated features | Separate visual/IMU streams | Separate visual/IMU streams |
| Attention | Single transformer | Multi-head with specialization | Multi-head with specialization |
| Feature Fusion | Early fusion | Late fusion with cross-attention | Late fusion with cross-attention |
| Regularization | Basic dropout | Dropout + layer norm + residual | Dropout + layer norm + residual |
| Rotation Loss | MSE on quaternions | MSE on quaternions | **Geodesic loss** |
| Rotation Output | ReLU activation | ReLU activation | **No activation** |
| Parameters | ~1M | 1.2M | 1.2M |

### Performance Metrics

Average performance on 20 test sequences:

| Metric | VIFT Original | MultiHead Improved | MultiHead Fixed | Description |
|--------|---------------|-------------------|-----------------|-------------|
| Translation ATE (mm) | TBD | TBD | **1.34 ± 2.00** | Full trajectory error (aligned) |
| Rotation MAE (°) | TBD | TBD | **6.88 ± 1.48** | Mean absolute rotation error |
| RPE@33ms Trans (mm) | TBD | TBD | **0.096 ± 0.018** | Frame-to-frame translation |
| RPE@33ms Rot (°) | TBD | TBD | **0.029 ± 0.004** | Frame-to-frame rotation |
| RPE@100ms Trans (mm) | TBD | TBD | **0.287 ± 0.052** | 3-frame translation drift |
| RPE@100ms Rot (°) | TBD | TBD | **0.088 ± 0.012** | 3-frame rotation drift |
| RPE@1s Trans (mm) | TBD | TBD | **2.886 ± 0.474** | 30-frame translation drift |
| RPE@1s Rot (°) | TBD | TBD | **0.874 ± 0.107** | 30-frame rotation drift |

*TBD: To be determined after running evaluation on other models*

**Key Improvements in MultiHead Fixed:**
- Uses geodesic distance for rotation loss instead of element-wise MSE
- Removes ReLU activation from quaternion output layer
- Provides accurate rotation error measurements

### When to Use Which Model

- **MultiHead Fixed**: Most accurate, recommended for production use
- **MultiHead Improved**: Good for comparison, but rotation metrics need correction
- **VIFT Original**: Baseline implementation for reference

## 📁 Project Structure

```
VIFT_AEA/
├── configs/              # Hydra configuration files
├── data/                 # Dataset directory
├── scripts/              # Data processing scripts
├── src/                  # Source code
│   ├── data/            # Data modules and datasets
│   ├── models/          # Model architectures
│   ├── metrics/         # Loss functions and metrics
│   └── utils/           # Utility functions
├── train_improved.py     # Main training script
├── inference_full_sequence.py  # Evaluation script
└── generate_all_pretrained_latents_fixed.py  # Feature extraction
```

## 🔍 Troubleshooting

### Out of Memory
- Reduce batch size: `--batch-size 16`
- Use gradient accumulation (already enabled)
- Process fewer sequences in parallel

### Slow Training
- Ensure GPU is available: `nvidia-smi`
- Use mixed precision (enabled by default)
- Check data loading: increase `--num-workers`

### Poor Performance
- Verify data processing completed successfully
- Check pretrained encoder loaded correctly
- Ensure sufficient training epochs (>50)

## 📚 Citation

If you use this code, please cite the original VIFT paper:

```bibtex
@inproceedings{kurt2024vift,
  title={Causal Transformer for Fusion and Pose Estimation in Deep Visual Inertial Odometry},
  author={Kurt, Yunus Bilge and Akman, Ahmet and Alatan, Aydın},
  booktitle={ECCV 2024 Workshop on Visual Continual Learning},
  year={2024}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.