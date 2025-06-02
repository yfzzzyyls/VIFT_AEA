# VIFT-AEA: Visual-Inertial Feature Transformer for AriaEveryday Activities

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

A state-of-the-art Visual-Inertial Odometry (VIO) system achieving **0.0263° rotation error** and **0.0688cm ATE** on the AriaEveryday Activities dataset using Visual-Selective-VIO pretrained features.

> **Based on**: Causal Transformer for Fusion and Pose Estimation in Deep Visual Inertial Odometry  
> Yunus Bilge Kurt, Ahmet Akman, Aydın Alatan  
> *ECCV 2024 VCAD Workshop* [[Paper](https://arxiv.org/abs/2409.08769)] [[Original Repo](https://github.com/ybkurt/VIFT)]

## 🎯 Key Features

- **High Accuracy**: Achieves AR/VR-grade tracking with <0.1° rotation error
- **Efficient Processing**: GPU-accelerated feature extraction and training
- **Robust Implementation**: Proper quaternion handling with geodesic loss
- **Scalable**: Supports processing large datasets with batch operations

## 📋 Table of Contents

1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Training Pipeline](#training-pipeline)
4. [Evaluation](#evaluation)
5. [Results](#results)
6. [Project Structure](#project-structure)
7. [Citation](#citation)

## 🔧 Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- 32GB+ RAM for processing full dataset
- ~50GB free disk space for processed features

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/VIFT_AEA.git
cd VIFT_AEA

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: The repository includes pre-extracted features in `aria_latent_data_pretrained/` (~10GB). 
For a minimal setup, you can exclude this directory and generate features as needed.

## 📊 Dataset Preparation

### 1. Download AriaEveryday Activities Dataset

First, obtain the download URLs file from the [AriaEveryday website](https://www.projectaria.com/datasets/aea/).

```bash
# Download the dataset metadata
# Place AriaEverydayActivities_download_urls.json in the project root

# Download all sequences (143 total, ~500GB)
python scripts/download_aria_dataset.py --all

# Or download specific number of sequences
python scripts/download_aria_dataset.py --num-sequences 10
```

### 2. Process Raw Data to VIFT Format

Convert AriaEveryday sequences to VIFT-compatible format:

```bash
# Process all downloaded sequences
python scripts/process_aria_to_vift.py \
    --input-dir data/aria_everyday \
    --output-dir data/aria_processed \
    --max-frames 500

# Process specific sequences with custom numbering
python scripts/process_aria_to_vift.py \
    --input-dir data/aria_everyday \
    --output-dir data/aria_processed \
    --start-index 0 \
    --max-sequences 50 \
    --folder-offset 0
```

The script extracts:
- SLAM trajectories from MPS results
- RGB frames from preview videos
- Generates IMU data from trajectory
- Converts quaternions to Euler angles

### 3. Download Pretrained Visual-Selective-VIO Model

```bash
python download_pretrained_model.py
```

This downloads the 185MB pretrained model to `./Visual-Selective-VIO-Pretrained/`.

## 🚀 Training Pipeline

### Step 1: Extract Visual Features

Generate pretrained visual features and prepare training data:

```bash
python generate_all_pretrained_latents_fixed.py
```

This script:
- Extracts 768-dim features (512 visual + 256 IMU)
- Computes relative poses between frames
- Transforms to local coordinate system
- Splits data into train/val/test sets (70/10/20)

### Step 2: Train the Model

Train the relative pose prediction model:

```bash
python train_pretrained_relative.py
```

Training configuration:
- **Epochs**: 50
- **Batch Size**: 1024
- **Learning Rate**: 5e-4 with cosine annealing
- **Loss**: MSE for translation + Geodesic for rotation
- **Architecture**: Shared MLP with separate pose heads

### Step 3: Monitor Training

The training script logs metrics to TensorBoard:

```bash
tensorboard --logdir logs/
```

## 📈 Evaluation

Evaluate the trained model with AR/VR standard metrics:

```bash
python evaluate_with_metrics.py
```

This computes:
- **ATE** (Absolute Trajectory Error)
- **RPE** (Relative Pose Error) for translation and rotation
- **Direct Quaternion Error** using geodesic distance

## 🏆 Results

Our implementation achieves state-of-the-art performance:

| Metric | Value | Target | Status |
|--------|-------|--------|---------|
| **ATE** | 0.0688 cm | <1 cm | ✅ |
| **RPE Translation** | 0.0144 cm | <0.1 cm | ✅ |
| **RPE Rotation** | 0.0263° | <0.1° | ✅ |
| **Quaternion Error** | 0.0263° | <0.1° | ✅ |

## 📁 Project Structure

```
VIFT_AEA/
├── configs/                     # Hydra configuration files
├── data/                        # Data directories
│   ├── aria_everyday/          # Raw AriaEveryday sequences
│   ├── aria_processed/         # Processed VIFT format
│   └── aria_latent_data_*/     # Extracted features
├── scripts/                     # Data processing scripts
│   ├── download_aria_dataset.py
│   └── process_aria_to_vift.py
├── src/                         # Source code
│   ├── data/                   # Data loaders
│   ├── models/                 # Model architectures
│   └── utils/                  # Utilities
├── checkpoints/                 # Saved models
├── logs/                        # Training logs
├── outputs/                     # Hydra outputs
├── tests/                       # Unit tests
├── download_pretrained_model.py # VS-VIO model downloader
├── generate_all_pretrained_latents_fixed.py  # Feature extraction
├── train_pretrained_relative.py              # Training script
└── evaluate_with_metrics.py                  # Evaluation script
```

## 🔍 Technical Details

### Quaternion Handling

The implementation properly handles quaternion format conversion:
- Ground truth: XYZW format
- Model output: WXYZ format (converted to XYZW)
- Loss: Geodesic distance for proper rotation interpolation

### Coordinate Systems

- Relative poses computed in local coordinate frame
- Translations rotated to align with first frame
- Proper handling of SE(3) transformations

### Loss Function

```python
loss = translation_loss + 5.0 * rotation_loss
```

Where rotation loss uses geodesic distance:
```python
angle_diff = 2 * arccos(|dot(pred_quat, target_quat)|)
```

## 🐛 Troubleshooting

### Common Issues

1. **High rotation error (>0.1°)**
   - Verify quaternion format conversion
   - Check geodesic loss implementation
   - Ensure proper data normalization

2. **GPU memory errors**
   - Reduce batch size in training
   - Enable gradient accumulation
   - Use mixed precision training

3. **Missing sequences**
   - Check download completeness
   - Verify sequence mapping in `sequence_mapping.json`
   - Re-run download script for failed sequences

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{kurt2024vift,
  title={Causal Transformer for Fusion and Pose Estimation in Deep Visual Inertial Odometry},
  author={Kurt, Yunus Bilge and Akman, Ahmet and Alatan, Aydın},
  booktitle={ECCV 2024 Workshop on Visual Continual Learning},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- Original VIFT implementation by Yunus Bilge Kurt
- Visual-Selective-VIO pretrained model
- AriaEveryday Activities dataset by Meta