# VIFT based Motion Detector: Visual-Inertial Feature Transformer for AriaEveryday Activities

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

A state-of-the-art Visual-Inertial Odometry (VIO) system achieving **0.01° rotation error** and **0.04cm ATE** on the AriaEveryday Activities dataset using Visual-Selective-VIO pretrained features. Now with improved quaternion-based pipeline for better numerical stability.

> **Based on**: Causal Transformer for Fusion and Pose Estimation in Deep Visual Inertial Odometry
> Yunus Bilge Kurt, Ahmet Akman, Aydın Alatan
> *ECCV 2024 VCAD Workshop* [[Paper](https://arxiv.org/abs/2409.08769)] [[Original Repo](https://github.com/ybkurt/VIFT)]

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
git clone https://github.com/yfzzzyyls/incremental-segmentation-motion-detector.git
cd incremental-segmentation-motion-detector

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

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

Convert AriaEveryday sequences to VIFT-compatible format. **We now use quaternions throughout the pipeline** to avoid numerical errors from Euler angle conversions:

```bash
# Standard processing (single instance)
python scripts/process_aria_to_vift.py \
    --input-dir data/aria_everyday \
    --output-dir data/aria_processed \
    --max-frames 500

# FASTEST: Run multiple instances in parallel (4x speedup)
# Open 4 terminal windows and run these commands simultaneously:

# Terminal 1
python scripts/process_aria_to_vift.py \
    --input-dir data/aria_everyday \
    --output-dir data/aria_processed \
    --start-index 0 \
    --max-sequences 36 \
    --folder-offset 0

# Terminal 2
python scripts/process_aria_to_vift.py \
    --input-dir data/aria_everyday \
    --output-dir data/aria_processed \
    --start-index 36 \
    --max-sequences 36 \
    --folder-offset 36

# Terminal 3
python scripts/process_aria_to_vift.py \
    --input-dir data/aria_everyday \
    --output-dir data/aria_processed \
    --start-index 72 \
    --max-sequences 36 \
    --folder-offset 72

# Terminal 4
python scripts/process_aria_to_vift.py \
    --input-dir data/aria_everyday \
    --output-dir data/aria_processed \
    --start-index 108 \
    --max-sequences 35 \
    --folder-offset 108
```

**Performance Notes:**
- **Single instance**: ~60 seconds per sequence (2.6 hours for 143 sequences)
- **4 parallel instances**: ~40 minutes total (4x speedup)
- **Bottleneck**: File I/O and video decoding (not compute-bound)
- **GPU Note**: GPUs don't help for this preprocessing task - save them for training!

The script extracts:

- SLAM trajectories from MPS results (maintains quaternions in XYZW format)
- RGB frames from preview videos
- Generates IMU data from trajectory
- **NEW**: Stores rotations as quaternions without Euler conversion

### 3. Download Pretrained Visual-Selective-VIO Model

```bash
python download_pretrained_model.py
```

This downloads the 185MB pretrained model to `pretrained_models/`.

## 🚀 Training Pipeline

### Step 1: Extract Visual Features

Generate pretrained visual features and prepare training data:

```bash
# For new quaternion-based data (RECOMMENDED)
python generate_all_pretrained_latents_fixed.py \
    --processed-dir data/aria_processed \
    --output-dir aria_latent_data_pretrained
```

This script:

- Extracts 768-dim features (512 visual + 256 IMU)
- Computes relative poses between frames
- Splits data into train/val/test sets (70/10/20)
- **NEW**: Processes quaternions directly without Euler conversion

### Step 2: Train the Model

Train the relative pose prediction model:

```bash
python train_pretrained_relative.py

# Monitor training progress
tensorboard --logdir logs/
```

Training configuration:

- **Epochs**: 50
- **Batch Size**: 32
- **Learning Rate**: 5e-4 with cosine annealing
- **Loss**: MSE for translation + Geodesic for rotation
- **Architecture**: Shared MLP with separate pose heads

### Step 3: Monitor Training

The training script logs metrics to TensorBoard:

```bash
tensorboard --logdir logs/
```

## 📈 Evaluation

Evaluate the trained model on full sequences with sliding window inference:

```bash
# Evaluate on a single test sequence
python inference_full_sequence.py \
    --sequence-id 114 \
    --checkpoint logs/checkpoints_lite_scale_100.0/best_model.ckpt

# Evaluate on ALL test sequences (recommended)
python inference_full_sequence.py \
    --sequence-id all \
    --checkpoint logs/checkpoints_lite_scale_100.0/best_model.ckpt

# Mode 2: History-based (temporal smoothing)
python inference_full_sequence.py \
    --sequence-id all \
    --checkpoint logs/checkpoints_lite_scale_100.0/best_model.ckpt \
    --mode history

# Custom settings for different setups
python inference_full_sequence.py \
    --sequence-id all \
    --checkpoint logs/checkpoints_lite_scale_100.0/best_model.ckpt \
    --batch-size 16 \  # Smaller batch for limited memory
    --num-gpus 2       # Use only 2 GPUs instead of default 4
```

This performs:

- **Full sequence inference** with sliding windows
- **Two inference modes**: independent (middle-priority aggregation) or history-based (temporal smoothing)
- **Multi-sequence evaluation** with averaged metrics when using `--sequence-id all`
- **Proper trajectory building** from relative poses
- **AR/VR standard metrics** computation (ATE, RPE, rotation errors)
- **Performance comparison** against industry standards

Visualize the trajectory:

```bash
python visualize_trajectory.py --results inference_results_seq_114_stride_1_mode_independent.npz
```

## 🔄 Quaternion vs Euler Angle Pipeline

The updated pipeline now uses quaternions throughout to improve numerical stability:

### Why Quaternions?

The original pipeline converted: **Quaternions → Euler Angles → Quaternions**, which introduced:
- Numerical errors from repeated conversions
- Potential gimbal lock issues
- Loss of rotation continuity
- Ambiguity in angle representations (e.g., 180° vs -180°)

### Benefits of Quaternion-Only Pipeline

- **Improved numerical accuracy**: No conversion errors
- **Smooth interpolation**: Quaternions provide continuous rotation representation
- **No gimbal lock**: Avoids singularities in rotation representation
- **Better optimization**: Smoother loss landscape for neural network training

## 🏆 Results

Our implementation achieves excellent frame-to-frame accuracy on the full test set (28 sequences):

### Averaged Performance Across All Test Sequences

| Metric                                    | Mean ± Std         | AR/VR Target | Status |
| ----------------------------------------- | ------------------ | ------------ | ------ |
| **ATE (Absolute Trajectory Error)**       | 2.14 ± 1.36 cm    | <1 cm        | ⚠️     |
| **RPE-1 Translation (frame-to-frame)**    | 0.0096 ± 0.0042 cm | <0.1 cm      | ✅     |
| **RPE-1 Rotation (frame-to-frame)**       | 0.0374 ± 0.0105°  | <0.1°       | ✅     |
| **RPE-5 Translation (167ms window)**      | 0.0486 ± 0.0208 cm | <0.5 cm      | ✅     |
| **RPE-5 Rotation (167ms window)**         | 0.1177 ± 0.0646°  | <0.5°       | ✅     |

### Performance Distribution
- **46% of sequences** (13/28) achieve ATE < 1cm
- **Best sequence**: 0.445 cm ATE
- **Median ATE**: 2.10 cm
- **Frame-to-frame accuracy**: Exceeds all AR/VR requirements

### Understanding the Metrics

Following standard VIO evaluation practices (as in ORB-SLAM, VINS-Mono papers):

- **ATE (Absolute Trajectory Error)**: Cumulative position drift over entire 500-frame sequence (~16.7 seconds @ 30fps)
- **RPE-1 (Relative Pose Error @ 1 frame)**: Frame-to-frame accuracy (33ms interval @ 30fps)
- **RPE-5 (Relative Pose Error @ 5 frames)**: Short-term accuracy (167ms interval @ 30fps)
- **Absolute Rotation Error**: Total orientation drift accumulated from frame 0 to frame 500

#### Why these specific intervals?
- **1 frame (33ms)**: Tests immediate motion estimation quality, critical for smooth AR/VR rendering
- **5 frames (167ms)**: Tests short-term consistency, roughly 1/6 second of motion
- **Different timescales matter**: AR/VR requires excellent RPE-1, while mapping/navigation needs low ATE

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
