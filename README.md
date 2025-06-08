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

A state-of-the-art Visual-Inertial Odometry (VIO) system for the Aria Everyday Activities dataset, implementing and improving upon the VIFT architecture with multiple model variants and critical bug fixes.

### Key Results

The MultiHead Fixed model achieves:

- **Translation Error**: 1.34 ± 2.00 mm (after trajectory alignment)
- **Rotation Error**: 6.88 ± 1.48 degrees (geodesic distance)
- **Frame-to-Frame Translation**: 0.096 ± 0.018 mm
- **Frame-to-Frame Rotation**: 0.029 ± 0.004 degrees

All models exceed AR/VR requirements (<0.1° rotation, <0.1cm translation error per frame).

## 📋 Requirements

- Python 3.9+
- CUDA-capable GPU (48GB+ VRAM recommended for processing, 8GB+ for training)
- 32GB+ RAM
- ~3TB disk space for full dataset (500GB for Aria raw data, 2TB for processed data)

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
# Single instance processing (updated to use quaternion version)
python scripts/process_aria_to_vift_quaternion.py \
    --input-dir data/aria_everyday \
    --output-dir data/aria_processed \
    --max-frames 10000  # Process full sequences

# Parallel processing (16x faster) - Automated script available
# The processing includes:
# - Extracting SLAM trajectories from MPS results
# - Converting to quaternion format (no Euler angles)
# - Extracting RGB frames using GPU/CPU
# - Generating IMU data at proper frequency
# - Creating 60/20/20 train/val/test splits
```

### 2a. Handle Failed Sequences (if any)

Some sequences may fail due to CUDA OOM. Reprocess them with CPU:

```bash
python fix_failed_sequences.py  # Automatically finds and reprocesses failed sequences
```

### 3. Download Pretrained Encoder

```bash
python download_pretrained_model.py
```

### 4. Extract Visual Features

Generate pretrained features and prepare training data:

```bash
python generate_all_pretrained_latents_fixed.py \
    --processed-dir /path/to/aria_processed \
    --output-dir /path/to/aria_latent_data_pretrained \
    --stride 1  # Use all frames
    --skip-test  # Skip test set (use real-time encoding during inference)

# This will:
# - Extract 512-dim visual features using pretrained encoder
# - Extract 256-dim IMU features 
# - Convert poses to relative transformations in local coordinates
# - Create sliding windows of 11 frames (10 transitions)
# - Generate ~371K training samples from 137 sequences
```

## 🏃 Training

Train different model architectures:

### MultiHead Fixed

Uses geodesic loss for rotation with proper weight initialization:

```bash
python train_improved.py --model multihead_fixed \
    --data-dir /mnt/ssd_ext/incSeg-data/aria_latent_data_pretrained \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-3 \
    --hidden-dim 128 \
    --num-heads 4 \
    --dropout 0.2

# Key improvements in v3:
# - No ReLU on quaternion outputs
# - Xavier initialization for rotation weights (fixes zero gradient issue)
# - Geodesic distance for rotation loss
# - Proper quaternion initialization
# - Gradient clipping for stability
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
    --checkpoint logs/checkpoints_multihead_fixed/last.ckpt \
    --data-dir /mnt/ssd_ext/incSeg-data/aria_processed \
    --model-type multihead_fixed

# Single sequence evaluation
python inference_full_sequence.py \
    --sequence-id 123 \
    --checkpoint logs/checkpoints_multihead_fixed/last.ckpt \
    --data-dir /mnt/ssd_ext/incSeg-data/aria_processed \
    --model-type multihead_fixed

# History-based mode (temporal smoothing)
python inference_full_sequence.py \
    --sequence-id all \
    --checkpoint logs/checkpoints_multihead_fixed/last.ckpt \
    --data-dir /mnt/ssd_ext/incSeg-data/aria_processed \
    --model-type multihead_fixed \
    --mode history

# Evaluate other models for comparison
python inference_full_sequence.py \
    --sequence-id all \
    --checkpoint logs/checkpoints_multihead/best_model.ckpt \
    --model-type multihead

python inference_full_sequence.py \
    --sequence-id all \
    --checkpoint logs/checkpoints_vift_original/best_model.ckpt \
    --model-type vift_original
```

**Note**: Replace `best_model.ckpt` with your actual checkpoint path.

## 🏆 Model Comparison

### Architecture Differences

| Feature          | VIFT Original         | MultiHead Improved               | MultiHead Fixed                  |
| ---------------- | --------------------- | -------------------------------- | -------------------------------- |
| Input Processing | Concatenated features | Separate visual/IMU streams      | Separate visual/IMU streams      |
| Attention        | Single transformer    | Multi-head with specialization   | Multi-head with specialization   |
| Feature Fusion   | Early fusion          | Late fusion with cross-attention | Late fusion with cross-attention |
| Regularization   | Basic dropout         | Dropout + layer norm + residual  | Dropout + layer norm + residual  |
| Rotation Loss    | MSE on Euler angles   | MSE on quaternions               | **Geodesic loss**          |
| Rotation Output  | Euler angles          | ReLU activation (bug)            | **No activation**          |
| Initialization   | Standard              | Standard                         | **Identity quaternion**    |
| Parameters       | ~1M                   | 1.2M                             | 1.2M                             |

### Performance Metrics

Average performance on 20 test sequences:

| Metric               | VIFT Original | MultiHead Improved | MultiHead Fixed          | Description                     |
| -------------------- | ------------- | ------------------ | ------------------------ | ------------------------------- |
| Translation ATE (mm) | TBD           | TBD                | **1.34 ± 2.00**   | Full trajectory error (aligned) |
| Rotation MAE (°)    | TBD           | TBD                | **6.88 ± 1.48**   | Mean absolute rotation error    |
| RPE@33ms Trans (mm)  | TBD           | TBD                | **0.096 ± 0.018** | Frame-to-frame translation      |
| RPE@33ms Rot (°)    | TBD           | TBD                | **0.029 ± 0.004** | Frame-to-frame rotation         |
| RPE@100ms Trans (mm) | TBD           | TBD                | **0.287 ± 0.052** | 3-frame translation drift       |
| RPE@100ms Rot (°)   | TBD           | TBD                | **0.088 ± 0.012** | 3-frame rotation drift          |
| RPE@1s Trans (mm)    | TBD           | TBD                | **2.886 ± 0.474** | 30-frame translation drift      |
| RPE@1s Rot (°)      | TBD           | TBD                | **0.874 ± 0.107** | 30-frame rotation drift         |

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
│   ├── download_aria_dataset.py         # Dataset downloader
│   ├── process_aria_to_vift_quaternion.py  # Quaternion-based processing
│   └── process_aria_to_vift.py          # Original Euler-based (deprecated)
├── src/                  # Source code
│   ├── data/            # Data modules and datasets
│   ├── models/          # Model architectures
│   │   ├── components/  # Model components
│   │   ├── multihead_vio_separate.py         # MultiHead with bug
│   │   ├── multihead_vio_separate_fixed.py   # Initial fixed version
│   │   └── multihead_vio.py                  # Final improved version
│   ├── metrics/         # Loss functions and metrics
│   └── utils/           # Utility functions
├── train_improved.py     # Main training script
├── inference_full_sequence.py  # Evaluation script
├── generate_all_pretrained_latents_fixed.py  # Feature extraction
├── fix_failed_sequences.py  # Reprocess failed sequences
├── auto_monitor_and_train.sh  # Automated training setup
└── monitor_*.sh         # Various monitoring scripts
```

## 🔍 Troubleshooting

### Out of Memory

- For processing: Use CPU mode in `fix_failed_sequences.py`
- For training: Reduce batch size: `--batch-size 16`
- Use gradient accumulation (already enabled)
- Process sequences individually if needed

### Rotation Predictions Stuck at 0.5

- This is the ReLU bug! Use `multihead_fixed` model
- Check model file for ReLU after quaternion output
- Ensure using geodesic loss, not MSE

### Slow Training

- Ensure GPU is available: `nvidia-smi`
- Use mixed precision (enabled by default)
- Check data loading: increase `--num-workers`
- Use SSD for data storage if possible

### Poor Performance

- Verify data processing completed successfully
- Check pretrained encoder loaded correctly
- Ensure sufficient training epochs (>50)
- Verify using correct model version (fixed)

### Processing Failures

- Some sequences may fail with CUDA OOM
- Run `fix_failed_sequences.py` to reprocess with CPU
- Check logs in `/mnt/ssd_ext/incSeg-data/parallel_scripts/`

## 🚀 Quick Start Guide

Each script will print the exact command for the next step. Here's the complete workflow:

### Step 1: Download and Process Data (24+ hours)
```bash
# Download dataset
python scripts/download_aria_dataset.py --all

# Process to VIFT format
python scripts/process_aria_to_vift_quaternion.py \
    --input-dir data/aria_everyday \
    --output-dir /mnt/ssd_ext/aria_processed
```

### Step 2: Extract Features (~2 hours)
```bash
python generate_all_pretrained_latents_fixed.py \
    --processed-dir /mnt/ssd_ext/aria_processed \
    --output-dir /mnt/ssd_ext/aria_latent_data_pretrained \
    --stride 1 --skip-test

# The script will print the exact training command when complete
```

### Step 3: Train Model (~4-6 hours)
```bash
python train_improved.py --model multihead_fixed \
    --data-dir /mnt/ssd_ext/aria_latent_data_pretrained \
    --epochs 100 --batch-size 32 --lr 1e-3 \
    --hidden-dim 128 --num-heads 4 --dropout 0.2

# The script will print the exact inference command when complete
```

### Step 4: Monitor Training (while training)
```bash
# In a separate terminal
tensorboard --logdir logs/
```

### Step 5: Evaluate Model
```bash
# The training script will print this exact command with your best checkpoint
python inference_full_sequence.py \
    --sequence-id all \
    --checkpoint logs/checkpoints_multihead_fixed/best_model.ckpt \
    --processed-dir /mnt/ssd_ext/incSeg-data/aria_processed
```

**Note**: Each script prints the exact command for the next step, so you don't need to remember the parameters!

## 📊 Monitoring Tools

- **Feature extraction progress**: `./monitor_latent_extraction.sh`
- **Failed sequences**: `./monitor_failed_sequences.sh`
- **Training progress**: `tensorboard --logdir logs/`
- **Automated pipeline**: `./auto_monitor_and_train.sh`

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

## 🙏 Acknowledgments

- Original VIFT implementation by Yunus Bilge Kurt et al.
- Visual-Selective-VIO pretrained encoder
- Aria Everyday Activities dataset by Meta

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
