______________________________________________________________________

<div align="center">

# VIFT-AEA: Visual-Inertial Fusion Transformer for Aria Everyday Activities

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2409.08769-B31B1B.svg)](https://arxiv.org/abs/2409.08769)

</div>

## Description

This repository implements Visual-Inertial Fusion Transformer (VIFT) for both KITTI and Aria Everyday Activities datasets. VIFT uses a causal transformer architecture for fusion and pose estimation in deep visual-inertial odometry.

**Key Features:**
- Original VIFT implementation for KITTI dataset
- Extended support for Aria Everyday Activities dataset
- Fixed IMU temporal alignment for proper VIO training
- Cross-domain evaluation capabilities

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/VIFT_AEA
cd VIFT_AEA

# Create conda environment
conda create -n vift python=3.9
conda activate vift

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt

# Install additional packages for VIFT
pip install hydra-core omegaconf pytorch-lightning
pip install rootutils colorlog natsort

# (Optional) For SEA-RAFT visual encoder
# Note: SEA-RAFT will be cloned locally, no pip install needed
```

## Quick Start

### Option 1: Train from Scratch (Recommended)
```bash
# 1. Process Aria data
python process_aria.py

# 2. Create data splits
python organize_data_splits.py --data-dir /mnt/ssd_ext/incSeg-data/aria_processed

# 3. Train with distributed GPU
torchrun --nproc_per_node=4 train_aria_from_scratch.py \
    --data-dir /mnt/ssd_ext/incSeg-data/aria_processed \
    --epochs 50 \
    --batch-size 16 \
    --checkpoint-dir checkpoints_distributed \
    --distributed

# 4. Evaluate
python evaluate_from_scratch.py \
    --checkpoint checkpoints_distributed/best_model.pt \
    --data-dir /mnt/ssd_ext/incSeg-data/aria_processed \
    --output-dir evaluation_results
```

### Option 2: Train with SEA-RAFT (Advanced Motion Features)
```bash
# 1. Setup SEA-RAFT
python setup_searaft.py

# 2. Download pretrained weights (REQUIRED!)
# Go to: https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW
# Download: 'Tartan-C-T-TSKH432x960-S.pth' (~35MB)
cp ~/Downloads/Tartan-C-T-TSKH432x960-S.pth third_party/SEA-RAFT/SEA-RAFT-Sintel.pth

# 3. Verify setup
python setup_searaft.py  # Should show "weights already present"
python test_searaft_integration.py

# 4. Train with SEA-RAFT
torchrun --nproc_per_node=4 train_aria_from_scratch.py \
    --data-dir /mnt/ssd_ext/incSeg-data/aria_processed \
    --epochs 50 \
    --batch-size 16 \
    --checkpoint-dir checkpoints_searaft \
    --distributed \
    --use-searaft
```

## Complete Workflows

### Workflow 1: Train and Test VIFT on KITTI

```bash
# 1. Setup environment
source venv/bin/activate

# 2. Download KITTI dataset
cd data && sh data_prep.sh && cd ..

# 3. Download pretrained Visual-Selective-VIO encoder
# Place vf_512_if_256_3e-05.model in pretrained_models/

# 4. Cache latent features
cd data
python latent_caching.py      # Extract training features
python latent_val_caching.py  # Extract validation features
cd ..

# 5. Train model (multi-GPU)
python src/train.py experiment=latent_kitti_vio_paper trainer=ddp trainer.devices=-1 test=False

# 6. Evaluate on KITTI
python src/eval.py \
    ckpt_path=logs/train/runs/[timestamp]/checkpoints/best.ckpt \
    model=weighted_latent_vio_tf \
    data=latent_kitti_vio \
    trainer=gpu trainer.devices=1 logger=csv
```

### Workflow 2: Train and Test on Aria Everyday Activities

**Option A: Using Pre-trained Feature Encoders**
```bash
# 1. Setup environment
source venv/bin/activate

# 2. Process Aria data with proper IMU alignment (between-frames)
python process_aria.py

# 3. Generate latent features with fixed IMU handling
python generate_all_pretrained_latents_between_frames.py \
    --processed-dir /mnt/ssd_ext/incSeg-data/aria_processed \
    --output-dir aria_latent \
    --stride 10 \
    --skip-test

# 4. Train stable model
python train_efficient.py \
    --epochs 50 \
    --batch-size 32 \
    --lr 5e-5 \
    --data-dir aria_latent \
    --checkpoint-dir checkpoints_vift_stable

# 5. Evaluate
python evaluate_stable_model.py \
    --checkpoint checkpoints_vift_stable/best_model.pt \
    --data-dir aria_latent \
    --output-dir evaluation_results
```

**Option B: Training From Scratch (All Components)**
```bash
# 1. Setup environment
source venv/bin/activate

# 2. Process Aria data with proper IMU alignment
python process_aria.py

# 3. Create train/val splits
python prepare_aria_splits.py --source-dir /mnt/ssd_ext/incSeg-data/aria_processed

# 4. Train all components from scratch (with quaternion output and improved loss)
torchrun --nproc_per_node=4 train_aria_from_scratch.py --epochs 200 --distributed --batch-size 4

# 5. Evaluate the trained model
python evaluate_from_scratch.py \
      --checkpoint checkpoints_from_scratch/best_model.pt \
      --data-dir /mnt/ssd_ext/incSeg-data/aria_processed \
      --output-dir evaluation_from_scratch \
      --batch-size 16 \
      --num-workers 4

# 5. Monitor training
tensorboard --logdir checkpoints_from_scratch
```

### Workflow 3: Cross-Domain Testing (KITTI→Aria)

```bash
# 1. Train on KITTI (complete Workflow 1 first)

# 2. Extract Aria features using KITTI's pretrained encoder
python extract_aria_latent_features_for_kitti.py

# 3. Reorganize Aria data to match KITTI format
python reorganize_aria_to_kitti_structure.py

# 4. Test KITTI model on Aria
python src/eval.py \
    ckpt_path=logs/train/runs/[kitti_timestamp]/checkpoints/best.ckpt \
    model=weighted_latent_vio_tf \
    data=latent_kitti_vio \
    data.test_loader.root_dir=/home/external/VIFT_AEA/data/aria_latent_as_kitti/val_10 \
    trainer=gpu trainer.devices=1 logger=csv
```

### Workflow 4: Train VIFT From Scratch on Aria (All Components)

```bash
# 1. Setup environment
source venv/bin/activate

# 2. Process Aria data with proper IMU alignment
python process_aria.py

# 3. Create train/val/test splits
python organize_data_splits.py --data-dir /mnt/ssd_ext/incSeg-data/aria_processed --train-ratio 0.7 --val-ratio 0.15

# 4. Train all components from scratch (improved loss weighting)

# Multi-GPU training with distributed data parallel
torchrun --nproc_per_node=4 train_aria_from_scratch.py \
    --data-dir /mnt/ssd_ext/incSeg-data/aria_processed \
    --epochs 50 \
    --batch-size 16 \
    --checkpoint-dir checkpoints_distributed \
    --distributed

# 5. Monitor training progress
tail -f train.log

# 6. Evaluate trained model
python evaluate_from_scratch.py \
    --checkpoint checkpoints_distributed/best_model.pt \
    --data-dir /mnt/ssd_ext/incSeg-data/aria_processed \
    --output-dir evaluation_from_scratch \
    --batch-size 16 \
    --num-workers 4
```

### Workflow 5: Train on Full Aria Dataset with Optimizations

This workflow processes the full Aria Everyday Activities dataset with a 7:1:2 train/val/test split and includes all short-term accuracy optimizations.

```bash
# 1. Process full Aria dataset (first 1000 frames per sequence)
python process_aria.py \
    --input-dir /mnt/ssd_ext/incSeg-data/aria_everyday \
    --output-dir /mnt/ssd_ext/incSeg-data/aria_processed_full \
    --max-frames 500 \
    --train-ratio 0.7 \
    --val-ratio 0.1 \
    --all

# This will create:
# /mnt/ssd_ext/incSeg-data/aria_processed_full/
#   ├── train/     (70% of sequences)
#   ├── val/       (10% of sequences)
#   ├── test/      (20% of sequences)
#   └── dataset_split.json

# 2. Train with all optimizations (multi-GPU)
python train_aria_from_scratch.py \
    --data-dir /mnt/ssd_ext/incSeg-data/aria_processed_full \
    --use-dataparallel \
    --batch-size 16 \
    --checkpoint-dir checkpoints_full_optimized \
    --use-searaft \
    --epochs 100

torchrun --nproc_per_node=4 train_aria_from_scratch.py \
    --data-dir /mnt/ssd_ext/incSeg-data/aria_processed_full \
    --epochs 100 \
    --batch-size 8 \
    --checkpoint-dir checkpoints_full_optimized \
    --distributed \
    --use-searaft

# 3. Evaluate on test set
python evaluate_from_scratch.py \
    --checkpoint checkpoints_full_optimized/best_model.pt \
    --data-dir /mnt/ssd_ext/incSeg-data/aria_processed_full \
    --output-dir evaluation_full_optimized \
    --use-searaft \
    --batch-size 16

# 4. Evaluate 5-second window performance (key metric for AR/VR)
python validate_5s_window.py \
    --checkpoint checkpoints_full_optimized/best_model.pt \
    --data-dir /mnt/ssd_ext/incSeg-data/aria_processed_full \
    --window-size 100 \
    --stride 50 \
    --output-dir evaluation_5s_full
```

**Optimizations Included:**
- Expanded bias-GRU window (20 samples)
- Tighter χ² gating (85% confidence)
- LSTM-enhanced ZUPT detection
- Adaptive Q/R noise scaling (connector ready)
- SEA-RAFT visual encoder for better motion features

### Workflow 6: Train with SEA-RAFT Visual Encoder (Advanced Motion Features)

SEA-RAFT is a state-of-the-art optical flow network that can replace the simple CNN encoder for better motion estimation.

```bash
# 1. Setup SEA-RAFT (one-time setup)
python setup_searaft.py

# 2. REQUIRED: Download pretrained weights (SEA-RAFT won't work without them!)
# SEA-RAFT was trained for weeks on large optical flow datasets.

# Download weights from Google Drive:
# 1. Go to: https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW
# 2. Download ALL .pth files (or at minimum, files ending with '-S.pth' for small model)
# 3. Recommended model: 'Tartan-C-T-TSKH432x960-S.pth' (~35MB)

# Copy the model to the correct location:
cd /home/external/VIFT_AEA
cp ~/Downloads/Tartan-C-T-TSKH432x960-S.pth third_party/SEA-RAFT/SEA-RAFT-Sintel.pth

# Or if you downloaded all files in a zip:
cp drive-download-20250626T060023Z-1-001/Tartan-C-T-TSKH432x960-S.pth third_party/SEA-RAFT/SEA-RAFT-Sintel.pth

# Verify weights are in place (should show ~35M)
ls -lh third_party/SEA-RAFT/SEA-RAFT-Sintel.pth

# Re-run setup to confirm weights are detected
python setup_searaft.py

# 3. Test SEA-RAFT integration (verify weights loaded correctly)
python test_searaft_integration.py

# 5. Train with SEA-RAFT encoder (multi-GPU distributed)
torchrun --nproc_per_node=4 train_aria_from_scratch.py \
    --data-dir /mnt/ssd_ext/incSeg-data/aria_processed \
    --epochs 50 \
    --batch-size 16 \
    --checkpoint-dir checkpoints_critical_searaft \
    --distributed \
    --use-searaft

# 6. Compare CNN vs SEA-RAFT performance
# Train baseline CNN model first
torchrun --nproc_per_node=4 train_aria_from_scratch.py \
    --data-dir /mnt/ssd_ext/incSeg-data/aria_processed \
    --epochs 50 \
    --batch-size 16 \
    --checkpoint-dir checkpoints_cnn_baseline \
    --distributed

# Evaluate both models (auto-detects encoder type)
# For CNN model:
python evaluate_from_scratch.py \
    --checkpoint checkpoints_cnn_baseline/best_model.pt \
    --data-dir /mnt/ssd_ext/incSeg-data/aria_processed \
    --output-dir evaluation_cnn \
    --batch-size 16 \
    --num-workers 4

# For SEA-RAFT model:
python evaluate_from_scratch.py \
    --checkpoint checkpoints_critical_searaft/best_model.pt \
    --use-searaft \
    --data-dir /mnt/ssd_ext/incSeg-data/aria_processed \
    --output-dir evaluation_searaft_critical \
    --batch-size 16 \
    --num-workers 4

# Notes:
# - The script automatically detects encoder type from checkpoint
# - Use --use-searaft flag only if auto-detection fails
# - Clear error messages guide you if there's a mismatch
```

**SEA-RAFT Notes:**
- **REQUIRES pretrained weights** - was trained for weeks on optical flow datasets
- Without pretrained weights, it's just a random CNN and won't work properly
- ~4.3x slower than CNN encoder but expected to improve accuracy by 10-15%
- Requires ~2GB additional GPU memory for feature extraction
- Uses frozen pretrained features from optical flow task
- Preserves spatial tokens (4×4) for better parallax reasoning

## Important: IMU Data Format

This repository includes a critical fix for IMU temporal alignment:

### Original Implementation (Issues)
- IMU window: [-25ms, +25ms] centered on each frame
- Creates overlapping data and violates causality
- Can lead to poor VIO performance

### Fixed Implementation (Recommended)
- IMU extracted between consecutive frames [t_i, t_{i+1})
- No overlap, proper temporal alignment
- Ensures correct motion association for VIO

**Always use the fixed version for new experiments!**

## Model Architecture

VIFT consists of three main components:

1. **Visual Encoder**: 
   - **Default**: 6-layer CNN processing consecutive image pairs
   - **Optional**: SEA-RAFT feature encoder for improved motion estimation (see Workflow 5)
2. **IMU Encoder**: 3-layer 1D CNN processing IMU measurements with multi-scale temporal pooling
3. **Pose Transformer**: 4-layer transformer with 8 attention heads (configurable)

The model outputs 7-DoF poses (3D translation + 4D quaternion rotation) between consecutive frames.

## Key Improvements in This Implementation

### 1. Corrected Loss Function
- **Original**: Translation loss was incorrectly down-weighted (`trans_loss / 3 + rot_loss`)
- **Fixed**: Proper weighting with `α × trans_loss + β × scale_loss + rot_loss`
- Default weights: α=10.0, β=5.0 (translation needs higher weight as it's in meters)

### 2. Scale Drift Prevention
- Added explicit scale consistency loss term
- Helps prevent accumulating scale errors in long sequences
- Computes L1 loss between predicted and ground truth translation magnitudes

### 3. Consistent IMU Preprocessing
- Training and evaluation now use identical IMU preprocessing
- Raw IMU data (including gravity) is used consistently
- Prevents distribution shift between training and testing

### 4. Improved Evaluation Metrics
- Added APE (Absolute Pose Error) and RPE (Relative Pose Error) metrics
- Scale drift percentage tracking
- Interactive 3D HTML trajectory visualizations using Plotly
- CSV export of trajectories for further analysis

### 5. SEA-RAFT Visual Encoder Option
- Integrated state-of-the-art optical flow features
- Uses pretrained SEA-RAFT's feature network (fnet)
- Preserves spatial structure for better motion reasoning
- Available via `--use-searaft` flag in training

## Performance Results

| Model | Training Data | Test Data | Translation Error | Rotation Error |
|-------|--------------|-----------|------------------|----------------|
| VIFT | KITTI | KITTI | 3.27% | 1.78° |
| VIFT | KITTI | Aria | 3.27% | 1.77° |
| VIFT-AEA | Aria | Aria | 0.84 cm | 1.3° |

## Project Structure

```
VIFT_AEA/
├── src/                              # Source code
│   ├── models/                       # Model architectures
│   │   └── components/               
│   │       ├── vsvio.py             # Main VIFT model
│   │       └── searaft_encoder.py   # SEA-RAFT feature encoder
│   ├── data/                         # Dataset loaders
│   └── metrics/                      # Loss functions
├── configs/                          # Hydra configuration files
├── scripts/                          # Processing scripts
├── pretrained_models/                # Pretrained encoders
├── third_party/                      # External dependencies (gitignored)
│   └── SEA-RAFT/                    # SEA-RAFT repository (after setup)
├── process_aria.py                   # Unified Aria data processing
├── train_aria_from_scratch.py        # Main training script with --use-searaft
├── setup_searaft.py                  # SEA-RAFT installation script
├── test_searaft_integration.py       # Test SEA-RAFT integration
└── evaluate_from_scratch.py          # Evaluation script
```

## Troubleshooting

1. **High validation errors**: Ensure you're using the fixed IMU extraction (`python process_aria.py`)
2. **Out of memory**: Reduce batch size or use gradient accumulation
3. **NaN losses**: Check for corrupted data samples or reduce learning rate
4. **SEA-RAFT import errors**: Run `python setup_searaft.py` to fix imports and dependencies
5. **SEA-RAFT weights missing**: Download manually from Google Drive (REQUIRED, not optional!)
6. **SEA-RAFT performing poorly**: Ensure you downloaded pretrained weights - random init won't work

## Citation

If you use this code, please cite:

```bibtex
@article{vift2024,
  title={Causal Transformer for Fusion and Pose Estimation in Deep Visual Inertial Odometry},
  author={...},
  journal={ECCV Workshop on Visual-Centric Autonomous Driving},
  year={2024}
}
```

## License

This project is licensed under the MIT License.