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
    --processed-dir aria_processed \
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
python prepare_aria_splits.py --source-dir aria_processed

# 4. Train all components from scratch (with quaternion output and improved loss)
torchrun --nproc_per_node=4 train_aria_from_scratch.py --opt-cnn sgd --opt-trf adamw --lr-cnn 1e-4 --lr-trf 5e-4 --epochs 37 --distributed --batch-size 4

# 5. Evaluate the trained model
python evaluate_from_scratch.py \
      --checkpoint checkpoints_from_scratch/best_model.pt \
      --data-dir aria_processed \
      --output-dir evaluation_from_scratch \
      --batch-size 4 \
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
python prepare_aria_splits.py --source-dir aria_processed

# 4. Train all components from scratch (improved loss weighting)

# Single GPU training
python train_aria_from_scratch.py \
    --data-dir aria_processed \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-4 \
    --checkpoint-dir checkpoints_from_scratch \
    --num-workers 4

# Multi-GPU training with distributed data parallel
torchrun --nproc_per_node=4 train_aria_from_scratch.py \
    --data-dir aria_processed \
    --epochs 50 \
    --batch-size 16 \
    --checkpoint-dir checkpoints_distributed \
    --distributed

# 5. Monitor training progress
tail -f train.log

# 6. Evaluate trained model
python evaluate_from_scratch.py \
    --checkpoint checkpoints_from_scratch/best_model.pt \
    --data-dir aria_processed \
    --output-dir evaluation_from_scratch \
    --batch-size 16 \
    --num-workers 4
```

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

1. **Visual Encoder**: 6-layer CNN processing consecutive image pairs
2. **IMU Encoder**: 3-layer 1D CNN processing IMU measurements
3. **Pose Transformer**: 4-layer transformer with 8 attention heads

The model predicts relative poses (3D translation + 3D rotation) between consecutive frames.

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
│   ├── data/                         # Dataset loaders
│   └── metrics/                      # Loss functions
├── configs/                          # Hydra configuration files
├── scripts/                          # Processing scripts
├── pretrained_models/                # Pretrained encoders
├── process_aria.py                   # Unified Aria data processing
├── train_efficient.py                # Aria training script
└── evaluate_stable_model.py          # Evaluation script
```

## Troubleshooting

1. **High validation errors**: Ensure you're using the fixed IMU extraction (`python process_aria.py`)
2. **Out of memory**: Reduce batch size or use gradient accumulation
3. **NaN losses**: Check for corrupted data samples or reduce learning rate

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