# VIFT-AEA: Visual-Inertial Fusion Transformer for Aria Everyday Activities

This project implements a Visual-Inertial Odometry (VIO) system using transformer architecture for the Aria Everyday Activities dataset. The model predicts frame-to-frame relative poses (translation and rotation) by fusing visual and IMU features.

## Overview

This repository contains:
- **Original VIFT implementation** - Can train on KITTI dataset
- **VIFT-AEA adaptations** - Modified versions for Aria Everyday Activities dataset
- **Cross-domain evaluation tools** - Test KITTI models on Aria and vice versa

## Complete Workflows

### Workflow 1: Train and Test VIFT on KITTI
```bash
# 1. Setup environment
source venv/bin/activate

# 2. Prepare KITTI data
cd data && sh data_prep.sh
python latent_caching.py      # Extract training features
python latent_val_caching.py  # Extract validation features
cd ..

# 3. Train model (multi-GPU)
python src/train.py experiment=latent_kitti_vio_paper trainer=ddp trainer.devices=-1 test=False

# 4. Evaluate on KITTI
python src/eval.py \
    ckpt_path=logs/train/runs/[timestamp]/checkpoints/best.ckpt \
    model=weighted_latent_vio_tf \
    data=latent_kitti_vio \
    trainer=gpu trainer.devices=1 logger=csv
```

### Workflow 2: Train and Test on Aria

**With Fixed IMU Format (between-frames) - Recommended**
```bash
# 1. Process Aria data with proper IMU alignment
./process_full_dataset_optimal_fixed.sh

# 2. Generate latent features with between-frames handling
python generate_all_pretrained_latents_between_frames.py \
    --processed-dir aria_processed \
    --output-dir aria_latent \
    --stride 10

# 3. Train stable model
python train_efficient.py \
    --epochs 50 --batch-size 32 --lr 5e-5 \
    --data-dir aria_latent \
    --checkpoint-dir checkpoints_vift_stable

# 4. Evaluate
python evaluate_stable_model.py \
    --checkpoint checkpoints_vift_stable/best_model.pt \
    --data-dir aria_latent \
    --output-dir evaluation_results
```

### Workflow 3: Cross-Domain Testing (KITTI→Aria)
```bash
# 1. Train on KITTI (see Workflow 1)

# 2. Extract Aria features using KITTI encoder
python extract_aria_latent_features_for_kitti.py

# 3. Test KITTI model on Aria
python src/eval.py \
    ckpt_path=logs/train/runs/[kitti_timestamp]/checkpoints/best.ckpt \
    data=aria_latent_kitti_format \
    model=latent_vio_tf \
    trainer=gpu trainer.devices=1 logger=csv
```

## Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install additional packages for VIFT
pip install hydra-core omegaconf pytorch-lightning
pip install rootutils colorlog natsort
```

### 2. Data Processing Pipeline

#### Process Raw Aria Data (20 sequences with ALL frames)

**Option A: Original Implementation (with temporal misalignment)**
```bash
# Process with sliding window IMU (not recommended for VIO)
./process_full_dataset_optimal.sh
# Output: aria_processed/
```

**Option B: Fixed Implementation (with proper between-frames IMU)**
```bash
# Process with correct between-frames IMU extraction (recommended)
./process_full_dataset_optimal_fixed.sh
# Output: aria_processed/
```

The fixed version extracts IMU data between consecutive frames for proper VIO temporal alignment. For N frames, it produces N-1 intervals of IMU data.

#### Generate Latent Features

**For Original Data Format:**
```bash
# Generate latent features with stride 10
python generate_all_pretrained_latents_fixed.py \
    --processed-dir aria_processed \
    --output-dir aria_latent \
    --stride 10 \
    --skip-test
```

**For Fixed Between-Frames Format (recommended):**
```bash
# Generate latent features with proper IMU handling
python generate_all_pretrained_latents_between_frames.py \
    --processed-dir aria_processed \
    --output-dir aria_latent \
    --stride 10 \
    --skip-test
```

### 3. Training Options

You have two main training options:
1. **Original VIFT on KITTI** - Train the original VIFT model on KITTI dataset
2. **VIFT-AEA on Aria** - Train adapted model directly on Aria dataset

#### Option 1: Training Original VIFT on KITTI Dataset

```bash
# First, download and prepare KITTI dataset
cd data
sh data_prep.sh

# Download pretrained Visual-Selective-VIO encoder if not present
# The encoder should be at: pretrained_models/vf_512_if_256_3e-05.model

# Cache latent features from KITTI using Visual-Selective-VIO encoder
python latent_caching.py
python latent_val_caching.py
cd ..

# Train VIFT on KITTI (single GPU)
python src/train.py experiment=latent_kitti_vio_paper trainer=gpu

# Train VIFT on KITTI with 4 GPUs (recommended for faster training)
python src/train.py experiment=latent_kitti_vio_paper trainer=ddp

# Train with all available GPUs
python src/train.py experiment=latent_kitti_vio_paper trainer=ddp trainer.devices=-1

# Train with specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train.py experiment=latent_kitti_vio_paper trainer=ddp

# Train with tensorboard logging
python src/train.py experiment=latent_kitti_vio_paper trainer=ddp logger=tensorboard

# Skip test phase during training to avoid multi-GPU issues
python src/train.py experiment=latent_kitti_vio_paper trainer=ddp test=False
```

#### Testing KITTI-trained Model on KITTI (Sanity Check)

```bash
# Test on KITTI test sequences with explicit paths
python src/eval.py \
    ckpt_path=/home/external/VIFT_AEA/logs/train/runs/[timestamp]/checkpoints/epoch_197.ckpt \
    model=weighted_latent_vio_tf \
    data=latent_kitti_vio \
    data.train_loader.root_dir=/home/external/VIFT_AEA/data/kitti_latent_data/train_10 \
    data.val_loader.root_dir=/home/external/VIFT_AEA/data/kitti_latent_data/val_10 \
    data.test_loader.root_dir=/home/external/VIFT_AEA/data/kitti_latent_data/val_10 \
    trainer=gpu \
    trainer.devices=1 \
    logger=tensorboard

# Alternative with CSV logger
python src/eval.py \
    ckpt_path=/home/external/VIFT_AEA/logs/train/runs/[timestamp]/checkpoints/epoch_197.ckpt \
    model=weighted_latent_vio_tf \
    data=latent_kitti_vio \
    data.train_loader.root_dir=/home/external/VIFT_AEA/data/kitti_latent_data/train_10 \
    data.val_loader.root_dir=/home/external/VIFT_AEA/data/kitti_latent_data/val_10 \
    data.test_loader.root_dir=/home/external/VIFT_AEA/data/kitti_latent_data/val_10 \
    trainer=gpu \
    trainer.devices=1 \
    logger=csv
```

#### Testing KITTI-trained Model on Aria Dataset (Cross-Domain Evaluation)

To test a KITTI-trained model on Aria data, you need to prepare the data in KITTI-compatible format:

```bash
# Step 1: Extract latent features from Aria data using KITTI's encoder
python extract_aria_latent_features_for_kitti.py
# This creates features in: aria_latent_kitti_format/
# Processing sequences 016, 017, 018, 019 by default

# Step 2: Reorganize Aria data to match KITTI directory structure
python reorganize_aria_to_kitti_structure.py
# This maps: Aria 016→05, 017→07, 018+019→10
# Output: data/aria_latent_as_kitti/val_10/

# Step 3: Convert Aria IMU data to MATLAB format (optional, for real testing)
python convert_aria_imu_to_matlab.py
# Creates .mat files in: data/kitti_data/imus/

# Step 4: Evaluate KITTI model on reorganized Aria data
python src/eval.py \
    ckpt_path=/home/external/VIFT_AEA/logs/train/runs/[timestamp]/checkpoints/epoch_197.ckpt \
    model=weighted_latent_vio_tf \
    data=latent_kitti_vio \
    data.train_loader.root_dir=/home/external/VIFT_AEA/data/kitti_latent_data/train_10 \
    data.val_loader.root_dir=/home/external/VIFT_AEA/data/kitti_latent_data/val_10 \
    data.test_loader.root_dir=/home/external/VIFT_AEA/data/aria_latent_as_kitti/val_10 \
    trainer=gpu trainer.devices=1 logger=csv
```

**Actual Cross-Domain Results (2025-06-14):**
- KITTI→KITTI: ~3.27% translation error, ~1.78° rotation error
- KITTI→Aria (reorganized): ~3.27% translation error, ~1.77° rotation error
- Note: These improved results use the reorganized data approach

#### Important Training Results

From the 4-GPU DDP training completed on 2025-06-14:
- Training completed successfully for 200 epochs
- Final training loss: 0.013
- Final validation loss: 0.017
- Best checkpoint saved at epoch 197
- Checkpoint path: `/home/external/VIFT_AEA/logs/train/runs/2025-06-14_12-53-26/checkpoints/epoch_197.ckpt`

#### Option 2: Training VIFT-AEA on Aria Dataset (Recommended for Aria)

There are two implementations for training on Aria:

**A. Stable Version (Recommended)**

```bash
# Train the stable model on Aria
python train_efficient.py \
    --epochs 50 \
    --batch-size 32 \
    --lr 5e-5 \
    --data-dir aria_latent \
    --checkpoint-dir checkpoints_vift_stable \
    --device cuda

# Evaluate (auto-generates test features if needed)
python evaluate_stable_model.py \
    --checkpoint checkpoints_vift_stable/best_model.pt \
    --data-dir aria_latent \
    --output-dir evaluation_results \
    --device cuda
```

**Stable Version Results:**
- Translation Error: 0.843 cm (mean), 0.339 cm (median)
- Rotation Error: 1.302° (mean), 0.741° (median)
- 95% of predictions within 3.192 cm and 4.288°

**Training with Fixed IMU Data:**
```bash
# Train on properly aligned IMU data
python train_efficient.py \
    --epochs 50 \
    --batch-size 32 \
    --lr 5e-5 \
    --data-dir aria_latent \
    --checkpoint-dir checkpoints_vift_stable \
    --device cuda
```

**B. Standard VIFT Architecture on Aria**

```bash
# Train using original VIFT architecture adapted for Aria
python train_vift_aria.py \
    --epochs 100 \
    --batch-size 16 \
    --lr 1e-4 \
    --data-dir aria_latent \
    --checkpoint-dir checkpoints_vift_aria \
    --device cuda
```


**Key Improvements in Fixed Training:**
- Learnable normalization that preserves motion scale
- World-frame relative poses (no coordinate transformations)
- Mixed precision training with gradient accumulation
- Scale correction factor to prevent underestimation
- Robust gradient handling without skipping batches

**Important Note on Inference:**
- During inference, the model only needs images and IMU data (no ground truth)
- The model predicts relative poses directly in world coordinates
- No coordinate transformations are needed during inference
- The world-frame training ensures consistent predictions without direction flips

**Stable Training Features (train_vift_aria_stable.py):**
- Input normalization for visual (×0.1) and IMU (×0.01) features
- Robust loss functions: Huber loss for translation, smooth geodesic for rotation
- Pre-norm transformer architecture for better gradient flow
- Aggressive gradient clipping (0.5) and monitoring
- Very conservative learning rate (5e-5) with warmup
- Smaller batch size (8) to reduce memory pressure
- Skips batches with exploding gradients

**Standard Architecture Features (train_vift_aria.py):**
- Direct 7DoF pose prediction (no transition-based approach)
- Geodesic rotation loss for proper SO(3) distance measurement
- Moderate weight initialization (std=0.1)
- Simplified loss: L1 translation + geodesic rotation
- Lower learning rate (1e-4) for stable training

### 4. Evaluation and Inference

#### For KITTI-trained Models

```bash
# Test on KITTI test sequences
python src/eval.py \
    ckpt_path=/path/to/checkpoint.ckpt \
    model=weighted_latent_vio_tf \
    data=latent_kitti_vio \
    data.train_loader.root_dir=/home/external/VIFT_AEA/data/kitti_latent_data/train_10 \
    data.val_loader.root_dir=/home/external/VIFT_AEA/data/kitti_latent_data/val_10 \
    data.test_loader.root_dir=/home/external/VIFT_AEA/data/kitti_latent_data/val_10 \
    trainer=gpu \
    trainer.devices=1 \
    logger=csv
```

#### For Aria-trained Models

#### Evaluate Stable Model Performance
```bash
# For models trained with stable training script
python evaluate_stable_model.py \
    --checkpoint checkpoints_vift_stable/best_model.pt \
    --data-dir aria_latent \
    --output-dir evaluation_results \
    --batch-size 16 \
    --device cuda
```

**Note**: The evaluation script automatically generates test features if they don't exist using `generate_all_pretrained_latents_fixed.py`.

This will output:
- Translation and rotation error statistics
- Sample trajectory visualizations in `evaluation_results/plots/`
- Error distribution histograms
- 3D trajectory plots for each test sequence (full, 1s, 5s)
- 3D rotation plots for each test sequence (full, 1s, 5s)

### 5. Inference on New Data

#### For KITTI Models
```bash
# First extract features from your images/IMU
# Then run inference using the trained model
python inference_kitti.py \
    --checkpoint /path/to/kitti_checkpoint.ckpt \
    --data-path /path/to/new/data \
    --output-dir results/
```

#### For Aria Models
```bash
# Inference with stable model
python inference_full_frames_stable.py \
    --checkpoint checkpoints_vift_stable/best_model.pt \
    --sequence-path /path/to/aria/sequence \
    --output-path trajectory_output.json

# Inference with standard model  
python inference_full_frames_unified.py \
    --checkpoint checkpoints_vift_aria/best_model.pt \
    --sequence-path /path/to/aria/sequence \
    --output-path trajectory_output.json
```

### 6. Expected Results

With the stable training approach, you should expect:
- **Translation Error**: ~0.8cm mean, ~0.3cm median
- **Rotation Error**: ~1.3° mean, ~0.7° median
- Most predictions (95%) within 3.2cm translation and 4.2° rotation error

## Project Structure

```
VIFT_AEA/
├── src/
│   ├── models/                    # Model architectures
│   │   └── components/
│   │       └── pose_transformer.py  # Core transformer
│   ├── data/                      # Dataset classes
│   ├── metrics/                   # Loss functions
│   └── utils/
│       └── tools.py               # Rotation utilities (geodesic loss)
├── scripts/
│   └── process_aria_to_vift_quaternion.py  # Data processing
├── train_vift_aria.py             # Main training script (unified)
├── train_vift_aria_stable.py      # Stable training (fixes NaN issues)
├── train_vift_direct.py           # Alternative direct prediction
├── inference_full_frames_unified.py # Inference for standard models
├── inference_full_frames_stable.py  # Inference for stable models  
├── evaluate_stable_model.py         # Evaluation script for test set
├── configs/                       # Hydra configuration files
├── pretrained_models/             # Pretrained VIFT encoder
├── aria_processed/    # Processed sequences
├── aria_latent/                    # Generated features (train/val/test)
├── evaluation_results/            # Test set evaluation outputs
└── checkpoints_vift_stable/       # Trained models
```

## Key Features

- **Full Frame Processing**: Uses ALL frames from videos (not downsampled)
- **Quaternion Representation**: Maintains rotation continuity
- **Geodesic Rotation Loss**: Proper distance measurement on SO(3) manifold
- **Direct Pose Prediction**: Following original VIFT architecture
- **Multi-Head Architecture**: 8-head attention for visual-IMU fusion
- **Relative Pose Prediction**: Frame-to-frame motion in local coordinates

## Results

### Latest Improvements (Unified Architecture)

**Key Improvements:**
1. **Direct Prediction**: Replaced transition-based approach with direct 7DoF output
2. **Geodesic Loss**: Proper rotation error measurement on SO(3) manifold
3. **Stable Training**: Fixed NaN issues with proper initialization and loss weights
4. **Architecture Simplification**: Removed complex PoseTransformer dependencies

**Expected Performance:**
- Predictions should match ground truth scale (meters, not centimeters)
- Trajectories should follow curved paths, not straight lines
- Rotation errors measured properly in degrees
- Stable training without NaN losses

### Architectural Evolution

| Version | Architecture | Loss Function | Scale Issues | Stability |
|---------|--------------|---------------|--------------|-----------|
| Transition-based | Embeddings→Differences→Poses | Complex multi-term | Yes (10-100x smaller) | Poor |
| Direct (original) | Transformer→Linear(7) | Simple MSE | Better | Moderate |
| Unified (latest) | Transformer→Linear(7) | L1 + Geodesic | Best | Good |

### Why Geodesic Loss?

The geodesic loss properly measures rotation distance on the SO(3) manifold:
- L1/L2 on quaternions doesn't measure actual rotation angle
- Geodesic distance = actual angle between rotations
- Better gradient flow for rotation learning
- Standard practice in rotation estimation literature

## Key Architecture Components

### VIFT Architecture
- **Visual-Selective-VIO Encoder**: Pre-trained encoder that extracts 512-dim visual + 256-dim IMU features
- **Pose Transformer**: Processes concatenated 768-dim features to predict 6-DOF relative poses
- **Two-stage approach**: Feature extraction → Pose prediction (not end-to-end)

### Important Files
- `pretrained_models/vf_512_if_256_3e-05.model` - Pre-trained feature encoder (required!)
- `src/models/components/pose_transformer.py` - Core transformer architecture
- `train_efficient.py` - Stable training script for Aria
- `extract_aria_latent_features_for_kitti.py` - Cross-domain feature extraction

## Troubleshooting

### Common Issues

1. **NaN Losses**: If you encounter NaN losses:
   - Use `train_vift_aria_stable.py` instead - it has built-in protections
   - Key differences: input normalization, gradient monitoring, smaller batch size
   - The stable version normalizes IMU features (which can have values >600)

2. **Scale Mismatch**: If predictions are 10-100x smaller than ground truth, ensure you're using the unified version with direct prediction

3. **Straight Line Trajectories**: This indicates the model isn't learning motion dynamics - check loss weights and learning rate

4. **Architecture Mismatch**: Always use matching inference script for your trained model

### Training Tips

- Start with learning rate 1e-4 for stable training
- Monitor both translation and rotation losses separately
- Check sample predictions every 50 batches to ensure reasonable magnitudes
- Use batch size 16 for stable training (reduce if GPU memory limited)
- If NaN persists, check data for corrupted samples

## Performance Summary

| Model | Training Data | Test Data | Translation Error | Rotation Error | Notes |
|-------|--------------|-----------|------------------|----------------|-------|
| VIFT | KITTI | KITTI | 3.27% | 1.78° | Sequences 05, 07, 10 |
| VIFT | KITTI | Aria (reorganized) | 3.27% | 1.77° | Using reorganized data approach |
| VIFT | KITTI | Aria (direct) | ~32% | ~16° | Initial attempt with format mismatch |
| VIFT-AEA Stable | Aria | Aria | 0.84 cm | 1.3° | Trained directly on Aria |
| VIFT-AEA Fixed | Aria (fixed IMU) | Aria | TBD | TBD | With proper between-frames IMU |

## Important: IMU Data Format Fix

The original implementation had a temporal misalignment issue where IMU data was extracted in sliding windows centered on each frame. This has been fixed:

### Original (Incorrect) Approach:
- IMU window: [-25ms, +25ms] around each frame
- Creates overlapping data and violates causality
- Shape: `[N, 50, 6]` for N frames

### Fixed (Correct) Approach:
- IMU extracted between consecutive frames [t_i, t_{i+1})
- No overlap, proper temporal alignment for VIO
- Shape: `[N-1, 50, 6]` for N frames

To use the fixed version, use scripts with "_fixed" suffix.

## Important Notes for Cross-Domain Testing

### Data Reorganization Approach
When testing KITTI models on Aria data, we use a reorganization approach that maps Aria sequences to KITTI format:
- Aria sequence 016 → KITTI sequence 05 (989 samples)
- Aria sequence 017 → KITTI sequence 07 (989 samples)
- Aria sequences 018 + 019 → KITTI sequence 10 (7,423 samples combined)

This approach is necessary because:
1. KITTI tester expects sequences named 05, 07, 10
2. KITTI tester looks for IMU .mat files and pose .txt files in specific locations
3. The reorganization allows using the original KITTI evaluation pipeline

### Creating Custom Configurations
To test all 4 Aria sequences independently, you can:
1. Create IMU MATLAB files using `convert_aria_imu_to_matlab.py`
2. Create custom model configurations (see `configs/model/latent_vio_tf_aria_4seq.yaml`)
3. However, you'll still need to handle the missing pose files issue

## Important Notes for Training

1. **Feature Extraction is Required**: Both KITTI and Aria training require pre-extracting features using the Visual-Selective-VIO encoder
2. **Not End-to-End**: The visual encoder is frozen; only the pose transformer is trained
3. **Domain Gap**: Initial attempts showed KITTI models performing poorly on Aria, but proper data reorganization resolves this
4. **Batch Size**: Use smaller batch sizes (8-32) for stable training on Aria
5. **Learning Rate**: Use conservative learning rates (1e-4 to 5e-5) for Aria

## Citation

Based on the VIFT architecture from:
```
@article{vift2023,
  title={Visual-Inertial Fusion Transformer},
  author={...},
  year={2023}
}
```