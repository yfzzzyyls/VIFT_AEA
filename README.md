# VIFT-AEA: Visual-Inertial Fusion Transformer for Aria Everyday Activities

This project implements a Visual-Inertial Odometry (VIO) system using transformer architecture for the Aria Everyday Activities dataset. The model predicts frame-to-frame relative poses (translation and rotation) by fusing visual and IMU features.

## Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Data Processing Pipeline

#### Process Raw Aria Data (20 sequences with ALL frames)
```bash
# Process 20 diverse sequences from aria_everyday dataset
./process_full_dataset_optimal.sh
```

This script processes 20 sequences from 4 different locations, extracting ALL frames (not downsampled). Output will be in `aria_processed_full_frames/`.

#### Generate Latent Features
```bash
# Generate latent features with stride 10
python generate_all_pretrained_latents_fixed.py \
    --processed-dir aria_processed_full_frames \
    --output-dir aria_latent_full_frames \
    --stride 10 \
    --pose-scale 100.0 \
    --skip-test
```

### 3. Training

```bash
# Train with original VIFT architecture (simple and stable)
# Uses the existing correct implementation, just changes output to 7DoF quaternions
python train_vift_original_simple.py \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-4 \
    --print-freq 20 \
    --data-dir /home/external/VIFT_AEA/aria_latent_full_frames \
    --checkpoint-dir checkpoints_vift_simple \
    --device cuda

# To see all available options:
python train_vift_original_simple.py --help
```

The model uses:
- Original VIFT unified transformer architecture with 7DoF output
- Multi-head attention (8 heads) for visual-IMU fusion
- 6 transformer layers
- Fixed relative motion loss with proper gradient flow
- Quaternion representation for rotations (x, y, z, w)
- AdamW optimizer with OneCycleLR scheduler
- Detailed 7-DoF logging every 20 batches showing:
  - Ground truth translation (cm) and quaternion values
  - Predicted translation (cm) and quaternion values
  - Loss values and rotation variance metrics

**Key Training Improvements:**
- Loss function encourages minimum motion (5cm) to prevent zero predictions
- Soft target for typical human motion (~50cm per frame)
- No conversion to meters in loss computation (maintains gradient scale)
- Diagnostic printing every 20 batches to monitor convergence

### 4. Inference and Visualization

```bash
# Run inference on test sequences and generate 3D plots
python inference_full_frames.py \
    --checkpoint full_frames_checkpoints/[timestamp]/best_model_epoch_X.pt \
    --processed-dir aria_processed_full_frames \
    --output-dir full_frames_results
```

This generates 3D trajectory plots comparing predicted vs ground truth trajectories.

## Project Structure

```
VIFT_AEA/
├── src/
│   ├── models/                    # Model architectures
│   │   └── multihead_vio_separate_fixed.py  # Main model
│   ├── data/                      # Dataset classes
│   └── metrics/                   # Loss functions
├── scripts/
│   └── process_aria_to_vift_quaternion.py  # Data processing
├── configs/                       # Hydra configuration files
├── pretrained_models/             # Pretrained VIFT encoder
├── aria_processed_full_frames/    # Processed sequences
├── aria_latent_full_frames/       # Generated features
└── full_frames_checkpoints/       # Trained models
```

## Key Features

- **Full Frame Processing**: Uses ALL frames from videos (not downsampled)
- **Quaternion Representation**: Maintains rotation continuity
- **Trajectory-Aware Training**: Encourages natural human motion patterns
- **Multi-Head Architecture**: Separate processing for rotation and translation
- **Relative Pose Prediction**: Frame-to-frame motion in local coordinates

## Results

With full frame data and fixed training approach:
- Model successfully learns to predict non-zero motion (was stuck at ~0.1cm)
- Predictions gradually improve from 5-6cm to more realistic values
- Natural curved trajectories instead of straight lines
- Robust to varying sequence lengths (1,000-9,000 frames)

**Training Progress:**
- Initial issue: Model predicted constant near-zero values
- After fix: 10-100x improvement in prediction magnitude
- Model now actively learns motion patterns instead of outputting constants

## Citation

Based on the VIFT architecture from:
```
@article{vift2023,
  title={Visual-Inertial Fusion Transformer},
  author={...},
  year={2023}
}
```