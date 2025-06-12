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
# Train with transition-based VIFT architecture (recommended)
# Key difference: Model outputs embeddings, transitions are computed as differences
python train_vift_aria.py \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-4 \
    --print-freq 20 \
    --data-dir /home/external/VIFT_AEA/aria_latent_full_frames \
    --checkpoint-dir checkpoints_vift_aria \
    --device cuda

# To see all available options:
python train_vift_aria.py --help
```

**Transition-based Architecture (train_vift_aria.py):**
- Model outputs embeddings, not poses directly
- Transitions computed as differences between consecutive embeddings
- Transitions projected to 7DoF pose space
- Prevents constant predictions through architectural bias
- Enhanced diversity loss weight (0.2) and embedding regularization

**Original Architecture (train_vift_original_simple.py):**
- Direct 7DoF pose prediction
- Multi-head attention (8 heads) for visual-IMU fusion
- 6 transformer layers
- Causal masking for temporal modeling
- Quaternion normalization

**Shared Features:**
- Fixed relative motion loss with proper gradient flow
- AdamW optimizer with cosine annealing scheduler
- Detailed logging every 20 batches showing:
  - Ground truth and predicted translations (cm)
  - Quaternion values and variance metrics
  - Individual loss components
  - Embedding statistics (transition-based only)

### 4. Inference and Visualization

```bash
# IMPORTANT: Use the fixed inference script for models trained with train_vift_aria.py
# This script uses the correct VIFTTransition architecture matching the training
python inference_full_frames_fixed.py \
    --checkpoint checkpoints_vift_aria_fixed_final/[timestamp]/best_model.pt \
    --processed-dir aria_processed_full_frames \
    --output-dir full_frames_results_fixed
```

This generates:
- 3D trajectory plots comparing predicted vs ground truth trajectories
- Relative pose analysis plots showing frame-by-frame predictions
- NPZ files with raw predictions for further analysis

**Note on Model Architecture Mismatch:** 
The original `inference_full_frames.py` uses a different model class than `train_vift_aria.py`, causing architecture mismatch errors. Always use `inference_full_frames_fixed.py` for models trained with the transition-based architecture.

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
├── train_vift_aria.py             # Main training script (transition-based)
├── train_vift_original_simple.py  # Original architecture training
├── inference_full_frames.py       # Original inference (architecture mismatch)
├── inference_full_frames_fixed.py # Fixed inference for transition models
├── configs/                       # Hydra configuration files
├── pretrained_models/             # Pretrained VIFT encoder
├── aria_processed_full_frames/    # Processed sequences
├── aria_latent_full_frames/       # Generated features
├── full_frames_results_fixed/     # Inference outputs
└── checkpoints_vift_aria_fixed_final/  # Trained models
```

## Key Features

- **Full Frame Processing**: Uses ALL frames from videos (not downsampled)
- **Quaternion Representation**: Maintains rotation continuity
- **Trajectory-Aware Training**: Encourages natural human motion patterns
- **Multi-Head Architecture**: Separate processing for rotation and translation
- **Relative Pose Prediction**: Frame-to-frame motion in local coordinates

## Results

### Recent Improvements (Fixed Architecture)

**Key Issues Resolved:**
1. **Model Architecture Mismatch**: Fixed inference script now uses the same `VIFTTransition` class as training
2. **Layer Normalization Bug**: Removed layer norm on transitions that was destroying motion magnitude
3. **Loss Function Issues**: Fixed temporal and transition losses that were preventing learning

**Current Performance:**
- Model now produces varying predictions instead of constant values
- Predictions show reasonable motion magnitudes (0.1-2cm per frame)
- Different sequences produce different trajectories (not all the same)
- Embeddings show temporal variation (std ~0.05-0.18)

**Remaining Challenges:**
- Predictions still form relatively straight lines instead of following curved paths
- Prediction magnitude is smaller than ground truth (10-50cm total vs 3000-4700cm)
- Model needs architectural improvements to better capture motion dynamics

### Training Progress History

**Phase 1 - Initial Issues:**
- Model predicted constant near-zero values (~0.001cm)
- All sequences produced identical straight-line trajectories
- NaN losses during training

**Phase 2 - After Fixes:**
- 10-100x improvement in prediction magnitude
- Model outputs varying predictions based on input
- Stable training without NaN losses
- Predictions still need improvement to match ground truth curves

## Citation

Based on the VIFT architecture from:
```
@article{vift2023,
  title={Visual-Inertial Fusion Transformer},
  author={...},
  year={2023}
}
```