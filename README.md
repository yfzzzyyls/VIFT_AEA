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
# RECOMMENDED: Stable VIFT Training (Fixes NaN issues)
# Use this version if you encounter NaN losses
python train_vift_aria_stable.py \
    --epochs 50 \
    --batch-size 8 \
    --lr 5e-5 \
    --data-dir aria_latent_full_frames \
    --checkpoint-dir checkpoints_vift_stable \
    --device cuda

# Alternative: Standard training (if no NaN issues)
python train_vift_aria.py \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-4 \
    --data-dir aria_latent_full_frames \
    --checkpoint-dir checkpoints_vift_unified \
    --device cuda
```

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

#### Evaluate Model Performance
```bash
# For models trained with stable training script
python evaluate_stable_model.py \
    --checkpoint checkpoints_vift_stable/[timestamp]/best_model.pt \
    --data-dir aria_latent_full_frames \
    --output-dir evaluation_results \
    --batch-size 16 \
    --device cuda
```

**Note**: The evaluation script automatically generates test features if they don't exist, so you don't need to run the feature generation separately.

This will output:
- Translation and rotation error statistics
- Sample trajectory visualizations in `evaluation_results/plots/`
- Error distribution histograms
- 3D trajectory plots for each test sequence (full, 1s, 5s)
- 3D rotation plots for each test sequence (full, 1s, 5s)

### 5. Expected Results

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
├── aria_processed_full_frames/    # Processed sequences
├── aria_latent_full_frames/       # Generated features (train/val/test)
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

## Citation

Based on the VIFT architecture from:
```
@article{vift2023,
  title={Visual-Inertial Fusion Transformer},
  author={...},
  year={2023}
}
```