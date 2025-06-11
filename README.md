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

A state-of-the-art Visual-Inertial Odometry (VIO) system for the Aria Everyday Activities dataset, implementing and improving upon the VIFT architecture with multiple model variants and AR/VR Adaptations.

## 🚨 Recent Updates (Jan 2025)

### Critical Issues Resolved
1. **Data Representation Bug**: Fixed issue where training data contained absolute poses instead of frame-to-frame relative poses
2. **Subsampling Issue**: Discovered pretrained latent data contains subsampled frames (~40cm apart) rather than consecutive frames
3. **Scale Mismatch**: Fixed meter/centimeter conversion issues in data pipeline
4. **Loss Function**: Balanced rotation and translation losses for stable training

### Latest Results
Training with relative poses shows improved performance:
- **Average ATE**: 0.71m (down from >100m)
- **Frame-to-frame error**: 6.15cm (down from >25cm) 
- **Target**: 1-5cm (still improving)

| Test Seq | ATE (m) | RPE@1f (cm) | RPE@1s (m) |
|----------|---------|-------------|------------|
| 016      | 0.24    | 7.41        | 1.02       |
| 017      | 0.12    | 3.20        | 0.39       |
| 018      | 0.15    | 6.03        | 0.81       |
| 019      | 1.07    | 5.18        | 0.78       |

## 🚀 Quick Reproduction Guide

To reproduce the results shown above, follow these exact commands:

### Prerequisites
- You need the preprocessed Aria data in `data/aria_processed/` (see Data Preparation section)
- You need the pretrained latent features in `data/aria_latent_data_pretrained/` 

If you don't have these, first follow the Data Preparation steps below.

### Reproduction Steps

```bash
# 1. Convert absolute poses to relative poses (if not already done)
python convert_to_relative_poses.py

# 2. Train the model (data is already in centimeters in aria_latent_data_cm/)
python train_improved.py \
    --data-dir aria_latent_data_cm \
    --batch-size 32 \
    --num-epochs 50 \
    --learning-rate 1e-4 \
    --log-every-n-steps 5 \
    --gradient-clip-val 1.0

# 3. Run inference on all test sequences
for seq in 016 017 018 019; do
    python inference_full_sequence.py \
        --sequence-id $seq \
        --checkpoint fixed_scale_v1/epoch_epoch=024_val_loss_val/total_loss=14.812485.ckpt \
        --processed-dir data/aria_processed \
        --stride 1 \
        --batch-size 64
done

# 4. Generate visualization plots for a sequence
python plot_short_term_trajectory.py \
    --npz-file inference_results_realtime_seq_016_stride_1.npz \
    --output-dir short_term_plots_seq016 \
    --duration 5

# 5. View results
ls short_term_plots_seq016/  # Contains 3D trajectory and rotation plots
```

### ⚠️ Known Issues with Current Model

The current model still shows straight-line predictions instead of natural curves. This is due to:
1. **Systematic bias**: ~16.5 cm/frame constant prediction
2. **Training data issues**: Very small motions (0.11 cm/frame) from 30-frame subsampling
3. **Mode collapse**: Model outputs nearly constant values

### 🔧 Recommended Fix

To properly fix the straight-line issue, retrain with bias correction:

```bash
# Train with bias-aware loss function (fixes straight-line predictions)
python train_fixed_bias.py \
    --data-dir aria_latent_data_cm \
    --batch-size 32 \
    --num-epochs 50 \
    --learning-rate 5e-5 \
    --rotation-weight 100.0 \
    --bias-weight 0.1 \
    --variance-weight 0.1 \
    --gradient-clip-val 1.0
```

This uses an improved loss that penalizes bias and encourages prediction diversity.
# 1. Setup Environment
source ~/venv/py39/bin/activate  # Or your Python 3.9 environment
pip install -r requirements.txt

# 2. Download Pretrained Encoder (if not already done)
python download_pretrained_model.py

# 3. Convert Absolute Poses to Relative Poses
python convert_to_relative_poses.py
# Input: data/aria_latent_data_pretrained/
# Output: data/aria_latent_data_relative/

# 4. Train VIFT Quaternion Model with Relative Poses
python train_improved.py \
    --model vift_quaternion \
    --data-dir data/aria_latent_data_relative \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-4 \
    --hidden-dim 128 \
    --dropout 0.1 \
    --rotation-weight 1.0 \
    --translation-weight 1.0 \
    --checkpoint-dir checkpoints_relative_v2 \
    --experiment-name relative_poses_v2 \
    --log-every-n-steps 5 \
    --optimizer adamw \
    --scheduler cosine \
    --gradient-accumulation 2

# Monitor training progress (in another terminal):
# tensorboard --logdir logs/
# Or check the training log:
# tail -f training_relative_v2.log

# 5. Run Inference on Test Sequences
for seq in 016 017 018 019; do
    python inference_full_sequence.py \
        --sequence-id $seq \
        --checkpoint checkpoints_relative_v2/epoch_epoch=021_val_loss_val/total_loss=12.422941.ckpt \
        --stride 20 \
        --no-plots
done

# 6. Generate Summary Results
python summarize_test_results.py

# 7. (Optional) Visualize Short-Term Trajectory
python plot_short_term_trajectory.py \
    --npz-file inference_results_realtime_seq_016_stride_20.npz \
    --output-dir short_term_plots_relative_v2
```

**Note**: The checkpoint path in step 5 should be updated to your best checkpoint after training completes.

## 📋 Requirements

- Python 3.9+
- CUDA-capable GPU (48GB+ VRAM recommended for processing, 8GB+ for training)
- 32GB+ RAM
- ~3TB disk space for full dataset (500GB for Aria raw data, 2TB for processed data)

## 🔧 Installation

```bash
# Clone repository
git clone https://github.com/yfzzzyyls/incremental-segmentation.git

# Create virtual environment
# python3.9 -m venv venv
source ~/venv/py39/bin/activate  # On Windows: venv\py39\Scripts\activate

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

# Parallel processing (10x faster) - Recommended for full dataset
# Uses 10 CPU workers for consistent performance
./process_full_dataset_optimal.sh

# Monitor progress while running:
watch -n 5 'ls -1 data/aria_full/processed | wc -l'  # Shows completed sequences
tail -f data/aria_full/logs/worker_*.log              # View individual worker logs
htop                                                   # Monitor CPU usage

# The processing includes:
# - Extracting SLAM trajectories from MPS results
# - Converting to quaternion format (no Euler angles)
# - Extracting RGB frames
# - Generating IMU data at proper frequency
# - Creates numbered folders (000, 001, 002, etc.)
# - Worker distribution: Workers 1-9 process 14 sequences each, Worker 10 processes 17
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
    --stride 10 \
    --pose-scale 100.0

# This will:
# - Extract 512-dim visual features using pretrained encoder
# - Extract 256-dim IMU features (in m/s²)
# - Convert poses to relative transformations in local coordinates (in meters)
# - Create sliding windows of 11 frames (10 transitions)
# - Generate ~371K training samples from 137 sequences
```

## 🏃 Training

#### MultiHead Fixed

Uses geodesic loss for rotation with proper weight initialization:

```bash
# Optimized configuration with 4 GPUs
python train_improved.py \
    --model multihead_fixed \
    --data-dir /path/to/aria_latent_data_pretrained \
    --epochs 50 \
    --optimizer adamw \
    --scheduler onecycle \
    --lr 2e-3 \
    --gradient-accumulation 4 \
    --checkpoint-dir experiment_name \
    --experiment-name my_experiment
```

```bash
./train_vift_original_fixed.sh
```

**Note**: The model trains and predicts in meters, maintaining consistency with the original data and IMU units (m/s²).

#### VIFT Original

```bash
python train_improved.py --model vift_original \
    --data-dir /path/to/aria_latent_data_pretrained \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --hidden-dim 128 \
    --num-heads 8 \
    --dropout 0.1
```

### Configuration Options

Key parameters for training:

- `--batch-size`: Batch size per GPU (default: 32, use 16 for limited GPU memory)
- `--lr`: Learning rate (default: 1e-3)
- `--epochs`: Number of training epochs (default: 100)
- `--hidden-dim`: Hidden dimension size (default: 128)
- `--num-heads`: Number of attention heads (default: 4)
- `--dropout`: Dropout rate (default: 0.2)
- `--gradient-clip`: Gradient clipping value (default: 1.0)
- `--gradient-accumulation`: Gradient accumulation steps (default: 4)
- `--optimizer`: Optimizer choice: adamw, adam, sgd (default: adamw)
- `--scheduler`: Learning rate scheduler: onecycle, cosine, none (default: onecycle)
- `--checkpoint-dir`: Directory to save checkpoints (default: logs/checkpoints_{model})
- `--experiment-name`: Name for the experiment (for logging)

**Multi-GPU Training**: The script automatically uses all available GPUs (default: 4) with DDP strategy.

Monitor training:

```bash
tensorboard --logdir logs/
```

## 📈 Evaluation

Evaluate trained models on test sequences:

```bash
# Evaluate on all test sequences
python inference_full_sequence.py \
    --sequence-id all \
    --checkpoint /path/to/checkpoint/last.ckpt \
    --processed-dir /path/to/aria_processed \
    --stride 1  # Real-time sliding window

# Single sequence evaluation
python inference_full_sequence.py \
    --sequence-id 123 \
    --checkpoint /path/to/checkpoint/last.ckpt \
    --processed-dir /path/to/aria_processed \
    --stride 1

# Faster evaluation with larger stride
python inference_full_sequence.py \
    --sequence-id all \
    --checkpoint /path/to/checkpoint/last.ckpt \
    --processed-dir /path/to/aria_processed \
    --stride 10  # Process every 10th frame

# Disable visualization plots (faster)
python inference_full_sequence.py \
    --sequence-id all \
    --checkpoint /path/to/checkpoint/last.ckpt \
    --processed-dir /path/to/aria_processed \
    --stride 1 \
    --no-plots
```

**Visualization**:
```bash
python organize_plots_by_sequence.py \
      --results-dir inference_results_realtime_all_stride_1 \
      --output-dir inference_results_realtime_all_stride_1/sequence_plots
```

```bash
python plot_single_sequence.py --npz-file
      ./inference_results_realtime_seq_123_stride_1.npz --output-dir
      trajectory_plots_v3
```

```bash
python plot_short_term_trajectory.py       --npz-file ./inference_results_realtime_seq_008_stride_10.npz       --output-dir short_term_plots_seq008_fixed       --duration 5
```

**Output**:

- Metrics displayed in meters
- Trajectory plots saved to `trajectory_plots_all_stride_{stride}/`
- Results JSON saved to `inference_results_realtime_averaged_stride_{stride}.json`
- Individual trajectories saved to `inference_results_realtime_all_stride_{stride}/`

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

## 🔍 Troubleshooting

### Common Issues and Solutions

#### Model Predicting Straight Lines
- **Cause**: Training on absolute poses instead of relative poses
- **Solution**: Use `convert_to_relative_poses.py` before training
- **Verify**: Check that poses in `data/aria_latent_data_relative/train/0_gt.npy` have first frame as [0,0,0,0,0,0,1]

#### High Training Loss (>50)
- **Cause**: Data scale mismatch or wrong pose representation
- **Solution**: Ensure using relative poses and correct scale (centimeters)
- **Expected**: Loss should start around 12-15 and decrease to <10

#### Training Metrics Not Logged
- **Cause**: `log_every_n_steps` too high for small batch sizes
- **Solution**: Use `--log-every-n-steps 5` (or lower)
- **Note**: With 588 train samples and batch size 8, there are only 18 steps per epoch

#### Out of Memory
- For processing: Use CPU mode in `fix_failed_sequences.py`
- For training: Reduce batch size: `--batch-size 4`
- Use gradient accumulation: `--gradient-accumulation 4`
- Process sequences individually if needed

#### Rotation Predictions Stuck
- **Cause**: Wrong model architecture or initialization
- **Solution**: Use `vift_quaternion` model with proper initialization
- **Check**: Model should output quaternions without ReLU activation

#### Poor Inference Performance
- **Cause**: Model trained on wrong data or scale mismatch
- **Target**: Frame-to-frame error should be 1-5cm
- **Current**: ~6cm (still improving with better data)

#### Processing Failures
- Some sequences may fail with CUDA OOM
- Run `fix_failed_sequences.py` to reprocess with CPU
- Check logs in data processing directory

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

## 🔧 Key Fixes and Improvements

### Data Processing Fixes
1. **Relative Pose Conversion** (`convert_to_relative_poses.py`)
   - Converts absolute trajectory poses to frame-to-frame relative poses
   - Handles quaternion transformations correctly
   - Scales translations to centimeters for training

2. **Scale Consistency** 
   - Fixed meter/centimeter conversions in `AriaLatentDataset`
   - Ensured consistent scaling across training and inference
   - Model now trains and predicts in centimeters

3. **Loss Function Improvements**
   - Balanced rotation and translation losses (equal weights)
   - Removed problematic log scaling
   - Fixed double-weighting bug in loss computation

4. **Model Initialization**
   - Improved weight initialization (gain=1.0 instead of 0.01)
   - Fixed bias initialization for quaternions [0,0,0,0,0,0,1]
   - Prevents model from getting stuck at zero predictions

### Training Configuration
- **Reduced learning rate**: 1e-4 (more stable than 1e-3)
- **Smaller batch size**: 8 (reduces memory usage)
- **Gradient accumulation**: 2 steps (effective batch size of 16)
- **Cosine scheduler**: Smooth learning rate decay
- **Log every 5 steps**: Ensures metrics are logged even with small batches

## 💡 Quick Start Example

Here's a complete example workflow for testing with 10 sequences:

```bash
# 1. Create a small test dataset
mkdir -p /path/to/small_aria_processed
for i in {000..009}; do
    cp -r /mnt/ssd_ext/incSeg-data/aria_processed/$i /path/to/small_aria_processed/
done

# 2. Generate features
python generate_all_pretrained_latents_fixed.py \
    --processed-dir /path/to/small_aria_processed \
    --output-dir /path/to/small_dataset_10seq \
    --stride 1

# 3. Train model
python train_improved.py \
    --model multihead_fixed \
    --data-dir /path/to/small_dataset_10seq \
    --epochs 50 \
    --optimizer adamw \
    --scheduler onecycle \
    --lr 2e-3 \
    --gradient-accumulation 4 \
    --checkpoint-dir test_experiment \
    --experiment-name test_run

# 4. Evaluate
python inference_full_sequence.py \
    --sequence-id all \
    --checkpoint test_experiment/last.ckpt \
    --processed-dir /path/to/small_aria_processed \
    --stride 1
```

## 🙏 Acknowledgments

- Original VIFT implementation by Yunus Bilge Kurt et al.
- Visual-Selective-VIO pretrained encoder
- Aria Everyday Activities dataset by Meta

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
