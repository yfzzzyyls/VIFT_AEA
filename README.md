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

## 🚨 Current Status

### Key Issue Resolved: Data Processing Bug
We discovered and fixed a critical bug in the data processing pipeline:
- The trajectory was being downsampled incorrectly from 1000Hz to 20Hz
- **Bug**: Taking consecutive frames instead of every 50th frame
- **Result**: Almost no motion between frames (0.001 seconds apart instead of 0.05 seconds)
- **Fix**: Proper downsampling in `scripts/process_aria_to_vift_quaternion.py`

### MultiHead Model Training Results
The MultiHead model with quaternion support failed to train properly:
- Training diverged with NaN weights after epoch 77
- Even early checkpoints show poor performance:
  - Translation ATE: 282.45 cm (with alignment: 181.07 cm)
  - Rotation error: 96.17° mean
  - Only 4% of frames have <2cm translation error

### Coordinate System Issues
Umeyama alignment analysis revealed:
- **Scale mismatch**: 3.37x (predictions too large)
- **Coordinate rotation**: 32° offset between frames
- **Translation offset**: 1253m initial position error
- **Y-axis inverted**: Strong negative correlation (-0.723)

### Next Steps
1. Train VIFT original architecture on corrected dataset
2. Use conservative hyperparameters for stability
3. Monitor coordinate system consistency

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

### 🔍 Troubleshooting

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
