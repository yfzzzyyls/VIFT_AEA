# FlowNet-LSTM-Transformer Architecture v1.0

A new visual-inertial odometry architecture that replaces VIFT with:
- **FlowNet** for motion-specific visual feature extraction
- **LSTM** for processing variable-length IMU sequences
- **Transformer** for multi-modal fusion and pose prediction

## Key Features

1. **No Fixed Window Size**: Process sequences of any length (5-50 frames)
2. **All IMU Data**: Uses all ~50 IMU samples between frames (Aria: 1000Hz IMU / 20Hz camera)
3. **All Frames**: Processes ALL frames from sequences (no subsampling) for best temporal resolution
4. **End-to-End Learning**: All components trained jointly on Aria dataset
5. **Step-wise Curriculum**: Gradually increases sequence length every 10 epochs
6. **Flexible Resolution**: Default 704×704 (2×2 binned from 1408×1408)

## Architecture Overview

```
Images[t] → FlowNet → Motion Features
    ↓                      ↓
IMU[t→t+1] → LSTM → Temporal Features → Transformer → Poses[t→t+1]
```

## Quick Start

### 1. Process Aria Data (if not already done)
```bash
# Default: Process with 704×704 resolution (2×2 binned from 1408×1408)
cd /home/external/VIFT_AEA
python process_aria.py \
    --input-dir /mnt/ssd_ext/incSeg-data/aria_everyday \
    --output-dir aria_processed \
    --max-frames 1000

# Note: The script automatically uses 2×2 binning for better quality
# First bins 1408×1408 → 704×704, then resizes to final resolution
```

### 2. Training Commands

#### Multi-GPU Training with 4× A6000 (Memory-Optimized)
```bash
cd /home/external/VIFT_AEA/new_architecture

# Latest optimized training with shared memory (uses all 4 GPUs efficiently)
./run_training_shared.sh

# This runs:
python train_flownet_lstm_transformer_shared.py \
    --distributed \
    --use-amp \
    --data-dir ../aria_processed \
    --encoder-type resnet18 \
    --pretrained \
    --warmup-epochs 2 \
    --batch-size 64 \
    --num-workers 0 \
    --learning-rate 4.8e-3 \
    --num-epochs 100 \
    --sequence-length 41 \
    --stride 5 \
    --translation-weight 1.0 \
    --rotation-weight 100.0 \
    --scale-weight 10.0 \
    --checkpoint-dir ../checkpoints_from_scratch \
    --experiment-name resnet_lstm_41frames_mmap_shared \
    --validate-every 1

# Key features:
# - Uses memory-mapped shared data (89GB total, not 4×89GB)
# - ResNet18 encoder (faster than FlowNet, similar performance)
# - Large batch size (64 per GPU = 256 total)
```

#### Monitor GPU Usage
```bash
# In another terminal, monitor GPU memory and utilization
python monitor_gpu_snapshot.py --interval 2
```

#### Alternative: 512×512 Resolution (1.9× faster, slightly lower quality)
```bash
torchrun --nproc_per_node=4 train_flownet_lstm_transformer.py \
    --distributed \
    --use-amp \
    --variable-length \
    --data-dir ../aria_processed \
    --batch-size 8 \
    --num-workers 4 \
    --image-height 512 \
    --image-width 512 \
    --learning-rate 4e-4 \
    --num-epochs 100 \
    --min-seq-len 10 \
    --max-seq-len 50 \
    --curriculum-step 10 \
    --curriculum-increment 5 \
    --stride 5 \
    --use-curriculum \
    --checkpoint-dir checkpoints/exp_512_4gpu \
    --experiment-name flownet_lstm_512
```

### 3. Monitor Training
```bash
# In another terminal
watch -n 1 nvidia-smi

# Check training progress
tail -f checkpoints/exp_512_4gpu/train.log
```

## Key Training Arguments

- `--data-dir`: Path to processed Aria data (use `../aria_processed` from new_architecture)
- `--encoder-type`: Visual encoder type (`resnet18` recommended, `flownet` for original)
- `--batch-size`: Batch size per GPU (64 with memory-mapped data on A6000)
- `--num-workers`: DataLoader workers per GPU (0 for shared memory to prevent duplication)
- `--learning-rate`: Base LR (4.8e-3 for effective batch size 256)
- `--warmup-epochs`: Learning rate warmup epochs (2 recommended for large batch)
- `--sequence-length`: Fixed sequence length (41 frames = 2.05s for AR/VR)
- `--stride`: Sliding window stride (default: 5)
- `--translation-weight`: Loss weight for translation (1.0)
- `--rotation-weight`: Loss weight for rotation (100.0 for quaternion geodesic)
- `--scale-weight`: Loss weight for scale consistency (10.0)

## Model Configuration

The model has three main components:

### 1. FlowNet Motion Encoder
- Processes consecutive image pairs to extract motion
- Uses correlation layer to compute pixel correspondences
- Output: 256-dimensional motion features

### 2. IMU LSTM Encoder
- Processes ALL IMU samples between frames (~50 per interval)
- Bidirectional LSTM with 3 layers
- Handles variable-length sequences naturally
- Output: 256-dimensional temporal features

### 3. Pose Transformer
- Fuses visual and IMU features
- 6-layer transformer with causal attention
- Predicts relative poses (3 translation + 4 quaternion)

## Dataset

The `AriaVariableIMUDataset` supports:
- Variable sequence lengths
- All raw IMU data (~50 samples per frame interval)
- Proper temporal alignment
- Efficient batching with custom collate function

## Loss Functions

- **Translation Loss**: MSE on predicted translations
- **Rotation Loss**: Geodesic distance for quaternions
- **Scale Consistency**: Prevents scale drift
- **Temporal Smoothness**: Encourages smooth trajectories

## Evaluation Metrics

- **ATE**: Absolute Trajectory Error
- **RPE**: Relative Pose Error (at different scales)
- **Scale Error**: Trajectory scale accuracy
- **Drift Rate**: End-to-end drift percentage

## File Structure

```
new_architecture/
├── models/
│   └── flownet_lstm_transformer.py    # Main model architecture
├── data/
│   └── aria_variable_imu_dataset.py   # Dataset with variable IMU
├── configs/
│   └── flownet_lstm_transformer_config.py  # Configuration
├── utils/
│   ├── losses.py                       # Loss functions
│   ├── metrics.py                      # Evaluation metrics
│   └── visualization.py                # Plotting utilities
├── train_flownet_lstm_transformer.py   # Training script
└── README.md                           # This file
```

## Resolution Comparison

| Resolution | Memory/GPU | Training Speed | Quality | Recommendation |
|------------|------------|----------------|---------|----------------|
| 512×512    | ~28GB      | 1.9× faster    | Good    | ✅ For experiments |
| 704×704    | ~45GB      | 1.0× (baseline)| Better  | ✅ **Default** |
| 1408×1408  | OOM        | N/A            | Best    | ❌ Too large |

**Why 704×704 as default?**
- Aria's native resolution is 1408×1408
- 704×704 = 2×2 binning (preserves detail better than resizing to 512)
- Best quality/speed tradeoff for production models
- Only 1.9× slower than 512×512 but noticeably better quality

## Curriculum Learning Schedule (optional)

```
Epochs   1-10:  11 frames (0.55s)  ← Start (matches original VIFT)
Epochs  11-20:  15 frames (0.75s)
Epochs  21-30:  19 frames (0.95s)
Epochs  31-40:  23 frames (1.15s)
Epochs  41-50:  27 frames (1.35s)  ← AR/VR typical
Epochs  51-60:  31 frames (1.55s)
Epochs  61-70:  35 frames (1.75s)
Epochs  71-80:  39 frames (1.95s)
Epochs  81-90:  41 frames (2.05s)  ← Maximum
```

## Evaluation

```bash
# Evaluate on test set (704×704 model)
python evaluate_flownet_lstm_transformer.py \
    --checkpoint checkpoints/exp_final_4gpu/best_model.pt \
    --data-dir ../aria_processed \
    --output-dir evaluation/exp_final_4gpu \
    --batch-size 6 \
    --sequence-length 31 \
    --image-height 704 \
    --image-width 704
```

## Advantages Over VIFT

1. **No Information Loss**: Uses all ~50 IMU samples per interval, not just 11
2. **Flexible Sequences**: Can process any length, not fixed to 11 frames
3. **Better Motion Features**: FlowNet explicitly models optical flow
4. **Temporal Modeling**: LSTM captures IMU dynamics better than CNN
5. **Attention Mechanism**: Transformer learns when to trust each modality

## Complete Workflow

### Step 1: Data Preparation

```bash
# 1.1 Process Aria VRS files to extract images, IMU, and poses
cd /home/external/VIFT_AEA

# Process ALL frames for best quality (no subsampling)
python process_aria.py \
    --input-dir /mnt/ssd_ext/incSeg-data/aria_everyday \
    --output-dir aria_processed \
    --num-workers 4

# Note: This processes ALL frames from each sequence
# - Ensures ~50 IMU samples between consecutive frames (20Hz camera, 1000Hz IMU)
# - Processing time: ~5-10 minutes per sequence depending on length
# - Total dataset size: ~50-100GB for 20 sequences with all frames

# Expected output structure:
# aria_processed/
# ├── 000/
# │   ├── visual_data.pt      # [N, 3, 704, 704] ALL frames
# │   ├── imu_data.pt         # List of N-1 variable-length IMU tensors (~50 samples each)
# │   └── poses_quaternion.json
# ├── 001/
# ├── ...
# ├── 019/
# └── splits.json             # train/val/test splits
```

### Step 2: Training

```bash
# 2.1 Navigate to new architecture directory
cd /home/external/VIFT_AEA/new_architecture

# 2.2 Start distributed training with 4 GPUs (memory-optimized)
./run_training_shared.sh

# Monitor training progress
# Terminal 1: Watch detailed GPU usage
python monitor_gpu_snapshot.py --interval 2

# Terminal 2: Monitor training output
# (Training logs appear in console, not separate file)

# Training will save:
# - Best model: checkpoints/exp_final_4gpu/best_model.pt
# - Periodic checkpoints: checkpoint_epoch_5.pt, epoch_10.pt, etc.
# - Training metrics: checkpoints/exp_final_4gpu/metrics.json
```

### Step 3: Evaluation

```bash
# 3.1 Evaluate on test set
python evaluate_flownet_lstm_transformer.py \
    --checkpoint checkpoints/exp_final_4gpu/best_model.pt \
    --data-dir ../aria_processed \
    --output-dir evaluation/exp_final_4gpu \
    --split test \
    --batch-size 6 \
    --sequence-length 31

# Outputs:
# - evaluation/exp_final_4gpu/metrics.json        # Quantitative metrics
# - evaluation/exp_final_4gpu/visualizations/     # Trajectory plots
# - evaluation/exp_final_4gpu/detailed_results.npz # Raw predictions
```

- [ ] Add uncertainty estimation
- [ ] Implement online/streaming mode
- [ ] Add self-supervised losses
- [ ] Support for other datasets (TUM-VI, EuRoC)
- [ ] Lightweight version for edge deployment