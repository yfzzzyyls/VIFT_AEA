---

<div align="center">

# VIFT-AEA: Visual-Inertial Fusion Transformer for Aria Everyday Activities

[![Paper](http://img.shields.io/badge/paper-arxiv.2409.08769-B31B1B.svg)](https://arxiv.org/abs/2409.08769)

## Description

This repository implements Visual-Inertial Fusion Transformer (VIFT) for both KITTI and Aria Everyday Activities datasets. VIFT uses a causal transformer architecture for fusion and pose estimation in deep visual-inertial odometry.


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

## Training Modes

### DataParallel vs DistributedDataParallel

This repository supports two multi-GPU training modes:

1. **DataParallel (Recommended for memory-constrained systems)**

   - Single process loads dataset once in RAM
   - Dataset shared across all GPUs
   - Memory usage: ~317GB for Aria dataset
   - Speed: ~70-80% of DDP
   - Usage: `python train_aria_from_scratch.py --use-dataparallel`
2. **DistributedDataParallel (Faster but memory-intensive)**

   - Each GPU process loads its own dataset copy
   - Memory usage: 317GB × N GPUs (e.g., 1.3TB for 4 GPUs)
   - Speed: ~2x faster than DataParallel
   - Usage: `torchrun --nproc_per_node=4 train_aria_from_scratch.py --distributed`

For systems with <1TB RAM, use DataParallel. For systems with abundant RAM, use DDP for faster training.


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
python organize_data_splits.py --data-dir aria_processed --train-ratio 0.7 --val-ratio 0.15

# 4. Train all components from scratch (improved loss weighting)

# Multi-GPU training with DataParallel (memory-efficient)
python train_aria_from_scratch.py \
    --use-dataparallel \
    --data-dir aria_processed \
    --epochs 50 \
    --batch-size 16 \
    --checkpoint-dir checkpoints_dataparallel

# 5. Monitor training progress
tail -f train.log

# 6. Evaluate trained model
python evaluate_from_scratch.py \
    --checkpoint checkpoints_dataparallel/best_model.pt \
    --data-dir aria_processed \
    --output-dir evaluation_from_scratch \
    --batch-size 16 \
    --num-workers 4
```

### Workflow 5: Train with SEA-RAFT Visual Encoder (Advanced Motion Features)

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

# Process Aria data with proper IMU alignment
python process_aria.py

# Create train/val/test splits
python organize_data_splits.py --data-dir aria_processed --train-ratio 0.7 --val-ratio 0.15

# Re-run setup to confirm weights are detected
python setup_searaft.py

# 3. Test SEA-RAFT integration (verify weights loaded correctly)
python test_searaft_integration.py

# 5. Train with SEA-RAFT encoder (multi-GPU with DataParallel)
python train_aria_from_scratch.py \
    --use-dataparallel \
    --data-dir aria_processed \
    --epochs 50 \
    --batch-size 64 \
    --checkpoint-dir checkpoints_searaft \
    --use-searaft

# 6. Compare CNN vs SEA-RAFT performance
# Train baseline CNN model first
python train_aria_from_scratch.py \
    --use-dataparallel \
    --data-dir aria_processed \
    --epochs 50 \
    --batch-size 16 \
    --checkpoint-dir checkpoints_cnn_baseline

# Evaluate both models (auto-detects encoder type)
# For CNN model:
python evaluate_from_scratch.py \
    --checkpoint checkpoints_cnn_baseline/best_model.pt \
    --data-dir aria_processed \
    --output-dir evaluation_cnn \
    --batch-size 16 \
    --num-workers 4

# For SEA-RAFT model:
python evaluate_from_scratch.py \
    --checkpoint checkpoints_searaft/best_model.pt \
    --use-searaft \
    --data-dir aria_processed \
    --output-dir evaluation_searaft \
    --batch-size 16 \
    --num-workers 4

```

**SEA-RAFT Notes:**

- **REQUIRES pretrained weights** - was trained for weeks on optical flow datasets
- Without pretrained weights, it's just a random CNN and won't work properly
- ~4.3x slower than CNN encoder but expected to improve accuracy by 10-15%
- Requires ~2GB additional GPU memory for feature extraction
- Uses frozen pretrained features from optical flow task
- Preserves spatial tokens (4×4) for better parallax reasoning


## License

This project is licensed under the MIT License.
