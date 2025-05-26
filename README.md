---

<div align="center">

# VIFT: A Video Is Worth a Thousand Images for Visual Inertial Odometry - AriaEveryday Edition

This is an adaptation of the VIFT implementation for the **AriaEveryday dataset**, optimized for training and inference on NVIDIA H100 GPUs.

> **Original Paper**: Causal Transformer for Fusion and Pose Estimation in Deep Visual Inertial Odometry
> Yunus Bilge Kurt, Ahmet Akman, Aydın Alatan
> *ECCV 2024 VCAD Workshop*
> [[Paper](https://arxiv.org/abs/2409.08769)] [[Original Repo](https://github.com/ybkurt/VIFT)]

`<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white">``</a>`
`<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white">``</a>`
`<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd">``</a>`

## AriaEveryday Dataset Integration

This repository adapts VIFT for the **AriaEveryday dataset** - a large-scale egocentric dataset with 143 sequences of real-world indoor activities recorded with Project Aria glasses.

### Key Features for AriaEveryday:

- **Real-world data**: 143 sequences of everyday activities (cooking, cleaning, social interactions)
- **Multi-modal sensors**: RGB cameras + IMU + SLAM ground truth trajectories
- **H100 optimization**: Leverages PyTorch 2.3+ with TF32, torch.compile, and channels_last
- **Robust processing**: Handles corrupted video files and extracts MPS SLAM poses as ground truth
- **Scalable training**: Train on 100 sequences, test on 43 held-out sequences

### Architecture Adaptations:

- **AriaEveryday data loader**: Custom dataset class for Project Aria sensor data
- **SLAM trajectory ground truth**: Uses MPS closed-loop SLAM poses as training targets
- **Video processing pipeline**: Extracts RGB frames from Project Aria recordings
- **IMU synchronization**: Aligns 1kHz IMU data with 30Hz video frames

## Installation for AriaEveryday

Make sure you have NVIDIA H100 GPU access and CUDA 12.1+ installed.

#### Environment Setup

```bash
# Clone this AriaEveryday adaptation
git clone https://github.com/yfzzzyyls/VIFT_AEA
cd VIFT_AEA

# [RECOMMENDED] Create conda environment
conda create -n vift_aria python=3.11
conda activate vift_aria

# Install PyTorch for H100 (CUDA 12.1+)
pip install torch>=2.3.0 torchvision>=0.18.0 --index-url https://download.pytorch.org/whl/cu121

# Install AriaEveryday-specific requirements
pip install -r requirements.txt
```

## Downloading AriaEveryday Dataset

The AriaEveryday dataset contains 143 sequences of real-world activities recorded with Project Aria glasses.

```bash
# Download first 10 sequences to get started quickly
python download_aria_with_json.py AriaEverydayActivities_download_urls.json --output-dir data/aria_everyday_subset

# Alternative: Download all sequences (adjust path as needed)
mkdir -p data/aria_everyday
# Follow Project Aria instructions to download the full dataset
# https://www.projectaria.com/datasets/
```

Expected directory structure:

```
data/aria_everyday/
├── loc1_script1_seq1_rec1/
│   ├── AriaEverydayActivities_1.0.0_loc1_script1_seq1_rec1_preview_rgb.mp4
│   ├── AriaEverydayActivities_1.0.0_loc1_script1_seq1_rec1_mps_slam_trajectories.zip
│   └── ...
├── loc1_script1_seq3_rec1/
└── ...
```

## AriaEveryday Processing Pipeline

### Step 1: Extract SLAM Trajectories and Process Sensor Data

```bash
# Process first 10 sequences for training
python scripts/process_aria_to_vift.py \
  --input-dir data/aria_everyday \
  --output-dir data/aria_real_train \
  --start-index 0 \
  --max-sequences 10

# Process remaining 11 sequences for testing/inference
python scripts/process_aria_to_vift.py \
  --input-dir data/aria_everyday \
  --output-dir data/aria_real_test \
  --start-index 10 \
  --max-sequences 11
```

This extracts:

- **SLAM poses**: MPS closed-loop trajectories as ground truth
- **Visual data**: RGB frames from Project Aria cameras
- **IMU data**: Synchronized inertial measurements

### Step 2: Cache Latent Features

Generate latent features using pretrained Visual-Selective-VIO encoders:

```bash
# Download pretrained encoders (same as original VIFT)
mkdir -p pretrained_models
# Download from Visual-Selective-VIO repository

# Cache latents for training data
python data/latent_caching_aria.py \
  --input-dir data/aria_real_train \
  --output-dir data/aria_latent_train

# Cache latents for test data  
python data/latent_caching_aria.py \
  --input-dir data/aria_real_test \
  --output-dir data/aria_latent_test
```

### Step 3: Train VIFT on AriaEveryday

```bash
# Train with H100 optimizations
python src/train.py \
  data=aria_vio \
  trainer=gpu \
  trainer.devices=1 \
  trainer.max_epochs=50 \
  trainer.precision=16 \
  model.optimizer.lr=2e-4 \
  data.batch_size=32
```

### Step 4: Inference and Evaluation

```bash
# Run inference on test sequences
python src/test.py \
  data=aria_vio \
  ckpt=lightning_logs/version_X/checkpoints/best.ckpt \
  logger=csv
```

## H100 Optimization Features

This adaptation includes several H100-specific optimizations:

- **TF32 precision**: Enabled by default for faster training
- **torch.compile**: Reduces overhead with graph optimization
- **channels_last memory**: Improves tensor core utilization
- **Mixed precision (FP16)**: Leverages H100's tensor cores
- **Larger batch sizes**: Take advantage of 80GB HBM3 memory

## Performance Expectations

- **Dataset size**: 143 sequences (~250k frames total)
- **Training split**: 100 sequences for training, 43 for testing
- **Training time**: ~6 hours on H100 for 50 epochs
- **Memory usage**: ~40-60GB with optimized batch sizes

## AriaEveryday vs KITTI

| Aspect       | KITTI              | AriaEveryday              |
| ------------ | ------------------ | ------------------------- |
| Environment  | Outdoor driving    | Indoor activities         |
| Camera setup | Car-mounted stereo | Egocentric AR glasses     |
| Ground truth | GPS/LiDAR          | MPS SLAM                  |
| Sequences    | 22 sequences       | 143 sequences             |
| Activities   | Driving            | Cooking, cleaning, social |

## Acknowledgments

This project builds upon:

- **Original VIFT**: [Causal Transformer for Fusion and Pose Estimation in Deep Visual Inertial Odometry](https://github.com/ybkurt/VIFT)
- **AriaEveryday Dataset**: [Project Aria](https://www.projectaria.com/datasets/)
- **Visual-Selective-VIO**: For pretrained encoders
- **Lightning-Hydra-Template**: For project structure

## Citation

If you use this AriaEveryday adaptation, please cite both the original VIFT paper and acknowledge the AriaEveryday dataset:

```bibtex
@article{kurt2024vift,
  title={Causal Transformer for Fusion and Pose Estimation in Deep Visual Inertial Odometry},
  author={Kurt, Yunus Bilge and Akman, Ahmet and Alatan, Ayd{\i}n},
  journal={arXiv preprint arXiv:2409.08769},
  year={2024}
}

@article{pan2023aria,
  title={Aria Everyday Activities Dataset},
  author={Pan, Xiaqing and others},
  journal={arXiv preprint arXiv:2309.16045},
  year={2023}
}
```
