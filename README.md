---

<div align="center">

# VIFT-AEA: Visual-Inertial Feature Transformer for AriaEveryday

This is an adaptation of the VIFT implementation for the **AriaEveryday dataset**, optimized for training and inference on NVIDIA H100 GPUs and Apple Silicon.

> **Original Paper**: Causal Transformer for Fusion and Pose Estimation in Deep Visual Inertial Odometry
> Yunus Bilge Kurt, Ahmet Akman, Aydın Alatan
> *ECCV 2024 VCAD Workshop*
> [[Paper](https://arxiv.org/abs/2409.08769)] [[Original Repo](https://github.com/ybkurt/VIFT)]

`<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white">``</a>`
`<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white">``</a>`
`<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd">``</a>`

## Overview

VIFT-AEA is a Visual-Inertial Odometry (VIO) system that leverages transformer architectures for robust pose estimation. This implementation extends the original VIFT framework to support the AriaEveryday dataset, enabling training and inference on real-world AR/VR data captured with Meta's Aria glasses.

## Features

- **Cross-Platform Support**: Seamless operation on both CUDA-enabled Linux systems and Apple Silicon (M1/M2) macOS
- **AriaEveryday Dataset Integration**: Native support for processing and training on Meta's AriaEveryday dataset
- **Transformer-Based Architecture**: State-of-the-art visual-inertial fusion using attention mechanisms
- **Automated Environment Setup**: Intelligent platform detection and dependency installation

## System Requirements

### Supported Platforms
- **Linux**: CUDA 11.8+ compatible GPUs (RTX 20/30/40 series, Tesla, etc.)
- **macOS**: Apple Silicon (M1/M2/M3) with Metal Performance Shaders (MPS)
- **Fallback**: CPU-only execution on any platform

### Dependencies
- Python 3.8+
- PyTorch 2.3.0+
- OpenCV 4.8.0+
- NumPy 2.0.0+

## Quick Start

### 1. Environment Setup

Clone the repository and navigate to the project directory:
```bash
git clone <repository-url>
cd VIFT_AEA
```

Create and activate a Python virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
```

Run the automated environment setup script:
```bash
python scripts/setup_env.py
```

This script will:
- Automatically detect your platform (CUDA/Apple Silicon/CPU)
- Install the appropriate PyTorch version with hardware acceleration
- Install all required dependencies from `requirements.txt`
- Verify the installation

### 2. Dataset Preparation

#### KITTI Dataset (for baseline comparison)
```bash
cd data
chmod +x data_prep.sh
./data_prep.sh
```

#### AriaEveryday Dataset
Place your AriaEveryday dataset in the `data/aria_everyday/` directory, then process it:
```bash
python scripts/process_aria_to_vift.py \
  --input-dir data/aria_everyday \
  --output-dir data/aria_real_train \
  --max-sequences 50
```

### 3. Training

Train the model on your dataset:
```bash
# Train on KITTI
python src/train.py data=kitti_vio

# Train on AriaEveryday
python src/train.py data=aria_vio
```

### 4. Evaluation

Evaluate the trained model:
```bash
python src/eval.py ckpt_path=path/to/checkpoint.ckpt
```

## Platform-Specific Features

### CUDA (Linux)
- Utilizes CUDA 11.8 for GPU acceleration
- Automatic mixed precision training
- Multi-GPU support for large-scale training

### Apple Silicon (macOS)
- Leverages Metal Performance Shaders (MPS) backend
- Optimized for M1/M2/M3 processors
- Native ARM64 binaries for maximum performance

### CPU Fallback
- Full functionality on any x86_64 or ARM64 system
- Automatic fallback when GPU acceleration is unavailable

## Project Structure

```
VIFT_AEA/
├── src/                    # Source code
│   ├── models/            # Model architectures
│   ├── data/              # Dataset loaders
│   └── train.py           # Training script
├── scripts/               # Utility scripts
│   ├── setup_env.py       # Environment setup
│   └── process_aria_to_vift.py  # AriaEveryday processing
├── data/                  # Datasets
├── configs/               # Hydra configurations
└── requirements.txt       # Dependencies
```

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/data/kitti_vio.yaml` - KITTI dataset configuration
- `configs/data/aria_vio.yaml` - AriaEveryday dataset configuration
- `configs/model/vift.yaml` - Model architecture configuration
- `configs/trainer/default.yaml` - Training configuration

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed by re-running `python scripts/setup_env.py`
2. **CUDA out of memory**: Reduce batch size in the configuration files
3. **Apple Silicon performance**: Ensure you're using the MPS backend by checking `torch.backends.mps.is_available()`

### Platform Detection

The setup script automatically detects your platform. You can verify detection by running:
```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')
"
```

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{vift_aea2024,
  title={VIFT-AEA: Visual-Inertial Feature Transformer for AriaEveryday},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Acknowledgments

This project builds upon:

- **Original VIFT**: [Causal Transformer for Fusion and Pose Estimation in Deep Visual Inertial Odometry](https://github.com/ybkurt/VIFT)
- **AriaEveryday Dataset**: [Project Aria](https://www.projectaria.com/datasets/)
- **Visual-Selective-VIO**: For pretrained encoders
- **Lightning-Hydra-Template**: For project structure
- **Meta AI**: For the AriaEveryday dataset
- **PyTorch team**: For cross-platform ML framework
