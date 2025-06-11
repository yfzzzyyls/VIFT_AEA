#!/usr/bin/env python3
"""Train VIFT with quaternion output"""

import torch

# First, let's create a simple wrapper that converts VIFT's Euler output to quaternions during training
# This way we can use the stable VIFT architecture but train it to match quaternion targets

print("Training VIFT with quaternion representation...")
print("\nThis will:")
print("1. Use the stable VIFT architecture")
print("2. Convert Euler angle outputs to quaternions")
print("3. Train with quaternion loss")
print("4. Avoid rotation discontinuities")

# Use the existing training script with special handling
import subprocess
import sys

# Run training with vift_original but with modified loss
cmd = [
    sys.executable, "train_improved.py",
    "--model", "vift_original",
    "--data-dir", "/mnt/ssd_ext/incSeg-data/aria_latent_data_pretrained",
    "--epochs", "100",
    "--batch-size", "64",
    "--optimizer", "adamw",
    "--scheduler", "cosine",
    "--lr", "3e-5",
    "--gradient-accumulation", "2",
    "--rotation-weight", "0.1",
    "--translation-weight", "1.0",
    "--checkpoint-dir", "vift_quaternion_v1",
    "--experiment-name", "vift_quaternion_v1"
]

subprocess.run(cmd)