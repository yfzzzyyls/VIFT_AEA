#!/usr/bin/env python3
"""Test script to verify scaling fixes."""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

# Test 1: Check generated data scale
print("=== Test 1: Check Generated Data Scale ===")
test_gt = np.load('test_latent_features/train/0_gt.npy')
print(f"Data shape: {test_gt.shape}")
print(f"First 3 translations (should be in cm):")
for i in range(min(3, len(test_gt))):
    trans = test_gt[i, :3]
    print(f"  Sample {i}: {trans} (magnitude: {np.linalg.norm(trans):.4f})")

# Test 2: Check datamodule loading
print("\n=== Test 2: Check DataModule Loading ===")
from src.data.aria_datamodule_fixed import AriaDataModuleFixed

datamodule = AriaDataModuleFixed(
    data_dir='test_latent_features',
    batch_size=4,
    num_workers=0,
    stride=20,
    pose_scale=100.0  # This should NOT scale again
)

datamodule.setup()
train_loader = datamodule.train_dataloader()

# Get a batch
batch = next(iter(train_loader))
visual_features, imu_features, poses = batch

print(f"\nBatch shapes:")
print(f"  Visual: {visual_features.shape}")
print(f"  IMU: {imu_features.shape}")
print(f"  Poses: {poses.shape}")

print(f"\nFirst pose in batch:")
first_pose = poses[0, 0].numpy()  # First sample, first timestep
print(f"  Translation: {first_pose[:3]} (magnitude: {np.linalg.norm(first_pose[:3]):.4f})")
print(f"  Quaternion: {first_pose[3:]}")

# Test 3: Check if poses match loaded data
print("\n=== Test 3: Verify No Double Scaling ===")
print(f"Original data translation magnitude: {np.linalg.norm(test_gt[0, :3]):.6f}")
print(f"Loaded data translation magnitude: {np.linalg.norm(first_pose[:3]):.6f}")
if np.allclose(np.linalg.norm(test_gt[0, :3]), np.linalg.norm(first_pose[:3]), rtol=1e-5):
    print("✓ No double scaling detected!")
else:
    print("✗ WARNING: Possible scaling mismatch!")

# Test 4: Statistics across batch
print("\n=== Test 4: Batch Statistics ===")
trans_norms = torch.norm(poses[:, :, :3], dim=2).numpy()
print(f"Translation magnitudes (cm):")
print(f"  Mean: {trans_norms.mean():.4f}")
print(f"  Std: {trans_norms.std():.4f}")
print(f"  Min: {trans_norms.min():.4f}")
print(f"  Max: {trans_norms.max():.4f}")

print("\n✓ Testing complete!")