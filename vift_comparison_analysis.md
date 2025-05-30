# VIFT Implementation Comparison & Recommendations

## Key Differences Found

### 1. Data Normalization Issues

#### Original VIFT (KITTI)
- **Images**: Simple normalization with `-0.5` (line 43 in custom_transform.py)
- **IMU**: No normalization, raw data from MATLAB files
- **IMU frequency**: 10 Hz, 11 samples per frame window

#### Our Implementation (Aria)
- **Images**: Using ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **IMU**: Raw data, but Aria has 1000 Hz (33 samples averaged per frame)
- **IMU processing**: We average 33 samples to get 1 per frame, then pad to 256 dims

**🔧 Recommendation**: Change image normalization to match VIFT:
```python
# In latent_caching_aria.py, replace:
self.transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# With:
self.transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.Lambda(lambda x: x - 0.5)  # Match VIFT normalization
])
```

### 2. Loss Function Configuration

#### Original VIFT
- **Angle weight**: 40-100 (much higher than translation)
- **Loss type**: RPMGPoseLoss (Riemannian manifold) for rotations
- **Sample weighting**: Based on rotation difficulty

#### Our Implementation
- Using same loss functions but may have different weights
- Check configs/experiment/deep_vift_training.yaml

**🔧 Recommendation**: Ensure angle_weight is set correctly (40-100 range)

### 3. Model Architecture

#### Original VIFT
- **Causal masking**: Uses autoregressive prediction with causal mask
- **Input dimension**: 768 (512 visual + 256 inertial)
- **Embedding dimension**: Configurable (128-768)

#### Our Implementation
- Same architecture but verify causal masking is enabled

### 4. Data Format

#### Original VIFT (KITTI)
- **Pose format**: [rx, ry, rz, tx, ty, tz] (Euler angles in radians)
- **Ground truth**: Relative poses between consecutive frames

#### Our Implementation (Aria)
- **Pose format**: Same 6-DoF format after quaternion→Euler conversion
- **Ground truth**: Should be relative poses, not absolute

**🔧 Recommendation**: Verify poses are relative, not absolute:
```python
# In process_aria_to_vift.py, compute relative poses:
def compute_relative_poses(poses):
    rel_poses = []
    for i in range(1, len(poses)):
        # Compute relative transformation from pose[i-1] to pose[i]
        rel_pose = compute_relative_transform(poses[i-1], poses[i])
        rel_poses.append(rel_pose)
    return rel_poses
```

## Critical Issues to Fix

### 1. Image Normalization Mismatch
The ImageNet normalization vs simple -0.5 could significantly impact performance since the pretrained encoder expects specific input ranges.

### 2. IMU Data Processing
- Original VIFT uses 101 IMU samples for 11 frames (10 Hz × 10.1 seconds)
- We use 11×33 = 363 samples averaged down to 11
- Consider resampling IMU to match VIFT's 10 Hz pattern

### 3. Loss Weight Configuration
Ensure the angle_weight in loss function matches original (40-100 range)

### 4. Relative vs Absolute Poses
VIFT predicts relative poses between consecutive frames, not absolute poses

## Recommended Next Steps

1. **Fix image normalization** in latent_caching_aria.py
2. **Verify pose format** - ensure we're using relative poses
3. **Check loss weights** in training config
4. **Test with small dataset** to verify corrections
5. **Consider IMU resampling** to match 10 Hz frequency

## Quick Test Script

Create a test to verify data format:
```python
# test_data_format.py
import numpy as np
import torch

# Load a sample
latent = np.load("aria_latent_data/train/0.npy")
gt = np.load("aria_latent_data/train/0_gt.npy")

print(f"Latent shape: {latent.shape}")  # Should be (11, 768)
print(f"GT shape: {gt.shape}")  # Should be (11, 6) or (10, 6) for relative poses
print(f"GT sample: {gt[0]}")  # Check if values are reasonable (rotations in radians)
```