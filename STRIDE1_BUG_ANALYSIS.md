# Stride=1 Performance Issue: Root Cause Analysis

## Executive Summary

The poor performance with stride=1 (0.1743° rotation error vs expected 0.0739°) is caused by an **incorrect relative pose conversion** in both `generate_all_pretrained_latents.py` and `train_pretrained_relative.py`. The bug affects how translations are computed for relative poses.

## The Bug

### Current (Incorrect) Implementation
```python
# In convert_absolute_to_relative() function
trans_diff = curr_trans - prev_trans  # WRONG: This is in WORLD coordinates
relative_poses[i, :3] = trans_diff
```

### Correct Implementation
```python
# Compute translation difference in world coordinates
trans_diff_world = curr_trans - prev_trans

# Transform to previous frame's coordinate system
prev_rot_matrix = quaternion_to_matrix(prev_rot)
trans_diff_local = prev_rot_matrix.T @ trans_diff_world

relative_poses[i, :3] = trans_diff_local
```

## Why This Matters

When a robot/camera rotates and then moves forward:
- **Wrong method**: The forward motion is recorded as motion in the original world direction
- **Correct method**: The forward motion is recorded as motion in the robot's current facing direction

This is critical for VIO because the model needs to learn motion patterns in the camera's local frame, not the world frame.

## Example Scenario

Consider a robot that:
1. Starts at origin facing +X
2. Moves 1m forward (along +X)
3. Turns 90° right (now facing +Y)
4. Moves 1m forward (along +Y in world)

**Current (wrong) output:**
- Frame 0→1: [100, 0, 0] cm (correct)
- Frame 2→3: [0, 100, 0] cm (WRONG - this is world coordinates)

**Fixed output:**
- Frame 0→1: [100, 0, 0] cm (correct)
- Frame 2→3: [100, 0, 0] cm (CORRECT - forward in local frame)

## Impact on Training

1. **Rotation Error**: The model learns incorrect motion patterns, especially after rotations
2. **Stride Sensitivity**: With stride=1, there are many more samples with small rotations where this error accumulates
3. **Performance Degradation**: The model can't properly predict motion in the camera frame

## Files Affected

1. `generate_all_pretrained_latents.py` - Lines 95-97
2. `train_pretrained_relative.py` - Lines 84-86
3. `preprocess_aria_with_fixed_quaternions.py` - Uses the buggy function from train script

## Solution

I've created fixed versions:
- `generate_all_pretrained_latents_fixed.py` - Properly transforms translations to local frame
- `train_pretrained_relative_fixed.py` - Same fix for training-time conversion

## Recommended Actions

1. **Regenerate data with the fixed script:**
   ```bash
   python generate_all_pretrained_latents_fixed.py --stride 1 --output-dir aria_latent_data_stride1_fixed
   ```

2. **Train with the fixed data:**
   ```bash
   python train_pretrained_relative_fixed.py --data_dir aria_latent_data_stride1_fixed
   ```

3. **Expected Results:**
   - Rotation error should drop from 0.1743° to ~0.0739° or better
   - ATE should remain excellent (<1cm)

## Verification

The fix ensures:
- First pose is always at origin [0,0,0, 0,0,0,1]
- Translations are in the previous frame's coordinate system
- Quaternions remain normalized
- The model learns proper local motion patterns

This explains why the commit ffd16b6c achieved excellent results - it likely had the correct coordinate transformation implemented.