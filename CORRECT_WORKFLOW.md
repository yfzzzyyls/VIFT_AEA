# Correct Workflow for Best Results

## The Problem
The old workflow has too many steps and potential mismatches:
1. `generate_all_pretrained_latents.py` (old version) → generates absolute poses
2. `preprocess_aria_with_fixed_quaternions.py` → converts to relative poses
3. Training on the converted data

This multi-step process can introduce errors and inconsistencies.

## The Solution
Use the updated `generate_all_pretrained_latents.py` that does everything in one shot:
- Generates features using Visual-Selective-VIO model
- Converts poses to relative format with correct quaternion handling
- Scales translations from meters to centimeters
- Uses stride=5 by default (matching the original successful configuration)

## Correct Single-Step Workflow

### 1. Clean Previous Data
```bash
rm -rf aria_latent_data_pretrained
rm -rf aria_latent_data_fixed
```

### 2. Generate Features with Relative Poses (One Step!)
```bash
# Make sure you have the corrected script
python generate_all_pretrained_latents.py
```

This single command:
- ✅ Loads the Visual-Selective-VIO pretrained model
- ✅ Extracts features with stride=5
- ✅ Converts absolute poses to relative poses
- ✅ Sets first pose to origin [0,0,0, 0,0,0,1]
- ✅ Scales translations to centimeters
- ✅ Saves everything in the correct format

Expected output:
- ~9,400 training samples (with stride=5)
- ~1,100 validation samples
- ~1,300 test samples

### 3. Train Directly on Generated Data
```bash
python train_pretrained_relative.py \
    --data_dir aria_latent_data_pretrained \
    --lr 5e-5 \
    --batch_size 64 \
    --epochs 30
```

Note: Use `aria_latent_data_pretrained` directly, NOT `aria_latent_data_fixed`!

### 4. Evaluate
```bash
python evaluate_with_metrics.py \
    --checkpoint logs/checkpoints_lite_scale_100.0/best_model.ckpt
```

## Why Your Current Results Are Worse

1. **Data mismatch**: The two-step process (generate then preprocess) can introduce inconsistencies
2. **Stride difference**: If you used stride=1, you get 5x more data but potentially noisier training
3. **Checkpoint issue**: The val_loss=0.0000 suggests potential training issues

## Verification
After running the corrected `generate_all_pretrained_latents.py`, verify:

```bash
python -c "
import numpy as np
poses = np.load('aria_latent_data_pretrained/train/0_gt.npy')
print(f'First pose: {poses[0]}')
print(f'At origin: {np.allclose(poses[0, :3], [0,0,0])}')
angles = []
for i in range(1, len(poses)):
    q = poses[i, 3:]
    angle = 2 * np.arccos(np.clip(q[3], -1, 1)) * 180 / np.pi
    angles.append(angle)
print(f'Mean rotation: {np.mean(angles):.4f}°')
"
```

Expected output:
- First pose: [0. 0. 0. 0. 0. 0. 1.]
- At origin: True
- Mean rotation: ~1-2° (small relative rotations)

## Summary
The key is to use the updated `generate_all_pretrained_latents.py` that integrates all the fixes:
- Correct quaternion handling (XYZW format)
- Relative pose conversion
- Proper scaling
- Stride=5 by default

This should give you the expected 0.0739° rotation error!