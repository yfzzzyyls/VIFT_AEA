# Bug Fix Summary: Incorrect Pose Scaling in AriaLatentDataset

## Issue
The training logs showed ground truth pose values that were way too large for relative poses:
```
0      [  39.35, -126.55,  164.75] [ 0.001,  0.028,  0.021,  0.999]
1      [  20.54, -133.10,  146.74] [ 0.001,  0.022,  0.015,  1.000]
```

These values (39.35, -126.55, 164.75) are 100x too large for frame-to-frame relative motion.

## Root Cause
The `AriaLatentDataset` class was incorrectly scaling the poses by 100:

```python
# CRITICAL: Scale translations from meters to centimeters to match training data
relative_poses[:, :3] *= 100.0
```

However, the GT files already contained values in centimeters, not meters! This caused a 100x scaling error.

## Evidence
1. **GT files contain small values (in cm):**
   ```
   Frame 0: Trans=[0.14, 0.10, 0.13] cm
   Frame 1: Trans=[0.10, 0.10, 0.12] cm
   ```

2. **Dataloader was returning 100x larger values:**
   - Before fix: Mean translation = 192.93 cm
   - After fix: Mean translation = 1.93 cm

## Fix Applied
Removed the incorrect scaling in `/home/external/VIFT_AEA/src/data/components/aria_latent_dataset.py`:

```python
# Before:
relative_poses[:, :3] *= 100.0

# After:
# NOTE: GT files already contain poses in centimeters, no scaling needed!
```

## Impact
- The model will now train on correct relative pose values (0.01-5 cm per frame)
- This should significantly improve training convergence and model performance
- The model predictions should now be in the correct scale

## Next Steps
1. Re-run training with the corrected data loading
2. The model should now learn realistic frame-to-frame motion patterns
3. Monitor that predictions are in the 0.01-5 cm range, not 100-500 cm