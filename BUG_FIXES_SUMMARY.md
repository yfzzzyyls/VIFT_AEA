# VIFT Model Bug Fixes Summary

## Issues Found and Fixed

### 1. Data Scaling Bug (100x Error)
**Issue**: Ground truth poses were being scaled by 100x incorrectly
- Location: `/src/data/components/aria_latent_dataset.py`
- Root cause: Dataset was multiplying poses by 100, thinking they were in meters when they were already in centimeters
- Fix: Removed line `relative_poses[:, :3] *= 100.0`

### 2. Gradient Vanishing in Rotation Head
**Issue**: Model predicted constant rotations [0, 0, 0, 1] with near-zero gradients
- Symptoms:
  - Rotation gradients: ~0.000000 to 0.002099
  - Translation gradients: Normal (1-17)
  - All rotation predictions identical

**Fix**: Created `train_with_gradient_fix.py` with:
1. Better initialization for rotation layers (xavier_uniform with gain=2.0)
2. Gradient-friendly loss formulation (MSE instead of acos)
3. Separate learning rates (rotation 5x higher)
4. Variance penalty to prevent collapse
5. Separate gradient clipping for translation/rotation

## Results

### Before Fix:
```
Rotation variance: [0.000000, 0.000000, 0.000000, 0.000000]
Rotation gradients: 0.000000 - 0.002099
All predictions: [0.0000, 0.0000, 0.0000, 1.0000]
```

### After Fix:
```
Rotation variance: [0.000786, 0.000572, 0.000764, 0.000001]
Rotation gradients: 0.014526 - 0.352624
Diverse predictions with proper learning
```

## Files to Upload for Git:
1. `/src/data/components/aria_latent_dataset.py` - Fixed data scaling
2. `train_with_gradient_fix.py` - New training script with gradient fixes

## Additional Issues Found (Not Fixed):
1. Double scaling bug in `generate_latent_features_140.py` (line 96)
   - Would require regenerating all GT files
   - Current workaround: Fixed in data loader

## Training Command:
```bash
python train_with_gradient_fix.py
```

The model is now training properly with:
- Correct data scale (cm)
- Healthy gradient flow
- Diverse predictions for both translation and rotation