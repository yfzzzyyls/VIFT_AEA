# Unit System Fix Implementation Plan

## Immediate Fixes Required:

### 1. Fix Double Scaling Bug in DataModule

**File: `src/data/aria_datamodule.py`**
```python
# REMOVE line 93:
# poses[:, :3] *= 100.0  # DELETE THIS LINE - already scaled during generation!
```

### 2. Ensure Consistent Training Configuration

**Keep these settings:**
- Feature generation: `pose_scale=100.0` (converts m → cm once)
- Training config: expects centimeters
- Model outputs: centimeters

### 3. Fix Inference Pipeline

**Update inference command:**
```bash
python inference_full_sequence.py \
    --checkpoint your_checkpoint.ckpt \
    --sequence-id all \
    --stride 1 \
    --pose-scale 100.0  # Must match training!
```

### 4. Update Visualization

**File: `organize_plots_by_sequence.py`**
```python
# Lines 19-20 are correct if data is in meters:
gt_pos = gt_traj[:, :3] * 100  # Convert m to cm for display
pred_pos = pred_traj[:, :3] * 100  # Convert m to cm for display

# But if data is already in cm (after fixing), change to:
gt_pos = gt_traj[:, :3]  # Already in cm
pred_pos = pred_traj[:, :3]  # Already in cm
```

## Recommended Training Pipeline:

### 1. Data Generation (Current - Correct)
```bash
python generate_full_dataset.py \
    --processed-dir /path/to/aria_processed \
    --output-dir aria_latent_full_603010_stride20 \
    --stride 20 \
    --pose-scale 100.0  # Convert m → cm
```

### 2. Training (After fixing datamodule)
```bash
python train_fixed.py \
    --config configs/experiment/fixed_training.yaml \
    --data.pose_scale 100.0
```

### 3. Inference (Must match training scale)
```bash
python inference_full_sequence.py \
    --checkpoint checkpoints/best_model.ckpt \
    --sequence-id all \
    --stride 20 \  # Match training stride for fair comparison
    --pose-scale 100.0  # Match training scale!
```

## Why This Matters:

1. **Current Issue**: Model trained on 10,000× scaled data, inference on 1× data
2. **After Fix**: Consistent centimeter scale throughout
3. **Expected Results**: Trajectories should align much better

## Validation Steps:

1. Check average frame-to-frame translation:
   - Should be ~0.5-2.0 cm at stride=1
   - Should be ~10-40 cm at stride=20

2. Verify no double scaling:
   - Print `poses[:, :3].mean()` in datamodule
   - Should be in reasonable cm range (not 1000s)

3. Test with known sequence:
   - Run inference on training sequence
   - Should see near-perfect alignment

## Long-term Improvements:

1. Add unit assertions in code:
   ```python
   assert 0.1 < poses[:, :3].abs().mean() < 100, "Poses should be in cm"
   ```

2. Document units clearly:
   ```python
   # All translations in CENTIMETERS
   # All rotations in RADIANS (quaternions are unitless)
   # IMU: m/s² for accel, rad/s for gyro
   ```

3. Add unit tests for scale consistency

This fix should resolve your trajectory prediction issues!