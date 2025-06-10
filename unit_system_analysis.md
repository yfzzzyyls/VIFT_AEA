# VIFT-AEA Unit System Analysis and Recommendations

## Executive Summary

After analyzing the entire VIFT-AEA codebase, I've identified critical unit system inconsistencies that affect training and inference. The main issue is a **mismatch between meters (used in data processing) and centimeters (expected by the model)**, along with unclear IMU unit specifications.

## Current Unit System Status

### 1. Data Processing Pipeline (`scripts/process_aria_to_vift.py`)
- **Translation**: Stored in **meters** (lines 79-80, 216-217)
- **Rotation**: Quaternions (XYZW format) - unitless
- **Accelerometer**: m/s² with gravity (9.81 m/s²) added (line 318)
- **Gyroscope**: rad/s (line 336)
- **No explicit unit conversion** during data processing

### 2. Feature Generation (`generate_all_pretrained_latents_fixed.py`)
- **Default pose_scale**: 100.0 (line 151)
- **Translation scaling**: Multiplies by pose_scale (line 214)
- This converts meters to centimeters: `window_relative_poses[:, :3] *= pose_scale`
- **Critical**: This scaling happens AFTER relative pose computation

### 3. Data Loading (`src/data/aria_datamodule.py`)
- **Explicit conversion**: Line 93 converts meters to centimeters
- `poses[i, :3] = torch.tensor(pose['translation']) * 100.0`
- **Issue**: Double scaling if features were already scaled!

### 4. Loss Functions (`src/metrics/arvr_loss.py`)
- **Translation thresholds**: 
  - Small: 0.01 (1cm) - line 27
  - Large: 0.05 (5cm) - line 28
- **Rotation thresholds**:
  - Small: 0.035 rad (~2°) - line 25
  - Large: 0.175 rad (~10°) - line 26
- **Assumes centimeter scale** for translations

### 5. Original Pretrained Model (Visual-Selective-VIO)
- Trained on **KITTI dataset**
- KITTI uses **meters** for all measurements
- Model likely expects **meter-scale** inputs
- IMU: Standard units (m/s² for accelerometer, rad/s for gyroscope)

## Key Issues Identified

### 1. **Double Scaling Problem**
- Features are scaled to cm during latent generation (×100)
- Then scaled again during data loading (×100)
- Results in 10,000× scale factor!

### 2. **Inconsistent Scale Assumptions**
- Pretrained encoder trained on meter-scale data
- Loss functions assume centimeter-scale thresholds
- No clear documentation of expected units

### 3. **IMU Unit Ambiguity**
- Accelerometer: Correctly uses m/s² with gravity
- Gyroscope: Uses rad/s (standard)
- But no normalization or scaling applied to match visual features

### 4. **Numerical Stability Concerns**
- Meter-scale translations (0.001-0.1m) are very small values
- Can cause gradient vanishing in neural networks
- Centimeter scale (0.1-10cm) provides better numerical range

## Recommendations

### 1. **Standardize on Centimeters** (Recommended)
**Rationale:**
- Better numerical stability (values in 0.1-10 range vs 0.001-0.1)
- Loss functions already assume cm scale
- AR/VR applications need sub-centimeter precision
- Easier to interpret (2cm error vs 0.02m error)

**Implementation:**
```python
# In process_aria_to_vift.py:
# Store translations in meters (keep as-is for compatibility)

# In generate_all_pretrained_latents_fixed.py:
pose_scale = 100.0  # Keep this - converts m to cm

# In aria_datamodule.py:
# REMOVE the ×100 scaling (line 93) - already done in features!
poses[i, :3] = torch.tensor(pose['translation'])  # No ×100!

# In loss functions:
# Keep current thresholds (already in cm)
```

### 2. **Fix the Double Scaling Issue**
```python
# Option A: Scale only during feature generation
# - Keep pose_scale=100 in generate_all_pretrained_latents_fixed.py
# - Remove ×100 in aria_datamodule.py

# Option B: Scale only during data loading
# - Set pose_scale=1.0 in generate_all_pretrained_latents_fixed.py
# - Keep ×100 in aria_datamodule.py

# Recommendation: Option A (scale early, once)
```

### 3. **Document IMU Units Clearly**
```python
# Add to data processing:
"""
IMU Data Units:
- Accelerometer: m/s² (includes gravity: +9.81 in Z)
- Gyroscope: rad/s
- Frequency: 1000 Hz (resampled to match frames)
"""
```

### 4. **Add Unit Validation**
```python
def validate_units(poses, imu_data):
    """Validate data is in expected units."""
    # Check translation magnitudes (should be in cm)
    trans_magnitudes = np.linalg.norm(poses[:, :3], axis=1)
    if np.median(trans_magnitudes) < 0.1:  # Likely in meters
        print("WARNING: Translations appear to be in meters!")
    
    # Check IMU magnitudes
    accel_magnitudes = np.linalg.norm(imu_data[:, :, 3:], axis=2).mean()
    if accel_magnitudes < 5.0:  # Should be ~9.81 with gravity
        print("WARNING: Accelerometer may be normalized!")
```

### 5. **Update Configuration Files**
```yaml
# In configs/data/aria_vio.yaml:
data_units:
  translation: "centimeters"  # After scaling
  rotation: "quaternion_xyzw"
  accelerometer: "m/s^2"
  gyroscope: "rad/s"
  
# Scaling parameters
translation_scale: 100.0  # m to cm
rotation_scale: 1.0      # No scaling for quaternions
```

## Immediate Action Items

1. **Fix Double Scaling** (Critical):
   ```bash
   # Edit src/data/aria_datamodule.py line 93
   # Remove the * 100.0 multiplication
   ```

2. **Regenerate Features** (If needed):
   ```bash
   # If you already generated features with pose_scale=100
   # AND your datamodule has ×100, you have 10,000× scale!
   # Regenerate with pose_scale=1.0 OR fix datamodule
   ```

3. **Add Unit Tests**:
   ```python
   def test_unit_consistency():
       # Load a sample
       # Check translation is in expected range (0.1-10 cm)
       # Check rotation is normalized quaternion
       # Check IMU has gravity component
   ```

## Alternative: Stick with Meters

If you prefer to use meters throughout:

1. Set `pose_scale = 1.0` in feature generation
2. Remove `* 100.0` in data loading  
3. Update loss function thresholds:
   - Small translation: 0.0001 (0.1mm)
   - Large translation: 0.0005 (0.5mm)
4. Risk: Very small gradients, potential numerical issues

## Conclusion

The unit system inconsistency is causing significant issues. The immediate fix is to:
1. Remove double scaling in `aria_datamodule.py`
2. Standardize on centimeters throughout
3. Document units clearly in code and configs
4. Add validation to catch unit mismatches

This will resolve the numerical stability issues and ensure consistent behavior between training and inference.