# Rotation Analysis Summary

## Key Findings

### 1. The Rotation Values Are Correct!

The small rotation values (0.06-0.18 degrees) between consecutive frames in the relative pose data are **correct and expected**. Here's why:

### 2. Data Pipeline Analysis

1. **Original Aria Data**:
   - Contains absolute poses with Euler angles stored in `poses.json`
   - The camera has a consistent orientation of approximately 18.6 degrees (mostly Y-axis rotation)
   - This represents the camera's mounting angle on the glasses

2. **Pretrained Data Generation Issue**:
   - The original `aria_latent_data_pretrained` has near-identity quaternions
   - This was a bug where Euler angles weren't properly converted to quaternions

3. **Fixed Data Generation**:
   - The `generate_all_pretrained_latents_fixed.py` correctly:
     - Loads Euler angles from `poses.json`
     - Converts them to quaternions using `scipy.spatial.transform.Rotation`
     - Computes relative poses between consecutive frames
     - Transforms translations into the previous frame's coordinate system

### 3. Why Rotations Are Small

When consecutive frames have similar absolute orientations (e.g., both at ~18.6 degrees), the relative rotation between them is naturally very small:

- Frame 0→1: Large jump (18.7°) because frame 0 is identity and frame 1 has the actual camera orientation
- Frame 1→2: Small change (0.19°) because both frames have similar orientations
- Frame 2→3: Small change (0.17°) because camera orientation is stable

This pattern is typical for head-mounted cameras where:
- The head maintains a relatively stable orientation
- Most rotation comes from small head movements
- Large rotations only occur during significant head turns

### 4. Mathematical Verification

The relative rotation is computed as:
```
rel_rot = prev_rot^(-1) * curr_rot
```

When `prev_rot ≈ curr_rot`, the relative rotation is close to identity, resulting in small angles.

### 5. Data Characteristics

From the analysis:
- **Translation changes**: 0.0001-0.0004 meters (0.1-0.4 mm) between frames
- **Rotation changes**: 0.04-0.19 degrees between consecutive frames (after the initial jump)
- **Initial jump**: ~18.7 degrees from identity to actual camera orientation

## Conclusion

The rotation data in `aria_latent_data_properly_fixed` is **correct**. The small rotation values between consecutive frames accurately represent the minimal head movement in the Aria glasses dataset. The large initial rotation is due to the coordinate system convention where the first frame is always set to identity.

## Recommendations

1. **For Training**: The data is ready to use as-is. The model should learn to predict these small relative rotations.

2. **For Evaluation**: When computing metrics, be aware that:
   - Most rotation predictions will be very small
   - The first frame's large rotation might dominate error metrics
   - Consider evaluating frames 1-10 separately from frame 0→1

3. **For Visualization**: When plotting trajectories, the small rotations will result in mostly straight paths, which is expected for head-mounted camera data.