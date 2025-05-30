# Root Cause Analysis: High Train/Inference Loss in VIFT-AEA

## Summary of Issues Found

After analyzing your implementation against the original VIFT codebase, I've identified **three critical issues** causing high losses:

### 1. **Image Normalization Mismatch** 🔴 CRITICAL
- **Your implementation**: Uses ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Original VIFT**: Uses simple `-0.5` normalization
- **Impact**: The pretrained VIFT encoder expects inputs normalized with `-0.5`. Using different normalization completely changes the input distribution, causing poor feature extraction.

### 2. **Absolute vs Relative Poses** 🔴 CRITICAL  
- **Your implementation**: Uses absolute poses from SLAM trajectory
- **Original VIFT**: Predicts relative poses between consecutive frames
- **Impact**: The model is trying to predict absolute world positions instead of frame-to-frame motion, which is much harder and leads to high losses.

### 3. **Data Range Issues**
- **Your poses**: Have very large values (e.g., translation up to 2.6m, rotation up to 5.8 radians)
- **Expected**: Relative poses should have much smaller values (typically <0.1m for indoor motion between frames)

## Current Performance
- **Training loss**: ~129-149 (very high)
- **Inference MSE**: 3.18
- **Translation RMSE**: 2.29 meters
- **Rotation RMSE**: 1.05 radians (60.2°)

## Required Fixes

### Fix 1: Image Normalization
In `data/latent_caching_aria.py`, change line 23-27:
```python
# OLD (ImageNet normalization)
self.transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# NEW (VIFT normalization)
self.transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.Lambda(lambda x: x - 0.5)
])
```

### Fix 2: Convert to Relative Poses
In `data/latent_caching_aria.py`, modify the ground truth preparation (around line 100):
```python
# Compute relative poses instead of using absolute poses
gt_poses = []
for i in range(len(poses_seq)):
    if i == 0:
        # First frame: no motion
        gt_poses.append([0, 0, 0, 0, 0, 0])
    else:
        # Compute relative transformation from pose[i-1] to pose[i]
        prev_pose = poses_seq[i-1]['pose_6dof']
        curr_pose = poses_seq[i]['pose_6dof']
        
        # Simple difference for now (better: use proper SE3 relative transform)
        rel_rotation = [curr_pose[j] - prev_pose[j] for j in range(3)]
        rel_translation = [curr_pose[j+3] - prev_pose[j+3] for j in range(3)]
        
        rel_pose = rel_rotation + rel_translation
        gt_poses.append(rel_pose)

ground_truths.append(np.array(gt_poses))
```

### Fix 3: Re-cache Features
After making these fixes, you need to re-generate all cached features:
```bash
# Remove old cached data
rm -rf aria_latent_data/

# Re-cache with fixed normalization and relative poses
python data/latent_caching_aria.py --data_dir data/aria_split/train --save_dir aria_latent_data/train --mode train --device cuda
python data/latent_caching_aria.py --data_dir data/aria_split/val --save_dir aria_latent_data/val --mode val --device cuda
python data/latent_caching_aria.py --data_dir data/aria_split/test --save_dir aria_latent_data/test --mode test --device cuda
```

## Expected Results After Fixes
Based on the original VIFT paper:
- **Training loss**: Should decrease to <10 within a few epochs
- **Translation RMSE**: Should be <0.1m for relative poses
- **Rotation RMSE**: Should be <0.1 rad (5.7°) for relative poses

## Additional Recommendations
1. **Verify pose format**: Add logging to check that relative poses have small magnitudes
2. **Monitor convergence**: Loss should drop significantly in first few epochs
3. **Consider learning rate**: May need adjustment after fixing data issues

## Why Performance Degraded
The combination of wrong normalization and absolute poses made the task nearly impossible:
1. Features extracted with wrong normalization don't match what the model expects
2. Predicting absolute world positions requires memorizing the entire trajectory
3. The loss values (100+) indicate the model outputs are completely off scale

These fixes should bring your performance much closer to the original VIFT results.