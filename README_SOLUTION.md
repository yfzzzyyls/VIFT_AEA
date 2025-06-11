# VIFT Model Training Issues and Solutions

## Problem Summary

The VIFT model is predicting straight lines instead of following natural motion curves. After thorough investigation, we found multiple issues:

### 1. **Systematic Bias in Predictions**
- Model outputs nearly constant relative poses regardless of input
- Bias magnitude: ~16.5 cm/frame (primarily in Y-axis: -16.4 cm)
- This causes linear drift when accumulated over time

### 2. **Training Data Issues**
- Training data has extremely small motions (0.11 cm/frame average)
- Expected walking speed motion: ~5 cm/frame
- This 50x difference suggests data was generated with large stride (30 frames)
- Model learned to predict near-zero motion with a bias

### 3. **Temporal Mismatch**
- Model trained on 30-frame intervals (1.5 seconds)
- Inference expects 1-frame intervals (50ms)
- This 30x temporal mismatch amplifies prediction errors

## Solutions

### Solution 1: Retrain with Bias Correction (Recommended)

```bash
# Train with bias-aware loss function
python train_fixed_bias.py \
    --data-dir aria_latent_data_cm \
    --batch-size 32 \
    --num-epochs 50 \
    --learning-rate 5e-5 \
    --rotation-weight 100.0 \
    --bias-weight 0.1 \
    --variance-weight 0.1
```

This uses a modified loss function that:
- Penalizes systematic bias in predictions
- Encourages prediction variance
- Adds data augmentation to prevent directional overfitting

### Solution 2: Regenerate Training Data with Consecutive Frames

```bash
# Generate proper frame-to-frame training data
python regenerate_training_data_fixed.py

# Train on consecutive frames
python train_fixed_bias.py \
    --data-dir aria_latent_data_consecutive \
    --batch-size 32 \
    --num-epochs 50
```

This creates training data with:
- Consecutive frames (stride=1) for realistic motion
- Proper relative poses between adjacent frames
- Motion magnitudes around 5 cm/frame

### Solution 3: Post-processing Bias Correction

For immediate testing without retraining:

```python
# In inference, subtract the learned bias
bias = np.array([-0.214, -16.411, 1.474])  # cm
corrected_prediction = prediction - bias
```

## Inference Commands

### For Current Model (Temporary Fix)
```bash
# Run with matched stride
python inference_full_sequence.py \
    --sequence-id 016 \
    --checkpoint fixed_scale_v1/epoch_epoch=024_val_loss_val/total_loss=14.812485.ckpt \
    --stride 30 \
    --batch-size 64
```

### For Retrained Model
```bash
# After retraining with bias correction
python inference_full_sequence.py \
    --sequence-id 016 \
    --checkpoint [new_checkpoint_path] \
    --stride 1 \
    --batch-size 64
```

## Visualization

```bash
# Generate 3D trajectory plots
python plot_short_term_trajectory.py \
    --npz-file inference_results_realtime_seq_016_stride_1.npz \
    --output-dir short_term_plots_fixed \
    --duration 5
```

## Expected Results After Fix

- Translation error: < 5 cm/frame (currently 418 cm)
- Rotation error: < 2 degrees/frame
- Predictions should follow ground truth curves
- No systematic drift in any direction

## Root Cause Summary

The model was trained on data with:
1. 30-frame subsampling (not consecutive frames)
2. Very small relative motions between these sparse frames
3. This led to mode collapse where the model outputs constant predictions

The fix requires either:
1. Retraining with proper bias-aware loss
2. Regenerating data with consecutive frames
3. Both approaches together for best results