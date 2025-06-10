# Complete Guide to Apply Training Fixes

## Problem Summary
Your model has collapsed to outputting constant values:
- Constant translation: 0.000115m per frame
- Constant rotation: 0.064° per frame  
- Trajectories go in opposite directions due to rotation drift

## Files Created for Fixes

### 1. **train_fixed.py** - Enhanced training script
- Monitors for model collapse
- Adds diversity loss to prevent constant outputs
- Integrates RPMG loss option
- Proper validation with direction checking

### 2. **configs/experiment/fixed_training.yaml** - Training configuration
- Uses stride=20 (1 second intervals)
- Enables RPMG loss
- Sets proper data scaling (100x for cm)
- Includes augmentation settings

### 3. **src/data/aria_datamodule_fixed.py** - Enhanced data loader
- Supports configurable stride
- Proper pose scaling
- Data augmentation
- Relative pose computation

## How to Apply the Fixes

### Step 1: Verify Your Data
First, check that larger strides give meaningful motion:

```bash
python verify_data_stride.py
```

This will show you motion magnitudes at different strides. You should see ~2cm motion at stride=20.

### Step 2: Quick Test Run
Test the setup with a small training run:

```bash
python train_fixed.py \
    data.batch_size=8 \
    trainer.max_epochs=5 \
    trainer.limit_train_batches=50 \
    trainer.limit_val_batches=20
```

Monitor the output for:
- `train/trans_std` and `train/rot_std` should be > 0.0001
- No "Model collapse detected" warnings
- `val/direction_similarity` should be close to 1.0

### Step 3: Full Training with RPMG
Run the full training with all fixes:

```bash
python train_fixed.py experiment=fixed_training
```

Or with custom settings:

```bash
python train_fixed.py \
    experiment=fixed_training \
    stride=20 \
    use_rpmg=true \
    angle_weight=100 \
    diversity_weight=0.1 \
    data.batch_size=32 \
    trainer.accumulate_grad_batches=4
```

### Step 4: Monitor Training

Watch for these key metrics:
1. **Diversity Metrics**:
   - `train/trans_std` > 0.001
   - `train/rot_std` > 0.001

2. **Direction Similarity**:
   - `val/direction_similarity` > 0.5 (ideally close to 1.0)

3. **Loss Convergence**:
   - `train/pose_loss` decreasing
   - `train/diversity_loss` stable

### Step 5: Update Your Main Training Script

If you want to integrate these fixes into your existing training:

```python
# In train_improved.py, add:

# 1. Import the fixed data module
from src.data.aria_datamodule_fixed import AriaDataModuleFixed

# 2. Use larger stride
cfg.data.stride = 20  # or from config

# 3. Add diversity monitoring
def check_prediction_diversity(predictions):
    trans_std = torch.std(predictions[:, :3], dim=0).mean()
    rot_std = torch.std(predictions[:, 3:], dim=0).mean()
    
    if trans_std < 1e-5 or rot_std < 1e-5:
        print(f"WARNING: Low diversity - trans_std={trans_std:.6f}, rot_std={rot_std:.6f}")
    
    return trans_std, rot_std

# 4. Switch to RPMG loss (optional but recommended)
from src.metrics.weighted_loss import RPMGPoseLoss
loss_fn = RPMGPoseLoss(angle_weight=100)
```

## Comparing Losses: Current vs RPMG

### Current ARVRLossWrapper
- ✅ Proper quaternion geodesic distance
- ✅ Smooth L1 for robustness
- ❌ No manifold-aware gradients
- ❌ May allow drift in SO(3) space

### RPMG Loss
- ✅ Manifold-aware optimization on SO(3)
- ✅ Prevents rotation representation issues
- ✅ Better for long sequences (less drift)
- ⚠️ Slightly more compute intensive
- ⚠️ May need tuning of τ and λ parameters

**Recommendation**: Use RPMG for this task since rotation drift is a major issue.

## Expected Results After Fixes

1. **Predictions vary with input**:
   - Translation: 0.5-5.0 cm per frame (at stride=20)
   - Rotation: 0.5-3.0° per frame

2. **Correct trajectory direction**:
   - Direction similarity > 0.8
   - No 180° flips

3. **Better unaligned metrics**:
   - Unaligned ATE < 10cm (vs 40cm currently)
   - No need for aggressive Umeyama correction

## Troubleshooting

If model still collapses:
1. Increase `diversity_weight` to 0.2-0.5
2. Try even larger stride (30-40)
3. Reduce batch size and learning rate
4. Check if pretrained encoder is frozen properly

If RPMG loss is unstable:
1. Reduce `angle_weight` to 50
2. Adjust RPMG τ parameter (default 0.25)
3. Use gradient clipping (already set to 1.0)

## Next Steps

After successful training:
1. Run inference with the fixed model
2. Check trajectories WITHOUT Umeyama alignment
3. Verify direction is correct
4. Fine-tune on stride=10 or stride=5 for better temporal resolution