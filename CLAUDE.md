# VIFT-AEA Project Context for Claude

## Project Overview
This is the Visual-Inertial Fusion Transformer (VIFT) implementation extended for both KITTI and Aria Everyday Activities datasets. The model uses a causal transformer architecture for visual-inertial odometry.

## Current Implementation Status

### Model Architecture (Default Configuration)
1. **Visual Encoder**: 6-layer CNN (default) for frame-to-frame optical flow estimation
   - Input: Consecutive frame pairs concatenated along channel dimension
   - Output: 256-dimensional visual features per frame transition
2. **IMU Encoder**: FlexibleInertialEncoder with variable-length support
   - Architecture: 3-layer 1D CNN with multi-scale temporal pooling
   - Handles variable number of IMU samples between consecutive frames (~50 samples at 1000Hz)
   - Multi-scale pooling: Local (8), Mid (4), Global (1) adaptive average pooling
   - Output: 256-dimensional IMU features per frame transition
3. **Pose Transformer**: 8 layers, 16 heads, 4096 FFN dimension (default)
   - Configurable via: `--transformer-layers`, `--transformer-heads`, `--transformer-dim-feedforward`
   - Input: Concatenated visual + IMU features [512 dims]
   - Output: 7-DoF poses (3 translation + 4 rotation quaternion)

### Key Implementation Details
- **Output**: Quaternion representation (3 translation + 4 rotation)
- **Loss**: Weighted combination with α=10.0 (translation), β=20.0 (scale), 100.0 (rotation)
- **IMU**: Variable-length sequences between consecutive frames [t_i, t_{i+1})
  - All IMU samples preserved (no downsampling from 1000Hz)
  - Typically ~50 samples per frame interval at 20Hz camera rate
- **Multi-GPU**: Both Aria and TUM VI support distributed training with torchrun

### Hardware Setup
- 4x NVIDIA A6000 GPUs (48GB each)
- Optimized batch sizes for this configuration

### Datasets
1. **Aria Everyday Activities**: Located in `aria_processed/`
   - RGB-D egocentric dataset with Project Aria glasses
   - 1000 Hz IMU, 20 Hz RGB images (corrected from initial 30Hz assumption)
   
2. **TUM VI**: Located in `/mnt/ssd_ext/incSeg-data/tumvi/`
   - All corridor sequences (1-5) are extracted and ready
   - Training uses: room1-4 + corridor1-3
   - Validation: room5 + corridor4
   - Test: room6 + corridor5
   - Stereo cameras, 200 Hz IMU

Note: Aria and TUM VI are separate datasets with different sensors and characteristics. They are trained and evaluated independently.

### Training Commands (with 4x A6000)
```bash
# Aria - Distributed training with large transformer
torchrun --nproc_per_node=4 train_aria_from_scratch.py \
    --data-dir aria_processed \
    --epochs 50 \
    --batch-size 64 \
    --checkpoint-dir checkpoints_from_scratch \
    --distributed

# TUM VI - Distributed training
torchrun --nproc_per_node=4 train_on_tumvi.py \
    --data-dir /mnt/ssd_ext/incSeg-data/tumvi \
    --epochs 30 \
    --batch-size 24 \
    --checkpoint-dir checkpoints_tumvi \
    --distributed
```

### Evaluation Commands
```bash
# Aria evaluation
python evaluate_from_scratch.py \
    --checkpoint checkpoints_from_scratch/best_model.pt \
    --data-dir aria_processed \
    --output-dir evaluation_aria

# TUM VI evaluation (on complete test set)
for seq in room6 corridor5; do
    python evaluate_tumvi_standalone.py \
        --checkpoint checkpoints_tumvi/best_model.pt \
        --sequence-dir /mnt/ssd_ext/incSeg-data/tumvi/dataset-${seq}_512_16 \
        --output-dir evaluation_tumvi/test_${seq}
done
```

### Recent Changes & Decisions
1. FlowNet-C is now the default encoder (better motion estimation)
2. Extra large transformer (8L, 16H) is default for best accuracy
3. Both training scripts support distributed training with torchrun
4. Batch sizes optimized for 4x A6000 GPUs
5. All TUM VI corridor sequences are extracted and included in training

### Important Files
- `train_aria_from_scratch.py`: Main Aria training script with DDP support
- `train_on_tumvi.py`: TUM VI training script with DDP support
- `src/models/components/vsvio.py`: Core model implementation
- `src/models/components/flownet_encoder.py`: FlowNet-C implementation

### Common Tasks
- To use smaller transformer: Add `--transformer-layers 4 --transformer-heads 8`
- To use CNN encoder: Add `--encoder-type cnn`
- To train on single GPU: Remove `torchrun` and `--distributed`
- To increase batch size: Adjust `--batch-size` (monitor GPU memory)

### Notes for Future Development
- The model supports variable sequence lengths through the transformer
- Checkpoint compatibility: Models trained with different encoders need matching encoder at inference
- Scale drift is addressed through explicit scale consistency loss
- IMU bias is handled through per-window bias removal

## Important Findings

### Train vs Eval Metric Mismatch (2025-01-25)
**Issue**: Training shows 0.2mm translation loss but evaluation shows 47.08cm error - a 2000x difference!

**Root Cause Analysis**:
1. **Training metrics show per-frame relative pose errors**:
   - Translation: ~0.2cm (displayed as 0.002m in training)
   - Rotation: ~2.48° (raw loss before weighting)

2. **Evaluation metrics show integrated trajectory errors**:
   - Small per-frame errors compound dramatically over long sequences
   - Example from sequence 016:
     - Per-frame: 2.30cm translation, 3.02° rotation
     - After 10 frames: 14.78cm error
     - After 100 frames: 151.34cm error  
     - After 1920 frames: 1171.03cm (11.7 meters!)

3. **Model learned systematic bias**:
   - Predictions are nearly constant: `[0.0061382, -0.01227435, 0.00769854]`
   - This bias accumulates catastrophically during trajectory integration

**Contributing Factors**:
- Training on only first 500 frames led to overfitting
- High rotation weight (1000.0) may have caused imbalanced learning
- Model predicts nearly constant relative poses regardless of input

**Key Insight**: The evaluation uses AriaDatasetMMapShared (same as training) so dataset differences are NOT the issue. The problem is fundamental - small biases in relative pose predictions accumulate dramatically when integrated over long trajectories.

## Changelog & Evolution

### 2025-01-25 (Latest)
- **Updated**: CLAUDE.md to reflect current FlexibleInertialEncoder implementation
- **Clarified**: Current architecture uses 1D CNN with multi-scale pooling, not LSTM
- **Removed**: Outdated "New Architecture" section about LSTM implementation
- **Key insight**: Current implementation already supports variable-length IMU sequences efficiently
- **Analyzed**: Faulty FlowNet-C implementation in VIFT_AEA_MAIN (see below)
- **Planned**: SEA-RAFT feature encoder to replace 6-layer CNN (see implementation plan below)
- **Fixed**: Train vs eval metric mismatch investigation
- **Changed**: Evaluation script to always use AriaDatasetMMapShared (removed AriaVariableIMUDataset)
- **Added**: Debug mode showing per-frame vs integrated errors
- **Discovered**: Model predicts nearly constant poses due to training on limited data

### 2025-01-23
- **Added**: Distributed training support for TUM VI dataset
- **Changed**: Default transformer from 4L/8H to 8L/16H for better accuracy
- **Changed**: Default encoder from CNN to FlowNet-C for better motion estimation
- **Optimized**: Batch sizes for 4x A6000 GPUs (32 for both Aria and TUM VI with DDP)
- **Fixed**: Extracted all TUM VI corridor sequences for training
- **Fixed**: Evaluation now correctly uses complete test set (room6 + corridor5)
- **Added**: `aggregate_tumvi_results.py` to compute weighted test set metrics
- **Optimized**: TUM VI training stride from 1 to 5 (default) for faster training
- **Fixed**: Distributed training checkpoint saving bug (indentation issue)
- **CRITICAL FIX**: TUM VI training was extremely slow (5s/batch) due to PNG loading
  - Created `preprocess_tumvi_images.py` to convert PNGs to numpy arrays
  - Created `tumvi_fast_dataset.py` to load preprocessed data
  - **10x speedup**: from 5s/batch to 0.5s/batch
  - **ALWAYS preprocess TUM VI data before training!**

### Deprecated/Outdated Information
- ~~Small batch sizes (12-20) due to memory constraints~~ → Now using 64+ with A6000s
- ~~CNN as default encoder~~ → FlowNet-C is now default
- ~~TUM VI missing corridor sequences~~ → All sequences extracted

### TODO/Questions for Next Session
- [ ] Experiment results comparing FlowNet-C vs CNN encoder
- [ ] Optimal learning rate scaling for large batch sizes
- [ ] Performance metrics on both datasets with new configuration
- [x] Create aggregate evaluation script that combines metrics from multiple test sequences ✓
- [x] Add overall test set metrics computation (weighted by sequence length) ✓
- [ ] Update evaluate_tumvi_standalone.py to save metrics in consistent JSON format

## Current Implementation Details

### FlexibleInertialEncoder (2025-01-25)
The current VIFT implementation uses a **FlexibleInertialEncoder** that handles variable-length IMU sequences:

#### Key Features
- **Variable-length support**: Processes different number of IMU samples between frames
- **Multi-scale temporal pooling**: Captures motion at different temporal resolutions
  - Local features: AdaptiveAvgPool1d(8) - fine-grained motion details
  - Mid-level features: AdaptiveAvgPool1d(4) - medium-scale patterns  
  - Global features: AdaptiveAvgPool1d(1) - overall motion statistics
- **3-layer 1D CNN backbone**: Maintains computational efficiency while handling variable lengths

#### Architecture Details
```python
# Input: [batch, seq_len, num_samples, 6] where num_samples varies (~50 at 20Hz)
1. Conv1d layers: 6→64→128→256 channels with BatchNorm and LeakyReLU
2. Multi-scale pooling: Adaptive pooling to fixed sizes (8, 4, 1)
3. Feature fusion: Concatenate multi-scale features (256*13 dims)
4. Output projection: Linear layers 256*13→512→256
```

### Data Format
- **AriaRawDataset**: Handles variable-length IMU sequences
- **IMU data structure**: Each transition contains all IMU samples between frames
- **Collation**: Uses `pad_sequence` for batching variable-length sequences

### Important Files
- `src/models/components/vsvio.py`: Contains FlexibleInertialEncoder implementation
- `src/data/components/aria_raw_dataset.py`: Dataset with variable IMU support
- `train_aria_from_scratch.py`: Main training script using current architecture

### Performance Expectations
- Training: ~100-150 sequences/second with 4x A6000
- Memory: ~28GB per GPU (room for larger batches)
- Inference: ~50-100 FPS single sequence

## Evaluation Output Requirements

### IMPORTANT: For ALL evaluations, the following outputs MUST be generated:
1. **3D Trajectory Plots** (both 1s and 5s windows):
   - `trajectory_3d_XXX_1s.png` - 3D trajectory comparison for 1 second
   - `trajectory_3d_XXX_5s.png` - 3D trajectory comparison for 5 seconds
   
2. **3D Rotation Plots** (both 1s and 5s windows):
   - `rotation_3d_XXX_1s.png` - 3D rotation visualization for 1 second
   - `rotation_3d_XXX_5s.png` - 3D rotation visualization for 5 seconds
   
3. **CSV Output**:
   - `trajectory_XXX_pred.csv` - Predicted trajectory data for further analysis

These outputs match the format in `~/VIFT_AEA/evaluation_from_scratch/` and are essential for comparing results across different models and configurations.

## Failed FlowNet-C Implementation Analysis (2025-01-25)

### Location
`VIFT_AEA_MAIN/src/models/components/flownet_encoder.py`

### Critical Issues Found

#### 1. **Incorrect Correlation Implementation**
```python
# WRONG: Computes dot product without proper normalization
corr = torch.sum(x1 * x2_shift, dim=1, keepdim=True)
# ...
correlation = correlation / C  # Normalizes AFTER accumulation
```
**Problem**: Normalization happens after accumulating all correlations, not per-correlation computation.

#### 2. **Inefficient Nested Loops**
```python
# WRONG: Python loops are extremely slow
for i, dx in enumerate(range(-self.max_displacement, self.max_displacement + 1)):
    for j, dy in enumerate(range(-self.max_displacement, self.max_displacement + 1)):
        # Creates intermediate tensors for EACH displacement
```
**Problem**: No GPU optimization, creates 441 intermediate tensors for max_displacement=20.

#### 3. **Aggressive Downsampling**
- Features downsampled 8× before correlation
- Correlation computed with stride=2 (additional 2× downsampling)
- Total: 16× downsampling loses fine motion details

#### 4. **Missing Feature Normalization**
- No L2 normalization before correlation
- Can cause gradient instability and poor convergence

#### 5. **No Learned Correlation**
- Uses raw dot products instead of learned similarity
- Standard FlowNet-C uses 1×1 convolutions for learnable correlation

### Lessons Learned

1. **Use Efficient Correlation**:
   - Implement with `F.unfold` + batch matrix multiplication
   - Or use existing optimized libraries (e.g., spatial-correlation-sampler)

2. **Preserve Resolution**:
   - Don't downsample too aggressively before correlation
   - Use pyramid processing for multi-scale

3. **Always Normalize**:
   - L2 normalize features before correlation
   - Ensures stable gradients

4. **Test Components Separately**:
   - Verify correlation produces meaningful similarity maps
   - Check gradient flow through each component

5. **Profile Performance**:
   - The nested loops likely caused significant training slowdown
   - Always benchmark custom operations

### Recommended Approach
Instead of FlowNet-C, consider:
- **PWC-Net**: More efficient pyramid warping correlation
- **RAFT**: State-of-the-art iterative refinement
- Or fix the correlation implementation with proper vectorization

## SEA-RAFT Feature Encoder Implementation Plan (2025-01-25)

### Overview
Replace the current 6-layer CNN visual encoder with SEA-RAFT's pretrained feature network to extract rich motion-aware features. This leverages SEA-RAFT's correlation-optimized features without the computational cost of full optical flow.

### Key Design Decisions

1. **Use Internal Features, Not Flow**
   - Extract SEA-RAFT's fnet features (256 channels at 1/8 resolution)
   - Compute motion via feature difference (feat2 - feat1)
   - Avoids expensive iterative flow refinement

2. **Memory Optimization**
   - Use only highest resolution pyramid (1/8 scale)
   - Add 256→128 bottleneck conv to reduce channels
   - Critical for 704×704 images on A6000 GPUs

3. **Spatial Token Preservation**
   - Keep 4×4 spatial tokens instead of global pooling
   - Enables transformer to reason about parallax
   - Output: 16 spatial tokens × 256 dims → 256 final features

### Implementation

```python
class SEARAFTFeatureEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # Load pretrained SEA-RAFT
        model = torch.hub.load('princeton-vl/SEA-RAFT', 'searaft')
        self.fnet = model.fnet
        self.fnet.eval()
        
        # Input normalization for RAFT (expects [-1,1] BGR)
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], 
            std=[0.5, 0.5, 0.5]
        )
        
        # Bottleneck to reduce channels (256→128)
        self.bottleneck = nn.Conv2d(256, 128, 1, 1, 0)
        
        # Motion encoder preserving spatial structure
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),  # Downsample
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # → [B, 256, 4, 4]
        )
        
        # Output projection
        self.output_proj = nn.Linear(256 * 16, opt.v_f_len)
        
    def forward(self, x):
        # x: [B, 6, H, W] concatenated RGB frames
        img1 = self.normalize(x[:, :3])
        img2 = self.normalize(x[:, 3:6])
        
        # Extract features without gradients
        with torch.no_grad():
            feat1 = self.fnet(img1)[0]  # [B, 256, H/8, W/8]
            feat2 = self.fnet(img2)[0]
        
        # Motion features
        motion = self.bottleneck(feat2 - feat1)  # [B, 128, H/8, W/8]
        
        # Encode with spatial awareness
        features = self.motion_encoder(motion)  # [B, 256, 4, 4]
        features = features.flatten(1)  # [B, 4096]
        
        return self.output_proj(features)  # [B, 256]
```

### Critical Implementation Notes

1. **Normalization**: RAFT requires [-1,1] normalized BGR input (not [0,1] RGB)
2. **No Gradients**: Wrap fnet calls in torch.no_grad() to save memory
3. **Single Pyramid Level**: Use only [0] index to avoid 4× memory explosion
4. **Spatial Tokens**: 4×4 adaptive pooling preserves parallax information
5. **Testing Order**: Unit test → Memory test → Speed test → Integration test

### Expected Performance
- **Latency**: ~6ms per frame pair (vs 0.7ms for CNN)
- **Memory**: ~2GB for feature extraction at 704×704
- **Accuracy**: 10-15% ATE improvement expected
- **Training**: 3-4 hours implementation + testing

### Integration Checklist
- [ ] Replace Encoder class in vsvio.py
- [ ] Add RGB conversion if using grayscale
- [ ] Verify normalization pipeline
- [ ] Test memory usage at full resolution
- [ ] A/B test against current CNN baseline

## Memories
- Always refer to the actual code when I ask about how the current repo implemented certain functionality, or what technique does it use. Do not generalize one without any evidence.
- When I tell you to check files within certain folder, please really read that file, because the file I want you to read is crucial for the understanding of our conversation.
- Thank you for holding me accountable to provide evidence for my claims. This is an important reminder to be more careful about distinguishing between:
  - What I think I know from general knowledge
  - What I can actually verify in the specific document
- The new FlowNet-LSTM-Transformer architecture (v1.0) is a complete replacement for VIFT, not an extension
- Aria dataset is 20Hz camera frequency, not 30Hz as initially assumed
- All IMU data (~50 samples per frame) is now used without downsampling
- Do not create fallback command if not explicitly mentions. We should only follow what the requirement is.