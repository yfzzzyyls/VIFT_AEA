# Detailed Attention Mechanism Explanation

## Architecture Overview

The model has THREE separate transformers:

1. **Shared Transformer** (in `PoseTransformer`)
2. **Rotation Transformer** (in `RotationSpecializedHead`)  
3. **Translation Transformer** (in `TranslationSpecializedHead`)

## Code Locations

All code is in `/home/external/VIFT_AEA/src/models/multihead_vio.py`:
- Lines 17-103: `RotationSpecializedHead`
- Lines 106-205: `TranslationSpecializedHead`
- Lines 208-464: `MultiHeadVIOModel` (main model)

Shared transformer is in `/home/external/VIFT_AEA/src/models/components/pose_transformer_new.py`

## Detailed Attention Mechanisms

### 1. Shared Transformer (SELF-ATTENTION)

**Location**: `pose_transformer_new.py`, lines 89-93
```python
transformer_output = self.transformer_encoder(
    projected_features,
    mask=mask,
    is_causal=self.use_causal_mask and mask is None
)
```

**Attention Type**: SELF-ATTENTION
- **Q (Query)**: projected_features[i] (each position)
- **K (Key)**: projected_features (all positions)
- **V (Value)**: projected_features (all positions)
- **What it does**: Each timestep can attend to all other timesteps to build contextual understanding

### 2. Rotation Transformer (SELF-ATTENTION)

**Location**: `multihead_vio.py`, line 80
```python
# Apply rotation-specific attention with causal mask
attended_features = self.angular_transformer(rot_features)
```

**Attention Type**: SELF-ATTENTION (standard TransformerEncoder)
- **Q (Query)**: rot_features[i] (each position)
- **K (Key)**: rot_features (all positions)
- **V (Value)**: rot_features (all positions)
- **What it does**: Refines rotation features by letting each timestep attend to others

### 3. Translation Transformer (TWO STAGES!)

This is the complex one - it has TWO attention mechanisms:

#### Stage 1: Cross-Attention (lines 177-179)
```python
# Apply spatial cross-attention
attended_features, _ = self.spatial_cross_attention(
    trans_features, trans_features, trans_features
)
```

**IMPORTANT**: Despite the name "cross_attention", this is actually SELF-ATTENTION because all three inputs are the same!
- **Q (Query)**: trans_features
- **K (Key)**: trans_features (same as Q!)
- **V (Value)**: trans_features (same as Q!)
- **What it does**: This is self-attention, not cross-attention

#### Stage 2: Self-Attention (line 185)
```python
# Apply translation transformer
refined_features = self.translation_transformer(attended_features)
```

**Attention Type**: SELF-ATTENTION (standard TransformerEncoder)
- **Q (Query)**: attended_features[i] (each position)
- **K (Key)**: attended_features (all positions)
- **V (Value)**: attended_features (all positions)
- **What it does**: Further refines the translation features

## Visual Flow Diagram

```
Input Features (768-dim)
        |
        v
[Shared Transformer]
    Q = K = V = features
    SELF-ATTENTION
        |
        v
Shared Features (256-dim)
        |
    +---+---+
    |       |
    v       v
[Rotation] [Translation]
    |       |
    |       +-> Stage 1: spatial_cross_attention
    |           Q = K = V = trans_features
    |           (Actually SELF-ATTENTION!)
    |           |
    |           v
    |       +-> Stage 2: translation_transformer  
    |           Q = K = V = attended_features
    |           SELF-ATTENTION
    |           |
    v           v
[angular_transformer] [Output]
Q = K = V = rot_features    |
SELF-ATTENTION              |
    |                       |
    v                       v
4D Quaternion           3D Position
```

## Summary

**ALL transformers use SELF-ATTENTION!** 
- There is NO true cross-attention in this architecture
- The `spatial_cross_attention` is misnamed - it's actually self-attention
- Each transformer lets positions attend to other positions in the same sequence

## Output Dimensions

The final output dimensions are determined by the last linear layers:
- **Rotation**: `nn.Linear(hidden_dim // 2, 4)` → 4D quaternion
- **Translation**: `nn.Linear(hidden_dim // 2, 3)` → 3D position

You could change these to output Euler angles (3D) instead of quaternions (4D) by simply changing the output dimension in the linear layer.