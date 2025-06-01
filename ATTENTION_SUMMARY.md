# Self-Attention Summary

## The Core Formula

For each output timestep `i`:
```
output[i] = Σ(j=0 to seq_len) attention_weight[i,j] * V[j]
```

Where:
```
attention_weight[i,j] = softmax(Q[i] · K[j] / √d_k)
```

## In Our VIO Model

### All Three Transformers Use Self-Attention:
- **Q = K = V = input_features**
- Each timestep can "look at" all other timesteps
- The attention weights determine how much each timestep influences the others

### Simple Example with 3 Timesteps:

**Input Features:**
```
Timestep 0: [1.0, 0.5, -0.5, 1.0]
Timestep 1: [0.5, 1.0, 0.0, -0.5]  
Timestep 2: [-0.5, 0.0, 1.0, 0.5]
```

**Attention Weights Matrix:**
```
         Attending to:  T0    T1    T2
From T0:              [0.63, 0.23, 0.14]  <- T0 mostly looks at itself
From T1:              [0.31, 0.51, 0.19]  <- T1 mostly looks at itself
From T2:              [0.21, 0.21, 0.58]  <- T2 mostly looks at itself
```

**Output for Timestep 0:**
```
output[0] = 0.63 * features[0] + 0.23 * features[1] + 0.14 * features[2]
          = weighted combination of all timesteps
```

## Why Self-Attention for VIO?

1. **Temporal Context**: Each pose prediction can use information from all other timesteps
2. **Motion Patterns**: The model learns which past/future frames are most relevant
3. **Smoothness**: Attention weights create smooth transitions between poses

## The Three Transformers in Our Model:

1. **Shared Transformer**: Builds general motion understanding
2. **Rotation Transformer**: Specializes in rotational patterns
3. **Translation Transformer**: Specializes in positional patterns (2-stage refinement)

All use the same self-attention mechanism, just applied to different features!