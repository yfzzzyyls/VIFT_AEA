# FlowNet-LSTM-Transformer Architecture v1.0

A new visual-inertial odometry architecture that replaces VIFT with:
- **FlowNet** for motion-specific visual feature extraction
- **LSTM** for processing variable-length IMU sequences
- **Transformer** for multi-modal fusion and pose prediction

## Key Features

1. **No Fixed Window Size**: Process sequences of any length (5-50 frames)
2. **All IMU Data**: Uses all ~50 IMU samples between frames (Aria: 1000Hz IMU / 20Hz camera)
3. **All Frames**: Processes ALL frames from sequences (no subsampling) for best temporal resolution
4. **End-to-End Learning**: All components trained jointly on Aria dataset
5. **Step-wise Curriculum**: Gradually increases sequence length every 10 epochs
6. **Flexible Resolution**: Default 704×704 (2×2 binned from 1408×1408)

## Architecture Overview

```
Images[t] → FlowNet → Motion Features
    ↓                      ↓
IMU[t→t+1] → LSTM → Temporal Features → Transformer → Poses[t→t+1]
```

## Quick Start

### 1. Process Aria Data (if not already done)
```bash
# Default: Process with 704×704 resolution (2×2 binned from 1408×1408)
cd /home/external/VIFT_AEA
python process_aria.py \
    --input-dir /mnt/ssd_ext/incSeg-data/aria_everyday \
    --output-dir aria_processed

# Note: The script automatically uses 2×2 binning for better quality
# First bins 1408×1408 → 704×704, then resizes to final resolution
```

### 2. Training Commands

#### Recommended: Multi-GPU Training with 4× A6000
```bash
cd /home/external/VIFT_AEA/new_architecture

# Default training with 704×704 images (best quality/speed tradeoff)
./train_distributed.sh

# OR manually:
torchrun --nproc_per_node=4 train_flownet_lstm_transformer.py \
    --distributed \
    --use-amp \
    --variable-length \
    --batch-size 6 \
    --learning-rate 4e-4 \
    --num-epochs 100 \
    --min-seq-len 10 \
    --max-seq-len 50 \
    --curriculum-step 10 \
    --curriculum-increment 5 \
    --stride 5 \
    --use-curriculum \
    --checkpoint-dir checkpoints/exp_704_4gpu \
    --experiment-name flownet_lstm_704
```

#### Alternative: 512×512 Resolution (1.9× faster, slightly lower quality)
```bash
torchrun --nproc_per_node=4 train_flownet_lstm_transformer.py \
    --distributed \
    --use-amp \
    --variable-length \
    --batch-size 8 \
    --image-height 512 \
    --image-width 512 \
    --learning-rate 4e-4 \
    --num-epochs 100 \
    --min-seq-len 10 \
    --max-seq-len 50 \
    --curriculum-step 10 \
    --curriculum-increment 5 \
    --stride 5 \
    --use-curriculum \
    --checkpoint-dir checkpoints/exp_512_4gpu \
    --experiment-name flownet_lstm_512
```

#### Single GPU Training (not recommended)
```bash
python train_flownet_lstm_transformer.py \
    --use-amp \
    --variable-length \
    --batch-size 8 \
    --image-height 512 \
    --image-width 512 \
    --min-seq-len 10 \
    --num-epochs 100
```

### 3. Monitor Training
```bash
# In another terminal
watch -n 1 nvidia-smi

# Check training progress
tail -f checkpoints/exp_512_4gpu/train.log
```

## Key Training Arguments

- `--image-height/width`: Resolution (512 or 704 recommended)
- `--batch-size`: Batch size per GPU (8 for 512×512, 6 for 704×704)
- `--learning-rate`: Base LR (scale by num_gpus, e.g., 4e-4 for 4 GPUs)
- `--min-seq-len`: Starting sequence length (default: 10 frames = 0.5s)
- `--curriculum-step`: Epochs between sequence length increases (default: 10)
- `--curriculum-increment`: Frames to add each step (default: 5)
- `--stride`: Sliding window stride (default: 5)

## Model Configuration

The model has three main components:

### 1. FlowNet Motion Encoder
- Processes consecutive image pairs to extract motion
- Uses correlation layer to compute pixel correspondences
- Output: 256-dimensional motion features

### 2. IMU LSTM Encoder
- Processes ALL IMU samples between frames (~50 per interval)
- Bidirectional LSTM with 3 layers
- Handles variable-length sequences naturally
- Output: 256-dimensional temporal features

### 3. Pose Transformer
- Fuses visual and IMU features
- 6-layer transformer with causal attention
- Predicts relative poses (3 translation + 4 quaternion)

## Dataset

The `AriaVariableIMUDataset` supports:
- Variable sequence lengths
- All raw IMU data (~50 samples per frame interval)
- Proper temporal alignment
- Efficient batching with custom collate function

## Loss Functions

- **Translation Loss**: MSE on predicted translations
- **Rotation Loss**: Geodesic distance for quaternions
- **Scale Consistency**: Prevents scale drift
- **Temporal Smoothness**: Encourages smooth trajectories

## Evaluation Metrics

- **ATE**: Absolute Trajectory Error
- **RPE**: Relative Pose Error (at different scales)
- **Scale Error**: Trajectory scale accuracy
- **Drift Rate**: End-to-end drift percentage

## File Structure

```
new_architecture/
├── models/
│   └── flownet_lstm_transformer.py    # Main model architecture
├── data/
│   └── aria_variable_imu_dataset.py   # Dataset with variable IMU
├── configs/
│   └── flownet_lstm_transformer_config.py  # Configuration
├── utils/
│   ├── losses.py                       # Loss functions
│   ├── metrics.py                      # Evaluation metrics
│   └── visualization.py                # Plotting utilities
├── train_flownet_lstm_transformer.py   # Training script
└── README.md                           # This file
```

## Example Usage

```python
from models import FlowNetLSTMTransformer
from configs import get_config

# Load configuration
config = get_config()

# Create model
model = FlowNetLSTMTransformer(config.model)

# Forward pass
# images: [B, T, 3, H, W]
# imu_sequences: List of B lists, each with T-1 variable-length IMU tensors
outputs = model(images, imu_sequences)

# Outputs contain:
# - poses: [B, T-1, 7] (3 trans + 4 quat)
# - translation: [B, T-1, 3]
# - rotation: [B, T-1, 4]
```

## Resolution Comparison

| Resolution | Memory/GPU | Training Speed | Quality | Recommendation |
|------------|------------|----------------|---------|----------------|
| 512×512    | ~28GB      | 1.9× faster    | Good    | ✅ For experiments |
| 704×704    | ~38GB      | 1.0× (baseline)| Better  | ✅ **Default** |
| 1408×1408  | ~45GB+     | 7.5× slower    | Best    | ❌ Too slow |

**Why 704×704 as default?**
- Aria's native resolution is 1408×1408
- 704×704 = 2×2 binning (preserves detail better than resizing to 512)
- Best quality/speed tradeoff for production models
- Only 1.9× slower than 512×512 but noticeably better quality

## Curriculum Learning Schedule

```
Epochs   1-10:  10 frames (0.50s)  ← Start simple
Epochs  11-20:  15 frames (0.75s)
Epochs  21-30:  20 frames (1.00s)
Epochs  31-40:  25 frames (1.25s)
Epochs  41-50:  30 frames (1.50s)  ← AR/VR typical
Epochs  51-60:  35 frames (1.75s)
Epochs  61-70:  40 frames (2.00s)
Epochs  71-80:  45 frames (2.25s)
Epochs  81-90:  50 frames (2.50s)  ← Maximum
```

## Evaluation

```bash
# Evaluate on test set (704×704 model)
python evaluate_flownet_lstm_transformer.py \
    --checkpoint checkpoints/exp_704_4gpu/best_model.pt \
    --data-dir aria_processed \
    --output-dir evaluation/exp_704_4gpu \
    --batch-size 6 \
    --sequence-length 31 \
    --image-height 704 \
    --image-width 704
```

## Advantages Over VIFT

1. **No Information Loss**: Uses all ~50 IMU samples per interval, not just 11
2. **Flexible Sequences**: Can process any length, not fixed to 11 frames
3. **Better Motion Features**: FlowNet explicitly models optical flow
4. **Temporal Modeling**: LSTM captures IMU dynamics better than CNN
5. **Attention Mechanism**: Transformer learns when to trust each modality

## Tips for Best Results

1. **Resolution Choice**:
   - Start with 512×512 for faster experimentation
   - Use 704×704 for final model if quality is critical
   
2. **Batch Size Tuning**:
   - 512×512: batch_size=8-12 per GPU
   - 704×704: batch_size=6 per GPU (default)
   - Use gradient accumulation if needed

3. **Learning Rate**:
   - Scale linearly with GPUs: 1e-4 × num_gpus
   - Use warmup for first 5 epochs

4. **Monitoring**:
   - Watch GPU memory usage in first epoch
   - Adjust batch size if OOM
   - Check scale drift in validation

## Complete Workflow

### Step 1: Data Preparation

```bash
# 1.1 Process Aria VRS files to extract images, IMU, and poses
cd /home/external/VIFT_AEA

# Process ALL frames for best quality (no subsampling)
python process_aria.py \
    --input-dir /mnt/ssd_ext/incSeg-data/aria_everyday \
    --output-dir aria_processed \
    --num-workers 4

# Note: This processes ALL frames from each sequence
# - Ensures ~50 IMU samples between consecutive frames (20Hz camera, 1000Hz IMU)
# - Processing time: ~5-10 minutes per sequence depending on length
# - Total dataset size: ~50-100GB for 20 sequences with all frames

# Expected output structure:
# aria_processed/
# ├── 000/
# │   ├── visual_data.pt      # [N, 3, 704, 704] ALL frames
# │   ├── imu_data.pt         # List of N-1 variable-length IMU tensors (~50 samples each)
# │   └── poses_quaternion.json
# ├── 001/
# ├── ...
# ├── 019/
# └── splits.json             # train/val/test splits
```

### Step 2: Training

```bash
# 2.1 Navigate to new architecture directory
cd /home/external/VIFT_AEA/new_architecture

# 2.2 Start distributed training with 4 GPUs
./train_distributed.sh

# Monitor training progress
# Terminal 1: Watch GPU usage
watch -n 1 nvidia-smi

# Terminal 2: Monitor training logs
tail -f checkpoints/exp_704_4gpu/train.log

# Training will save:
# - Best model: checkpoints/exp_704_4gpu/best_model.pt
# - Periodic checkpoints: checkpoint_epoch_5.pt, epoch_10.pt, etc.
# - Training metrics: checkpoints/exp_704_4gpu/metrics.json
```

### Step 3: Evaluation

```bash
# 3.1 Evaluate on test set
python evaluate_flownet_lstm_transformer.py \
    --checkpoint checkpoints/exp_704_4gpu/best_model.pt \
    --data-dir ../aria_processed \
    --output-dir evaluation/exp_704_4gpu \
    --split test \
    --batch-size 6 \
    --sequence-length 31

# Outputs:
# - evaluation/exp_704_4gpu/metrics.json        # Quantitative metrics
# - evaluation/exp_704_4gpu/visualizations/     # Trajectory plots
# - evaluation/exp_704_4gpu/detailed_results.npz # Raw predictions
```

### Step 4: Inference

#### 4.1 Batch Inference on New Data
```python
# inference_example.py
import torch
from models.flownet_lstm_transformer import FlowNetLSTMTransformer
from configs.flownet_lstm_transformer_config import ModelConfig
from data.aria_variable_imu_dataset import AriaVariableIMUDataset, collate_variable_imu
from torch.utils.data import DataLoader

# Load model
config = ModelConfig()
model = FlowNetLSTMTransformer(config)
checkpoint = torch.load('checkpoints/exp_704_4gpu/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.cuda()

# Create dataset for new data
dataset = AriaVariableIMUDataset(
    data_dir='../aria_processed',
    split='test',  # or your custom split
    variable_length=False,
    sequence_length=31,
    image_size=(704, 704)
)

dataloader = DataLoader(
    dataset, 
    batch_size=1, 
    collate_fn=collate_variable_imu
)

# Run inference
with torch.no_grad():
    for batch in dataloader:
        images = batch['images'].cuda()
        imu_sequences = batch['imu_sequences']
        
        # Move IMU to GPU
        for b in range(len(imu_sequences)):
            for t in range(len(imu_sequences[b])):
                imu_sequences[b][t] = imu_sequences[b][t].cuda()
        
        # Predict poses
        outputs = model(images, imu_sequences)
        poses = outputs['poses']  # [1, 30, 7]
        
        # poses contain relative transformations
        # Format: [dx, dy, dz, qx, qy, qz, qw]
        print(f"Predicted {poses.shape[1]} relative poses")
```

#### 4.2 Real-time Inference (Sliding Window)
```python
# realtime_inference.py
import collections
import numpy as np

class RealTimeVIO:
    def __init__(self, model_path, window_size=31):
        self.window_size = window_size
        self.image_buffer = collections.deque(maxlen=window_size)
        self.imu_buffer = collections.deque(maxlen=window_size-1)
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.eval()
        
    def process_frame(self, image, imu_data):
        """Process new frame with associated IMU data."""
        # Add to buffers
        self.image_buffer.append(image)
        if len(self.image_buffer) > 1:
            self.imu_buffer.append(imu_data)
        
        # Run inference when buffer is full
        if len(self.image_buffer) == self.window_size:
            poses = self.run_inference()
            # Return the latest pose estimate
            return poses[-1]  # Most recent relative pose
        
        return None
    
    def run_inference(self):
        # Convert buffers to tensors
        images = torch.stack(list(self.image_buffer))
        images = images.unsqueeze(0)  # Add batch dimension
        
        # Prepare IMU sequences
        imu_sequences = [[imu for imu in self.imu_buffer]]
        
        # Run model
        with torch.no_grad():
            outputs = self.model(images.cuda(), imu_sequences)
            return outputs['poses'].cpu().numpy()

# Usage
vio = RealTimeVIO('checkpoints/exp_704_4gpu/best_model.pt')

# Process incoming frames
for frame, imu in data_stream:
    relative_pose = vio.process_frame(frame, imu)
    if relative_pose is not None:
        # Update absolute pose
        update_trajectory(relative_pose)
```

### Step 5: Fine-tuning (Optional)

```bash
# 5.1 Fine-tune on specific sequences or scenarios
python train_flownet_lstm_transformer.py \
    --checkpoint checkpoints/exp_704_4gpu/best_model.pt \
    --data-dir ../aria_processed_custom \
    --learning-rate 1e-5 \
    --num-epochs 20 \
    --batch-size 4 \
    --checkpoint-dir checkpoints/finetuned \
    --experiment-name flownet_lstm_finetuned
```

### Tips for Production Deployment

1. **Model Optimization**:
   ```bash
   # Convert to TorchScript for faster inference
   python export_model.py \
       --checkpoint checkpoints/exp_704_4gpu/best_model.pt \
       --output model.pt
   ```

2. **Batch Processing**:
   - Use larger batch sizes for offline processing
   - Enable CUDA graphs for consistent input sizes

3. **Memory Management**:
   - Clear gradients: `torch.cuda.empty_cache()`
   - Use half precision: `model.half()`

4. **Integration with ROS/C++**:
   - Export to ONNX for cross-platform deployment
   - Use LibTorch for C++ inference

## Future Improvements

- [ ] Add uncertainty estimation
- [ ] Implement online/streaming mode
- [ ] Add self-supervised losses
- [ ] Support for other datasets (TUM-VI, EuRoC)
- [ ] Lightweight version for edge deployment