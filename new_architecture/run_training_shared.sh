#!/bin/bash

echo "FlowNet-CNN-Transformer Training with Memory-Mapped Shared Data"
echo "Configuration:"
echo "- Visual Encoder: FlowNet (optical flow for motion estimation)"
echo "- IMU Encoder: 1D CNN (processes all ~50 IMU samples between frames)"
echo "- Loss: Smooth L1 for translation and scale, geodesic for rotation"
echo ""

# Install required packages if not already installed
pip install filelock psutil --quiet

# Kill any existing training
pkill -f train_flownet_lstm_transformer

# Clear shared memory cache if exists
echo "Clearing any existing shared memory cache..."
rm -rf /dev/shm/aria_mmap_cache

# Set environment variables for optimal performance
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training with shared memory
# Note: We use the new script that handles multiprocessing internally
python train_flownet_lstm_transformer_shared.py \
    --distributed \
    --use-amp \
    --data-dir ../aria_processed \
    --encoder-type flownet \
    --imu-encoder-type cnn \
    --warmup-epochs 2 \
    --batch-size 4 \
    --num-workers 0 \
    --learning-rate 1e-3 \
    --num-epochs 100 \
    --sequence-length 21 \
    --stride 5 \
    --translation-weight 10.0 \
    --rotation-weight 100.0 \
    --scale-weight 20.0 \
    --checkpoint-dir ../checkpoints_from_scratch \
    --experiment-name flownet_cnn_smoothl1_allsamples \
    --validate-every 1