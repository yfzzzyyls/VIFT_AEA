#!/bin/bash

echo "FlowNet-LSTM-Transformer Training with Memory-Mapped Shared Data"
echo "Using true shared memory across all processes"
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

# Run training with shared memory
# Note: We use the new script that handles multiprocessing internally
python train_flownet_lstm_transformer_shared.py \
    --distributed \
    --use-amp \
    --data-dir ../aria_processed \
    --encoder-type resnet18 \
    --pretrained \
    --batch-size 16 \
    --num-workers 0 \
    --learning-rate 1.2e-3 \
    --num-epochs 100 \
    --sequence-length 41 \
    --stride 5 \
    --translation-weight 1.0 \
    --rotation-weight 100.0 \
    --scale-weight 10.0 \
    --checkpoint-dir ../checkpoints_from_scratch \
    --experiment-name resnet_lstm_41frames_mmap_shared \
    --validate-every 1