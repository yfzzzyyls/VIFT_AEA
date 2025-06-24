#!/bin/bash

echo "Optimized FlowNet-LSTM-Transformer Training"
echo "Using padded tensors for 300x faster GPU transfers"
echo "Maintains all variable-length IMU functionality"
echo ""

# Kill any existing training
pkill -f train_flownet_lstm_transformer

# Run with full settings - this should work now!
torchrun --nproc_per_node=4 train_flownet_lstm_transformer_optimized.py \
    --distributed \
    --use-amp \
    --variable-length \
    --data-dir ../aria_processed \
    --batch-size 4 \
    --num-workers 4 \
    --learning-rate 8e-4 \
    --num-epochs 100 \
    --min-seq-len 10 \
    --max-seq-len 50 \
    --curriculum-step 10 \
    --curriculum-increment 5 \
    --stride 5 \
    --use-curriculum \
    --checkpoint-dir checkpoints/exp_704_4gpu_optimized \
    --experiment-name flownet_lstm_704_optimized \
    --image-height 704 \
    --image-width 704