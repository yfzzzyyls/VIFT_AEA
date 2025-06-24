#!/bin/bash

echo "Simple working solution - no pre-loading, conservative settings"
echo ""

# Kill any existing processes
pkill -f train_flownet_lstm_transformer

# Run with minimal dataset and very conservative settings
torchrun --nproc_per_node=2 train_flownet_lstm_transformer_minimal.py \
    --distributed \
    --use-amp \
    --variable-length \
    --data-dir ../aria_processed \
    --batch-size 1 \
    --num-workers 0 \
    --learning-rate 1e-4 \
    --num-epochs 100 \
    --min-seq-len 10 \
    --max-seq-len 15 \
    --stride 20 \
    --use-curriculum \
    --checkpoint-dir checkpoints/exp_704_2gpu_simple \
    --experiment-name flownet_lstm_704_simple \
    --image-height 704 \
    --image-width 704