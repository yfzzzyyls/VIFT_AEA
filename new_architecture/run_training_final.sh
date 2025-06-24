#!/bin/bash

echo "Final FlowNet-LSTM-Transformer Training"
echo "Optimized for successful distributed training"
echo ""

# Kill any existing training
pkill -f train_flownet_lstm_transformer

# Run with reduced settings for stability
torchrun --nproc_per_node=4 train_flownet_lstm_transformer_optimized.py \
    --distributed \
    --use-amp \
    --variable-length \
    --data-dir ../aria_processed \
    --batch-size 8 \
    --num-workers 2 \
    --learning-rate 1.2e-3 \
    --num-epochs 100 \
    --min-seq-len 8 \
    --max-seq-len 40 \
    --curriculum-step 10 \
    --curriculum-increment 4 \
    --stride 5 \
    --use-curriculum \
    --checkpoint-dir checkpoints/exp_final_4gpu \
    --experiment-name flownet_lstm_final \
    --validate-every 5