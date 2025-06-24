#!/bin/bash

echo "Final ResNet18-LSTM-Transformer Training"
echo "Using ResNet18 encoder for better speed and memory efficiency"
echo ""

# Kill any existing training
pkill -f train_flownet_lstm_transformer

# Run with reduced settings for stability
torchrun --nproc_per_node=4 train_flownet_lstm_transformer_optimized.py \
    --distributed \
    --use-amp \
    --variable-length \
    --data-dir ../aria_processed \
    --encoder-type resnet18 \
    --pretrained \
    --batch-size 16 \
    --num-workers 2 \
    --learning-rate 1.2e-3 \
    --num-epochs 100 \
    --min-seq-len 11 \
    --max-seq-len 41 \
    --curriculum-step 10 \
    --curriculum-increment 4 \
    --stride 5 \
    --use-curriculum \
    --translation-weight 1.0 \
    --rotation-weight 100.0 \
    --scale-weight 10.0 \
    --checkpoint-dir checkpoints/exp_final_4gpu \
    --experiment-name flownet_lstm_final \
    --validate-every 5