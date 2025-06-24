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
    --data-dir ../aria_processed \
    --encoder-type resnet18 \
    --pretrained \
    --batch-size 16 \
    --num-workers 2 \
    --learning-rate 1.2e-3 \
    --num-epochs 100 \
    --sequence-length 41 \
    --stride 5 \
    --translation-weight 1.0 \
    --rotation-weight 100.0 \
    --scale-weight 10.0 \
    --checkpoint-dir checkpoints/final \
    --experiment-name resnet_lstm_41frames \
    --validate-every 1