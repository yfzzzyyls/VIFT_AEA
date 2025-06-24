#!/bin/bash

echo "Training with hybrid dataset (pre-computed windows like working script)"
echo "This should provide fast training with DDP support"
echo ""

# Create hybrid version of training script
cp train_flownet_lstm_transformer.py train_flownet_lstm_transformer_hybrid.py
sed -i 's/from data.aria_variable_imu_dataset import/from data.aria_variable_imu_dataset_hybrid import/' train_flownet_lstm_transformer_hybrid.py

# Kill any existing training processes
pkill -f train_flownet_lstm_transformer

# Run with proper settings
torchrun --nproc_per_node=4 train_flownet_lstm_transformer_hybrid.py \
    --distributed \
    --use-amp \
    --variable-length \
    --data-dir ../aria_processed \
    --batch-size 16 \
    --num-workers 4 \
    --learning-rate 8e-4 \
    --num-epochs 100 \
    --min-seq-len 10 \
    --max-seq-len 50 \
    --curriculum-step 10 \
    --curriculum-increment 5 \
    --stride 5 \
    --use-curriculum \
    --checkpoint-dir checkpoints/exp_704_4gpu_hybrid \
    --experiment-name flownet_lstm_704_hybrid \
    --image-height 704 \
    --image-width 704