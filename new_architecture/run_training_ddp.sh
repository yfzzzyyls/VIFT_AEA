#!/bin/bash

echo "Training with DDP-compatible dataset (like working script)"
echo "Using smaller images (512x512) and proper data loading"
echo ""

# First update the import to use DDP-compatible dataset
cp train_flownet_lstm_transformer.py train_flownet_lstm_transformer_ddp.py
sed -i 's/from data.aria_variable_imu_dataset import/from data.aria_variable_imu_dataset_ddp import/' train_flownet_lstm_transformer_ddp.py

# Run with full resolution and proper settings
torchrun --nproc_per_node=4 train_flownet_lstm_transformer_ddp.py \
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
    --checkpoint-dir checkpoints/exp_704_4gpu_ddp \
    --experiment-name flownet_lstm_704_ddp \
    --image-height 704 \
    --image-width 704