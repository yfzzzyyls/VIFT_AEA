#!/bin/bash

echo "Training with minimal dataset (no pre-loading)"
echo "Starting with conservative settings"
echo ""

# Create minimal version of training script
cp train_flownet_lstm_transformer.py train_flownet_lstm_transformer_minimal.py
sed -i 's/from data.aria_variable_imu_dataset import/from data.aria_variable_imu_dataset_minimal import/' train_flownet_lstm_transformer_minimal.py

# Run with very conservative settings first
torchrun --nproc_per_node=4 train_flownet_lstm_transformer_minimal.py \
    --distributed \
    --use-amp \
    --variable-length \
    --data-dir ../aria_processed \
    --batch-size 4 \
    --num-workers 0 \
    --learning-rate 1e-4 \
    --num-epochs 100 \
    --min-seq-len 10 \
    --max-seq-len 20 \
    --curriculum-step 10 \
    --curriculum-increment 5 \
    --stride 5 \
    --use-curriculum \
    --checkpoint-dir checkpoints/exp_704_4gpu_minimal \
    --experiment-name flownet_lstm_704_minimal \
    --image-height 704 \
    --image-width 704