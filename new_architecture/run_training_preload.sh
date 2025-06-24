#!/bin/bash

# First, update the training script to use preloaded dataset
sed -i 's/from data.aria_variable_imu_dataset import/from data.aria_variable_imu_dataset_preload import/' train_flownet_lstm_transformer.py

# Run training with preloaded dataset and workers
torchrun --nproc_per_node=4 train_flownet_lstm_transformer.py \
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
    --checkpoint-dir checkpoints/exp_704_4gpu_bs16_preload \
    --experiment-name flownet_lstm_704_bs16_preload

# Restore original import
sed -i 's/from data.aria_variable_imu_dataset_preload import/from data.aria_variable_imu_dataset import/' train_flownet_lstm_transformer.py