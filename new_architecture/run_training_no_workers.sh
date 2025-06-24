#!/bin/bash

# Training with no data loading workers to avoid memory issues
torchrun --nproc_per_node=4 train_flownet_lstm_transformer.py \
    --distributed \
    --use-amp \
    --variable-length \
    --data-dir ../aria_processed \
    --batch-size 16 \
    --num-workers 0 \
    --learning-rate 8e-4 \
    --num-epochs 100 \
    --min-seq-len 10 \
    --max-seq-len 50 \
    --curriculum-step 10 \
    --curriculum-increment 5 \
    --stride 5 \
    --use-curriculum \
    --checkpoint-dir checkpoints/exp_704_4gpu_bs16_no_workers \
    --experiment-name flownet_lstm_704_bs16_safe