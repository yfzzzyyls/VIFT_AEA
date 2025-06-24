#!/bin/bash

echo "Conservative training setup to avoid memory issues"
echo "Using: batch_size=8, num_workers=0 (no multiprocessing)"
echo ""

# Clear any cached memory first
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches 2>/dev/null || echo "Unable to clear cache (needs sudo)"

torchrun --nproc_per_node=4 train_flownet_lstm_transformer_preload.py \
    --distributed \
    --use-amp \
    --variable-length \
    --data-dir ../aria_processed \
    --batch-size 8 \
    --num-workers 0 \
    --learning-rate 4e-4 \
    --num-epochs 100 \
    --min-seq-len 10 \
    --max-seq-len 50 \
    --curriculum-step 10 \
    --curriculum-increment 5 \
    --stride 5 \
    --use-curriculum \
    --checkpoint-dir checkpoints/exp_704_4gpu_bs8_conservative \
    --experiment-name flownet_lstm_704_bs8_conservative