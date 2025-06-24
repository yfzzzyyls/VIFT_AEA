#!/bin/bash

echo "Running tested configuration with reduced memory usage"
echo ""

# Use the minimal dataset that loads on-demand
# Start with single GPU to verify it works
echo "Step 1: Testing with single GPU..."
python train_flownet_lstm_transformer_minimal.py \
    --use-amp \
    --variable-length \
    --data-dir ../aria_processed \
    --batch-size 2 \
    --num-workers 0 \
    --learning-rate 1e-4 \
    --num-epochs 1 \
    --min-seq-len 10 \
    --max-seq-len 15 \
    --stride 10 \
    --use-curriculum \
    --checkpoint-dir checkpoints/exp_test_single \
    --experiment-name test_single \
    --image-height 704 \
    --image-width 704 \
    --save-every 10 \
    --validate-every 10

if [ $? -eq 0 ]; then
    echo "Single GPU test passed!"
    echo ""
    echo "Step 2: Running distributed training..."
    
    # Now run distributed with conservative settings
    torchrun --nproc_per_node=4 train_flownet_lstm_transformer_minimal.py \
        --distributed \
        --use-amp \
        --variable-length \
        --data-dir ../aria_processed \
        --batch-size 2 \
        --num-workers 0 \
        --learning-rate 1e-4 \
        --num-epochs 100 \
        --min-seq-len 10 \
        --max-seq-len 20 \
        --stride 10 \
        --use-curriculum \
        --checkpoint-dir checkpoints/exp_704_4gpu_tested \
        --experiment-name flownet_lstm_704_tested \
        --image-height 704 \
        --image-width 704
else
    echo "Single GPU test failed! Not proceeding with distributed training."
fi