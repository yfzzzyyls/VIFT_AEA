#!/bin/bash

echo "Comprehensive FlowNet-LSTM-Transformer Evaluation"
echo "================================================"
echo ""

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --checkpoint PATH     Path to model checkpoint (required)"
    echo "  -s, --split SPLIT        Dataset split: val or test (default: test)"
    echo "  -b, --batch-size SIZE    Batch size for evaluation (default: 6)"
    echo "  -t, --stride STRIDE      Stride for sliding window (default: 5 to match training)"
    echo "  -o, --overlap            Use overlapping windows with stride=1 (last prediction only)"
    echo "  -g, --gpu GPU_ID         GPU device ID to use (default: 0)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Basic evaluation on test set"
    echo "  $0 -c ../checkpoints_from_scratch/resnet_lstm_41frames_mmap_shared/best_model.pt"
    echo ""
    echo "  # Evaluation with overlapping windows"
    echo "  $0 -c checkpoint.pt -o"
    echo ""
    echo "  # Evaluation on validation set with custom batch size"
    echo "  $0 -c checkpoint.pt -s val -b 8"
    exit 1
}

# Default values
CHECKPOINT_PATH=""
SPLIT="test"
BATCH_SIZE=6
STRIDE=5  # Match training stride by default
GPU_ID=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        -s|--split)
            SPLIT="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -t|--stride)
            STRIDE="$2"
            shift 2
            ;;
        -o|--overlap)
            STRIDE=1
            shift
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check required arguments
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint path is required"
    usage
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Extract experiment name from checkpoint path
EXPERIMENT_NAME=$(basename $(dirname "$CHECKPOINT_PATH"))

# Determine evaluation mode
if [ $STRIDE -eq 1 ]; then
    EVAL_MODE="last_prediction_only"
elif [ $STRIDE -eq 20 ]; then  # sequence_length - 1 for seq_len=21
    EVAL_MODE="non_overlapping"
else
    EVAL_MODE="overlapping_stride${STRIDE}"
fi

# Create output directory
OUTPUT_DIR="evaluation/${EXPERIMENT_NAME}_${SPLIT}_${EVAL_MODE}"
mkdir -p $OUTPUT_DIR

# Display configuration
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Split: $SPLIT"
echo "  Batch size: $BATCH_SIZE"
echo "  Stride: $STRIDE (${EVAL_MODE})"
echo "  GPU: $GPU_ID"
echo "  Output dir: $OUTPUT_DIR"
echo ""

# Set environment variables
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Run evaluation
echo "Starting evaluation..."
echo ""

python evaluate_flownet_lstm_transformer.py \
    --checkpoint $CHECKPOINT_PATH \
    --data-dir ../aria_processed \
    --output-dir $OUTPUT_DIR \
    --split $SPLIT \
    --batch-size $BATCH_SIZE \
    --sequence-length 21 \
    --stride $STRIDE \
    --num-workers 0 \
    --device cuda

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Evaluation completed successfully!"
    echo "  Results saved to: $OUTPUT_DIR"
    echo ""
    
    # Display metrics if available
    if [ -f "$OUTPUT_DIR/metrics.json" ]; then
        echo "=== Evaluation Metrics ==="
        python -c "
import json
with open('$OUTPUT_DIR/metrics.json', 'r') as f:
    metrics = json.load(f)
    print(f'  ATE Mean: {metrics.get(\"ate_mean\", 0):.4f} m')
    print(f'  ATE Std:  {metrics.get(\"ate_std\", 0):.4f} m')
    print(f'  RPE Trans: {metrics.get(\"rpe_trans_mean\", 0):.4f} m')
    print(f'  RPE Rot:   {metrics.get(\"rpe_rot_mean\", 0):.4f}°')
    print(f'  Scale Error: {metrics.get(\"scale_error_mean\", 0):.4f}')
    print(f'  Drift: {metrics.get(\"drift_mean\", 0):.2f}%')
"
    fi
    
    # List generated visualizations
    echo ""
    echo "=== Generated Visualizations ==="
    if [ -d "$OUTPUT_DIR/visualizations" ]; then
        ls -la $OUTPUT_DIR/visualizations/*.png 2>/dev/null | head -5
        TOTAL_VIS=$(ls $OUTPUT_DIR/visualizations/*.png 2>/dev/null | wc -l)
        echo "  ... and $TOTAL_VIS total visualization files"
    fi
else
    echo ""
    echo "✗ Evaluation failed! Check the error messages above."
    exit 1
fi