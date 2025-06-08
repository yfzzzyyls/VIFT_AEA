#!/bin/bash

echo "Optimal Parallel Processing for Full Aria Dataset (143 sequences)"
echo "==============================================================="
echo "Using 10 CPU workers for consistent performance"
echo ""

# Configuration
ARIA_PATH="/mnt/ssd_ext/incSeg-data/aria_everyday"
OUTPUT_BASE="data/aria_full"
PROCESSED_DIR="$OUTPUT_BASE/processed"
MAX_FRAMES=10000
NUM_WORKERS=10

# Create directories
mkdir -p $PROCESSED_DIR
mkdir -p $OUTPUT_BASE/logs

# Activate environment
source ~/venv/py39/bin/activate

# Total sequences: 143
# Divide among 10 workers: ~14-15 sequences each
SEQUENCES_PER_WORKER=14

echo "Launching $NUM_WORKERS parallel CPU workers..."
echo "Each worker will process ~$SEQUENCES_PER_WORKER sequences"
echo "All workers using CPU for consistency"
echo ""

# Launch workers
PIDS=()
for i in $(seq 0 $((NUM_WORKERS-1))); do
    START_IDX=$((i * SEQUENCES_PER_WORKER))
    
    # Calculate sequences for this worker
    if [ $i -eq $((NUM_WORKERS-1)) ]; then
        # Last worker gets remaining sequences
        MAX_SEQ=$((143 - START_IDX))
    else
        MAX_SEQ=$SEQUENCES_PER_WORKER
    fi
    
    echo "Worker $((i+1)): sequences $START_IDX-$((START_IDX + MAX_SEQ - 1)) on CPU"
    
    python scripts/process_aria_to_vift_quaternion.py \
        --input-dir $ARIA_PATH \
        --output-dir $PROCESSED_DIR \
        --max-frames $MAX_FRAMES \
        --start-index $START_IDX \
        --max-sequences $MAX_SEQ \
        --folder-offset $START_IDX \
        --device cpu \
        > $OUTPUT_BASE/logs/worker_$((i+1)).log 2>&1 &
    
    PIDS+=($!)
    
    # Small delay to prevent simultaneous file access issues
    sleep 2
done

echo ""
echo "All workers launched!"
echo "PIDs: ${PIDS[@]}"
echo ""
echo "Monitor overall progress:"
echo "  watch -n 5 'ls -1 $PROCESSED_DIR | wc -l'"
echo ""
echo "Monitor individual workers:"
echo "  tail -f $OUTPUT_BASE/logs/worker_*.log"
echo ""
echo "Check CPU usage:"
echo "  htop"
echo ""
echo "Worker distribution:"
echo "  Workers 1-9: 14 sequences each (126 total)"
echo "  Worker 10: 17 sequences (remaining)"
echo ""

# Function to check if all workers are done
check_workers() {
    for pid in ${PIDS[@]}; do
        if kill -0 $pid 2>/dev/null; then
            return 1  # Still running
        fi
    done
    return 0  # All done
}

# Wait with progress updates
echo "Processing... (press Ctrl+C to stop monitoring, workers will continue)"
while ! check_workers; do
    COMPLETED=$(ls -1 $PROCESSED_DIR 2>/dev/null | wc -l)
    echo -ne "\rProgress: $COMPLETED/143 sequences completed..."
    sleep 10
done

echo -e "\n\n✅ All sequences processed!"
echo ""
echo "Next step: Generate latent features"
echo "python generate_all_pretrained_latents_fixed.py \\"
echo "    --processed-dir $PROCESSED_DIR \\"
echo "    --output-dir $OUTPUT_BASE/latent \\"
echo "    --stride 10"