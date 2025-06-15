#!/bin/bash

echo "Processing 20 Selected Sequences with Proper Between-Frames IMU"
echo "=============================================================="
echo "Using 4 CPU workers for parallel processing"
echo "IMU data will be extracted BETWEEN consecutive frames"
echo ""

# Configuration
ARIA_PATH="/mnt/ssd_ext/incSeg-data/aria_everyday"
OUTPUT_BASE="/home/external/VIFT_AEA"
PROCESSED_DIR="$OUTPUT_BASE/aria_processed_fixed"
MAX_FRAMES=1000  # Default 1000 frames evenly sampled
NUM_WORKERS=4

# Create directories
mkdir -p $PROCESSED_DIR
mkdir -p $OUTPUT_BASE/logs_fixed

# Activate environment
source ~/venv/py39/bin/activate

# Automatically pick 20 random sequences from ARIA_PATH
echo "Selecting 20 random sequences from $ARIA_PATH"
mapfile -t SEQ_DIRS < <(find "$ARIA_PATH" -mindepth 1 -maxdepth 1 -type d | shuf -n 20)
SEQUENCES=()
for dir in "${SEQ_DIRS[@]}"; do
    SEQUENCES+=("$(basename "$dir")")
done

# Create sequence mapping file
echo "Creating sequence mapping..."
cat > $PROCESSED_DIR/sequence_mapping.json << EOF
{
EOF

for i in "${!SEQUENCES[@]}"; do
    idx=$(printf "%03d" $i)
    if [ $i -eq $((${#SEQUENCES[@]}-1)) ]; then
        echo "  \"$idx\": \"${SEQUENCES[$i]}\"" >> $PROCESSED_DIR/sequence_mapping.json
    else
        echo "  \"$idx\": \"${SEQUENCES[$i]}\"," >> $PROCESSED_DIR/sequence_mapping.json
    fi
done

echo "}" >> $PROCESSED_DIR/sequence_mapping.json

echo "Processing ${#SEQUENCES[@]} sequences with $NUM_WORKERS workers"
echo "Each sequence will have $MAX_FRAMES frames extracted (evenly sampled)"
echo "IMU data format: Between-frames (proper VIO alignment)"
echo ""

# Function to process sequences for a worker
process_sequences() {
    local worker_id=$1
    local sequences_per_worker=$((${#SEQUENCES[@]} / NUM_WORKERS))
    local start_idx=$((worker_id * sequences_per_worker))
    local end_idx=$((start_idx + sequences_per_worker - 1))
    
    # Last worker gets remaining sequences
    if [ $worker_id -eq $((NUM_WORKERS-1)) ]; then
        end_idx=$((${#SEQUENCES[@]} - 1))
    fi
    
    echo "[Worker $((worker_id+1))] Processing sequences $start_idx-$end_idx" > $OUTPUT_BASE/logs_fixed/worker_$((worker_id+1)).log
    
    for idx in $(seq $start_idx $end_idx); do
        if [ $idx -lt ${#SEQUENCES[@]} ]; then
            seq_name=${SEQUENCES[$idx]}
            output_idx=$(printf "%03d" $idx)
            
            echo "[Worker $((worker_id+1))] Processing $seq_name -> $output_idx" >> $OUTPUT_BASE/logs_fixed/worker_$((worker_id+1)).log
            
            # Process single sequence using real IMU script
            python scripts/process_aria_raw_with_real_imu_fixed.py \
                --input-dir "$ARIA_PATH" \
                --output-dir "$PROCESSED_DIR" \
                --sequences "$seq_name" \
                --max-frames $MAX_FRAMES \
                >> $OUTPUT_BASE/logs_fixed/worker_$((worker_id+1)).log 2>&1
            
            # Rename output folder to match expected index
            if [ -d "$PROCESSED_DIR/$seq_name" ]; then
                mv "$PROCESSED_DIR/$seq_name" "$PROCESSED_DIR/$output_idx"
                echo "[Worker $((worker_id+1))] Renamed $seq_name to $output_idx" >> $OUTPUT_BASE/logs_fixed/worker_$((worker_id+1)).log
            fi
            
            # Check if successful
            if [ -f "$PROCESSED_DIR/$output_idx/poses_quaternion.json" ]; then
                frame_count=$(python -c "import json; print(len(json.load(open('$PROCESSED_DIR/$output_idx/poses_quaternion.json'))))" 2>/dev/null)
                echo "[Worker $((worker_id+1))] ✓ $seq_name: $frame_count frames" >> $OUTPUT_BASE/logs_fixed/worker_$((worker_id+1)).log
            else
                echo "[Worker $((worker_id+1))] ✗ $seq_name: Failed" >> $OUTPUT_BASE/logs_fixed/worker_$((worker_id+1)).log
            fi
        fi
    done
    
    echo "[Worker $((worker_id+1))] Completed" >> $OUTPUT_BASE/logs_fixed/worker_$((worker_id+1)).log
}

# Launch workers
echo "Launching $NUM_WORKERS parallel workers..."
PIDS=()

for i in $(seq 0 $((NUM_WORKERS-1))); do
    process_sequences $i &
    PIDS+=($!)
    echo "Worker $((i+1)) launched with PID ${PIDS[$i]}"
    sleep 1
done

echo ""
echo "All workers launched!"
echo ""
echo "Monitor progress:"
echo "  watch -n 2 'ls -1 $PROCESSED_DIR | grep -E \"^[0-9]{3}$\" | wc -l'"
echo ""
echo "View logs:"
echo "  tail -f $OUTPUT_BASE/logs_fixed/worker_*.log"
echo ""

# Wait for all workers
echo "Waiting for workers to complete..."
for pid in ${PIDS[@]}; do
    wait $pid
done

echo ""
echo "✅ All workers completed!"
echo ""

# Summary
COMPLETED=$(ls -1 $PROCESSED_DIR | grep -E '^[0-9]{3}$' | wc -l)
echo "Processed $COMPLETED out of ${#SEQUENCES[@]} sequences"
echo ""

# Check for failures
echo "Checking for failures..."
grep -h "Failed\|✗" $OUTPUT_BASE/logs_fixed/worker_*.log | sort -u

echo ""
echo "IMU Data Format:"
echo "  - Between-frames extraction (proper VIO alignment)"
echo "  - N-1 intervals of IMU data for N frames"
echo "  - Each interval contains 50 IMU samples at 1000Hz"
echo ""
echo "Next steps:"
echo "1. Generate latent features with the fixed data:"
echo "   python generate_all_pretrained_latents_fixed.py \\"
echo "       --processed-dir $PROCESSED_DIR \\"
echo "       --output-dir $OUTPUT_BASE/aria_latent_fixed \\"
echo "       --stride 10 \\"
echo "       --skip-test"
echo ""
echo "2. Train model with properly aligned IMU data"
echo "3. Expect improved VIO performance!"