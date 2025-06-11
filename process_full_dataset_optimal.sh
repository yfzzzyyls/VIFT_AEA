#!/bin/bash

echo "Processing 20 Selected Sequences with ALL Frames"
echo "==============================================="
echo "Using 4 CPU workers for parallel processing"
echo ""

# Configuration
ARIA_PATH="/mnt/ssd_ext/incSeg-data/aria_everyday"
OUTPUT_BASE="/home/external/VIFT_AEA"
PROCESSED_DIR="$OUTPUT_BASE/aria_processed_full_frames"
MAX_FRAMES=-1  # Process ALL frames
NUM_WORKERS=4

# Create directories
mkdir -p $PROCESSED_DIR
mkdir -p $OUTPUT_BASE/logs_full_frames

# Activate environment
source ~/venv/py39/bin/activate

# Selected 20 sequences from 4 locations
SEQUENCES=(
    "loc1_script1_seq1_rec1"
    "loc1_script1_seq5_rec1"
    "loc1_script2_seq3_rec1"
    "loc1_script2_seq6_rec2"
    "loc1_script3_seq2_rec1"
    "loc2_script1_seq2_rec1"
    "loc2_script1_seq4_rec1"
    "loc2_script2_seq1_rec1"
    "loc2_script2_seq3_rec1"
    "loc2_script3_seq5_rec1"
    "loc3_script1_seq1_rec1"
    "loc3_script1_seq3_rec1"
    "loc3_script2_seq2_rec1"
    "loc3_script2_seq4_rec1"
    "loc3_script3_seq1_rec1"
    "loc4_script1_seq1_rec1"
    "loc4_script1_seq3_rec1"
    "loc4_script2_seq2_rec1"
    "loc4_script2_seq4_rec1"
    "loc4_script3_seq3_rec1"
)

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
echo "Each sequence will have ALL frames extracted"
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
    
    echo "[Worker $((worker_id+1))] Processing sequences $start_idx-$end_idx" > $OUTPUT_BASE/logs_full_frames/worker_$((worker_id+1)).log
    
    for idx in $(seq $start_idx $end_idx); do
        if [ $idx -lt ${#SEQUENCES[@]} ]; then
            seq_name=${SEQUENCES[$idx]}
            output_idx=$(printf "%03d" $idx)
            
            echo "[Worker $((worker_id+1))] Processing $seq_name -> $output_idx" >> $OUTPUT_BASE/logs_full_frames/worker_$((worker_id+1)).log
            
            # Process single sequence using wrapper
            python process_single_aria_sequence.py \
                --sequence-name "$seq_name" \
                --aria-path "$ARIA_PATH" \
                --output-dir "$PROCESSED_DIR" \
                --output-idx "$output_idx" \
                --max-frames $MAX_FRAMES \
                >> $OUTPUT_BASE/logs_full_frames/worker_$((worker_id+1)).log 2>&1
            
            # Check if successful
            if [ -f "$PROCESSED_DIR/$output_idx/poses_quaternion.json" ]; then
                frame_count=$(python -c "import json; print(len(json.load(open('$PROCESSED_DIR/$output_idx/poses_quaternion.json'))))" 2>/dev/null)
                echo "[Worker $((worker_id+1))] ✓ $seq_name: $frame_count frames" >> $OUTPUT_BASE/logs_full_frames/worker_$((worker_id+1)).log
            else
                echo "[Worker $((worker_id+1))] ✗ $seq_name: Failed" >> $OUTPUT_BASE/logs_full_frames/worker_$((worker_id+1)).log
            fi
        fi
    done
    
    echo "[Worker $((worker_id+1))] Completed" >> $OUTPUT_BASE/logs_full_frames/worker_$((worker_id+1)).log
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
echo "  tail -f $OUTPUT_BASE/logs_full_frames/worker_*.log"
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
grep -h "Failed\|✗" $OUTPUT_BASE/logs_full_frames/worker_*.log | sort -u

echo ""
echo "Next steps:"
echo "1. Generate latent features with stride 10:"
echo "   python generate_all_pretrained_latents_fixed.py \\"
echo "       --processed-dir $PROCESSED_DIR \\"
echo "       --output-dir $OUTPUT_BASE/aria_latent_full_frames \\"
echo "       --stride 10 \\"
echo "       --pose-scale 100.0 \\"
echo "       --skip-test"
echo ""
echo "2. Train model with full frame data"
echo "3. Run inference and generate plots"