#!/bin/bash
# Script to extract TUM VI corridor datasets

TUMVI_DIR="/mnt/ssd_ext/incSeg-data/tumvi"

echo "Extracting TUM VI corridor datasets..."
echo "This will require approximately 15.5 GB of additional disk space"
echo

# Check if directory exists
if [ ! -d "$TUMVI_DIR" ]; then
    echo "Error: TUM VI directory not found at $TUMVI_DIR"
    exit 1
fi

cd "$TUMVI_DIR"

# List of corridor tar files
CORRIDOR_TARS=(
    "dataset-corridor1_512_16.tar"
    "dataset-corridor2_512_16.tar"
    "dataset-corridor3_512_16.tar"
    "dataset-corridor4_512_16.tar"
    "dataset-corridor5_512_16.tar"
)

# Extract each tar file
for tar_file in "${CORRIDOR_TARS[@]}"; do
    if [ -f "$tar_file" ]; then
        echo "Extracting $tar_file..."
        tar -xf "$tar_file"
        if [ $? -eq 0 ]; then
            echo "✓ Successfully extracted $tar_file"
        else
            echo "✗ Failed to extract $tar_file"
        fi
    else
        echo "Warning: $tar_file not found"
    fi
    echo
done

echo "Extraction complete!"
echo
echo "Verifying extracted directories:"
for i in {1..5}; do
    dir="dataset-corridor${i}_512_16"
    if [ -d "$dir/mav0" ]; then
        echo "✓ $dir is ready"
    else
        echo "✗ $dir/mav0 not found"
    fi
done