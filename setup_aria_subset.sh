#!/bin/bash
# Download and setup 10-sequence AriaEveryday subset for VIFT training

set -e  # Exit on any error

echo "🎯 Setting up AriaEveryday 10-sequence subset for VIFT"
echo "=============================================="

# Set up directories
cd /vast/fy2243/VIFT_AEA

# Step 1: Download 10 sequences with essential files
echo "📥 Step 1: Downloading 10 sequences with essential files..."
python download_aria_with_json.py AriaEverydayActivities_download_urls.json \
  --num-sequences 10 \
  --file-types video_main_rgb mps_slam_trajectories mps_slam_summary mps_slam_calibration \
  --output-dir ./data/aria_everyday_10seq

echo "✅ Downloaded 10 sequences"

# Step 2: Process sequences to VIFT format
echo "🔄 Step 2: Processing sequences to VIFT format..."
python scripts/process_aria_to_vift.py \
  --aria-data-dir ./data/aria_everyday_10seq \
  --output-dir ./data/aria_processed \
  --start-index 0 \
  --max-sequences 10 \
  --max-frames 100

echo "✅ Processed 10 sequences"

# Step 3: Create train/test split (6 train, 4 test)
echo "📋 Step 3: Creating sequence lists (6 train, 4 test)..."
python scripts/create_sequence_lists.py \
  --processed-data-dir ./data/aria_processed \
  --train-ratio 0.6

echo "✅ Created sequence lists"

# Step 4: Test dataset loading
echo "🧪 Step 4: Testing dataset loading..."
python data/aria_dataset.py

echo "🎉 Setup complete!"
echo ""
echo "📊 Summary:"
echo "  - Downloaded: 10 AriaEveryday sequences"
echo "  - Training: 6 sequences"
echo "  - Testing: 4 sequences"
echo "  - Max frames per sequence: 100"
echo ""
echo "🚀 Next steps:"
echo "  1. Run latent caching: python data/latent_caching_aria.py"
echo "  2. Train VIFT: python src/train.py data=aria_vio"
echo "  3. Test inference: python src/test.py data=aria_vio"