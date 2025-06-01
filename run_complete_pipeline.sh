#!/bin/bash

# Complete VIFT-AEA Pipeline Script
# This script runs the entire pipeline from feature generation to evaluation

# Activate virtual environment
echo "Activating Python 3.9 virtual environment..."
source ~/venv/py39/bin/activate

# Step 1: Generate features and prepare data
echo "Step 1: Generating features and preparing data..."
python generate_all_pretrained_latents_fixed.py

# Step 2: Train the model
echo "Step 2: Training the model..."
python train_pretrained_relative.py

# Step 3: Evaluate the model
echo "Step 3: Evaluating the model..."
python evaluate_with_metrics.py

echo "Pipeline complete!"