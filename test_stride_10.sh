#!/bin/bash
# Test with stride=10 for better temporal separation

source ~/venv/py39/bin/activate

echo "Testing with stride=10..."
echo "This will generate fewer but more temporally separated samples"

# Clean previous data
rm -rf aria_latent_data_pretrained

# Generate with stride=10
python generate_all_pretrained_latents.py --stride 10

# Show statistics
echo -e "\nGenerated data statistics:"
find aria_latent_data_pretrained -name "[0-9]*.npy" | wc -l