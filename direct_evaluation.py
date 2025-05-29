#!/usr/bin/env python3
"""
Direct evaluation script that bypasses complex config system
"""
import sys
import os
sys.path.append('/home/external/VIFT_AEA')

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, '/home/external/VIFT_AEA/src')

print("🚀 Starting direct evaluation")

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

# Import required modules
try:
    from models.vio_module import VIOLitModule
    from data.components.latent_kitti_dataset import LatentVectorDataset
    print("✅ Successfully imported modules")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Paths
checkpoint_path = "/home/external/VIFT_AEA/logs/aria_vio/runs/2025-05-29_13-44-07/checkpoints/epoch_000.ckpt"
test_data_path = "/home/external/VIFT_AEA/aria_latent_data/test"

print(f"📂 Loading model from: {checkpoint_path}")
print(f"📁 Loading test data from: {test_data_path}")

# Load model
try:
    # Try to load with map_location to handle any device issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model instance and load state dict manually if needed
    model = VIOLitModule.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully")
    print(f"📊 Model parameter count: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Load test dataset
try:
    test_dataset = LatentVectorDataset(test_data_path)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=0,  # No multiprocessing to avoid issues
        pin_memory=False
    )
    print(f"✅ Test dataset loaded: {len(test_dataset)} samples, {len(test_loader)} batches")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    sys.exit(1)

# Run evaluation
print("🚀 Starting evaluation...")
print("=" * 60)

total_loss = 0.0
total_samples = 0
batch_losses = []
successful_batches = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        try:
            # Get batch data
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            predictions = model(x)
            
            # Calculate loss using model's criterion
            loss = model.criterion(predictions, y)
            
            # Accumulate statistics
            batch_loss = loss.item()
            batch_size_actual = x.size(0)
            
            total_loss += batch_loss * batch_size_actual
            total_samples += batch_size_actual
            batch_losses.append(batch_loss)
            successful_batches += 1
            
            # Print progress every 5 batches
            if (batch_idx + 1) % 5 == 0:
                avg_loss_so_far = total_loss / total_samples
                print(f"  Batch {batch_idx + 1:3d}/{len(test_loader)}: Loss = {batch_loss:.6f}, Running Avg = {avg_loss_so_far:.6f}")
                
        except Exception as e:
            print(f"❌ Error in batch {batch_idx}: {e}")
            continue

# Calculate final statistics
if total_samples > 0:
    avg_loss = total_loss / total_samples
    min_loss = min(batch_losses) if batch_losses else 0
    max_loss = max(batch_losses) if batch_losses else 0
    std_loss = np.std(batch_losses) if batch_losses else 0
    
    print("\n" + "=" * 60)
    print("🎉 EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"📊 RESULTS:")
    print(f"  • Total samples processed: {total_samples:,}")
    print(f"  • Successful batches: {successful_batches}/{len(test_loader)}")
    print(f"  • Average test loss: {avg_loss:.6f}")
    print(f"  • Min batch loss: {min_loss:.6f}")
    print(f"  • Max batch loss: {max_loss:.6f}")
    print(f"  • Loss std deviation: {std_loss:.6f}")
    print("=" * 60)
    
    # Save results
    results = {
        'avg_loss': avg_loss,
        'min_loss': min_loss,
        'max_loss': max_loss,
        'std_loss': std_loss,
        'total_samples': total_samples,
        'successful_batches': successful_batches,
        'total_batches': len(test_loader)
    }
    
    results_path = "/home/external/VIFT_AEA/test_results.txt"
    with open(results_path, 'w') as f:
        f.write("VIFT-AEA Test Results\n")
        f.write("=" * 30 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"💾 Results saved to: {results_path}")
    print("✅ Evaluation completed successfully!")
    
else:
    print("\n❌ No samples were successfully processed!")
    sys.exit(1)
