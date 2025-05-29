#!/usr/bin/env python3
"""
Very simple evaluation by loading just numpy arrays and calculating basic metrics
"""
import numpy as np
import os
from pathlib import Path

def simple_evaluation():
    """Simple evaluation that just loads the test data and reports basic statistics"""
    
    test_dir = Path("/home/external/VIFT_AEA/aria_latent_data/test")
    
    print("🔍 Simple Test Data Analysis")
    print("=" * 50)
    
    # Get all latent feature files (without _gt, _rot, _w suffix)
    feature_files = [f for f in os.listdir(test_dir) if f.endswith('.npy') and '_' not in f]
    feature_files = sorted(feature_files, key=lambda x: int(x.split('.')[0]))
    
    print(f"📊 Found {len(feature_files)} test samples")
    
    # Load a few samples to understand the data
    sample_features = []
    sample_targets = []
    
    for i, feature_file in enumerate(feature_files[:10]):  # Load first 10 samples
        try:
            # Load feature
            feature_path = test_dir / feature_file
            feature = np.load(feature_path)
            
            # Load corresponding ground truth
            sample_num = feature_file.split('.')[0]
            gt_path = test_dir / f"{sample_num}_gt.npy"
            
            if gt_path.exists():
                gt = np.load(gt_path)
                sample_features.append(feature)
                sample_targets.append(gt)
            
        except Exception as e:
            print(f"❌ Error loading sample {feature_file}: {e}")
            continue
    
    if sample_features:
        sample_features = np.array(sample_features)
        sample_targets = np.array(sample_targets)
        
        print(f"✅ Successfully loaded {len(sample_features)} samples")
        print(f"📐 Feature shape: {sample_features.shape}")
        print(f"📐 Target shape: {sample_targets.shape}")
        print(f"📈 Feature statistics:")
        print(f"   Mean: {np.mean(sample_features):.6f}")
        print(f"   Std:  {np.std(sample_features):.6f}")
        print(f"   Min:  {np.min(sample_features):.6f}")
        print(f"   Max:  {np.max(sample_features):.6f}")
        
        print(f"📈 Target statistics:")
        print(f"   Mean: {np.mean(sample_targets):.6f}")
        print(f"   Std:  {np.std(sample_targets):.6f}")
        print(f"   Min:  {np.min(sample_targets):.6f}")
        print(f"   Max:  {np.max(sample_targets):.6f}")
        
        # Simple baseline: predict mean target
        mean_target = np.mean(sample_targets, axis=0)
        print(f"📊 Mean target prediction: {mean_target}")
        
        # Calculate baseline MSE (predicting mean)
        baseline_mse = np.mean((sample_targets - mean_target) ** 2)
        print(f"📊 Baseline MSE (predicting mean): {baseline_mse:.6f}")
        
        print("=" * 50)
        print("✅ Simple analysis complete!")
        print(f"📝 Note: This is a basic data analysis.")
        print(f"    For full model evaluation, resolve the import issues with the Lightning framework.")
        
    else:
        print("❌ No samples could be loaded")

if __name__ == "__main__":
    simple_evaluation()
