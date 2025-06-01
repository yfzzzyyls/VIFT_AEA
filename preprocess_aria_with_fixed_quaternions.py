#!/usr/bin/env python3
"""
Preprocess Aria data with corrected quaternion handling for relative poses.
"""
import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import json

# Add current directory to path to import the fixed quaternion functions
sys.path.append('.')
from train_pretrained_relative import quaternion_multiply, quaternion_inverse, convert_absolute_to_relative


def load_pretrained_features(data_dir: str, split: str):
    """Load pre-extracted features from the pretrained model."""
    feature_dir = Path(f"aria_latent_data_pretrained/{split}")
    
    if not feature_dir.exists():
        raise ValueError(f"Feature directory not found: {feature_dir}")
    
    # Find all samples
    samples = []
    i = 0
    consecutive_misses = 0
    
    while consecutive_misses < 100:
        feature_path = feature_dir / f"{i}.npy"
        gt_path = feature_dir / f"{i}_gt.npy"
        
        if feature_path.exists() and gt_path.exists():
            samples.append(i)
            consecutive_misses = 0
        else:
            consecutive_misses += 1
        
        i += 1
    
    print(f"Found {len(samples)} samples in {feature_dir}")
    return feature_dir, samples


def process_and_save_relative_poses(feature_dir: Path, samples: list, output_dir: Path, pose_scale: float = 100.0):
    """Process samples and save with relative poses."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(samples)} samples...")
    
    # Track statistics
    rotation_angles = []
    translation_norms = []
    
    for idx, sample_id in enumerate(tqdm(samples)):
        try:
            # Load features
            features = np.load(feature_dir / f"{sample_id}.npy")
            
            # Load ground truth poses (absolute)
            absolute_poses = np.load(feature_dir / f"{sample_id}_gt.npy")
            
            # Convert to relative poses using the fixed functions
            relative_poses = convert_absolute_to_relative(absolute_poses)
            
            # Scale translation from meters to centimeters
            relative_poses[:, :3] *= pose_scale
            
            # Save processed data
            np.save(output_dir / f"{idx}.npy", features)
            np.save(output_dir / f"{idx}_gt.npy", relative_poses)
            
            # Copy other files if they exist
            for suffix in ['_rot.npy', '_w.npy']:
                src_path = feature_dir / f"{sample_id}{suffix}"
                if src_path.exists():
                    data = np.load(src_path)
                    np.save(output_dir / f"{idx}{suffix}", data)
            
            # Collect statistics (skip first frame which is always identity)
            for i in range(1, len(relative_poses)):
                # Translation norm
                trans_norm = np.linalg.norm(relative_poses[i, :3])
                translation_norms.append(trans_norm)
                
                # Rotation angle from quaternion
                quat = relative_poses[i, 3:]
                angle = 2 * np.arccos(np.clip(quat[3], -1, 1)) * 180 / np.pi
                rotation_angles.append(angle)
                
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            continue
    
    # Print statistics
    if rotation_angles:
        rotation_angles = np.array(rotation_angles)
        translation_norms = np.array(translation_norms)
        
        print(f"\nDataset Statistics:")
        print(f"Rotation angles (degrees):")
        print(f"  Mean: {np.mean(rotation_angles):.4f}")
        print(f"  Std:  {np.std(rotation_angles):.4f}")
        print(f"  Max:  {np.max(rotation_angles):.4f}")
        print(f"  95%:  {np.percentile(rotation_angles, 95):.4f}")
        
        print(f"\nTranslation norms (cm):")
        print(f"  Mean: {np.mean(translation_norms):.4f}")
        print(f"  Std:  {np.std(translation_norms):.4f}")
        print(f"  Max:  {np.max(translation_norms):.4f}")
        print(f"  95%:  {np.percentile(translation_norms, 95):.4f}")


def main():
    # Process all splits
    splits = ['train', 'val', 'test']
    pose_scale = 100.0  # Convert meters to centimeters
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")
        
        try:
            # Load existing features
            feature_dir, samples = load_pretrained_features("aria_latent_data_pretrained", split)
            
            # Process and save with corrected relative poses
            output_dir = Path(f"aria_latent_data_fixed/{split}")
            process_and_save_relative_poses(feature_dir, samples, output_dir, pose_scale)
            
            print(f"✓ Completed {split} split: {len(samples)} samples saved to {output_dir}")
            
        except Exception as e:
            print(f"✗ Error processing {split} split: {e}")
    
    print(f"\n{'='*60}")
    print("Data preprocessing complete!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Train the model with fixed data:")
    print("   python train_pretrained_relative.py --data_dir aria_latent_data_fixed")
    print("\n2. Evaluate the trained model:")
    print("   python evaluate_with_metrics.py --checkpoint logs/checkpoints_lite_scale_100.0/best_model.ckpt")


if __name__ == "__main__":
    main()