#!/usr/bin/env python3
"""
Generate properly fixed rotation data for all splits
"""

import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from tqdm import tqdm
import shutil

def generate_proper_quaternions_for_split(input_dir, output_dir, max_samples=None):
    """Generate proper quaternions from angular velocity data for a split"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of samples
    samples = []
    i = 0
    while os.path.exists(os.path.join(input_dir, f"{i}_gt.npy")):
        samples.append(i)
        i += 1
        if max_samples and i >= max_samples:
            break
    
    print(f"Found {len(samples)} samples in {input_dir}")
    
    rotation_stats = []
    translation_stats = []
    
    for idx in tqdm(samples, desc=f"Processing {os.path.basename(input_dir)}"):
        gt_file = os.path.join(input_dir, f'{idx}_gt.npy')
        w_file = os.path.join(input_dir, f'{idx}_w.npy')
        
        gt_poses = np.load(gt_file)
        new_poses = gt_poses.copy()
        
        # Keep translations as they are (they seem fine)
        # Fix rotations using angular velocity
        if os.path.exists(w_file):
            w_data = np.load(w_file)
            
            # Initialize with identity quaternion (XYZW format)
            new_poses[0, 3:] = [0, 0, 0, 1]
            
            dt = 1/30.0  # 30Hz sampling rate
            
            # For relative poses, each quaternion represents the rotation from previous frame
            for i in range(1, min(len(gt_poses), len(w_data))):
                # Get angular velocity (first 3 components)
                omega = w_data[i, :3]  # rad/s
                
                # Compute rotation angle and axis
                omega_norm = np.linalg.norm(omega)
                angle = omega_norm * dt
                
                if angle > 1e-8:
                    axis = omega / omega_norm
                    # Create rotation for this time step
                    delta_rotation = R.from_rotvec(axis * angle)
                    # Convert to quaternion (XYZW format)
                    q_xyzw = delta_rotation.as_quat()
                else:
                    # No rotation
                    q_xyzw = [0, 0, 0, 1]
                
                new_poses[i, 3:] = q_xyzw
                rotation_stats.append(np.degrees(angle))
                
                # Track translation stats
                trans = new_poses[i, :3]
                trans_norm = np.linalg.norm(trans)
                translation_stats.append(trans_norm)
        else:
            print(f"Warning: No angular velocity data for sample {idx}")
            # Keep original near-zero rotations
            for i in range(1, len(gt_poses)):
                rotation_stats.append(0.0)
                trans_norm = np.linalg.norm(new_poses[i, :3])
                translation_stats.append(trans_norm)
        
        # Save fixed poses
        np.save(os.path.join(output_dir, f"{idx}_gt.npy"), new_poses)
        
        # Copy other files
        for suffix in ['.npy', '_rot.npy', '_w.npy']:
            src = os.path.join(input_dir, f"{idx}{suffix}")
            if os.path.exists(src) and suffix != '_gt.npy':
                dst = os.path.join(output_dir, f"{idx}{suffix}")
                shutil.copy2(src, dst)
    
    # Print statistics
    if rotation_stats:
        rotation_stats = np.array(rotation_stats)
        translation_stats = np.array(translation_stats)
        
        print(f"\nStatistics for {os.path.basename(input_dir)}:")
        print(f"Rotation angles (degrees per frame):")
        print(f"  Mean: {np.mean(rotation_stats):.4f}")
        print(f"  Std:  {np.std(rotation_stats):.4f}")
        print(f"  Max:  {np.max(rotation_stats):.4f}")
        print(f"  95%:  {np.percentile(rotation_stats, 95):.4f}")
        
        print(f"\nTranslation norms (cm):")
        print(f"  Mean: {np.mean(translation_stats):.4f}")
        print(f"  Std:  {np.std(translation_stats):.4f}")
        print(f"  Max:  {np.max(translation_stats):.4f}")
        print(f"  95%:  {np.percentile(translation_stats, 95):.4f}")
    
    return len(samples)


def main():
    """Generate fixed data for all splits"""
    
    input_base = "aria_latent_data_pretrained"
    output_base = "aria_latent_data_properly_fixed"
    
    splits = ['train', 'val', 'test']
    
    print("=== GENERATING PROPERLY FIXED ROTATION DATA ===\n")
    
    total_samples = 0
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        input_dir = os.path.join(input_base, split)
        output_dir = os.path.join(output_base, split)
        
        if os.path.exists(input_dir):
            num_samples = generate_proper_quaternions_for_split(input_dir, output_dir)
            total_samples += num_samples
        else:
            print(f"Warning: {input_dir} not found, skipping...")
    
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Total samples processed: {total_samples}")
    print(f"\nFixed data saved to: {output_base}/")
    print("\nNext steps:")
    print("1. Train with fixed data:")
    print("   python train_pretrained_relative.py --data_dir aria_latent_data_properly_fixed")
    print("\n2. Evaluate the model:")
    print("   python evaluate_with_metrics.py --checkpoint <checkpoint_path> --data_dir aria_latent_data_properly_fixed")


if __name__ == "__main__":
    main()