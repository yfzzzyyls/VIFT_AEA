#!/usr/bin/env python3
"""
Convert Aria IMU data from PyTorch format to MATLAB format compatible with KITTI.
"""

import torch
import scipy.io as sio
import numpy as np
import os
from pathlib import Path

def convert_aria_imu_to_matlab(aria_processed_dir, output_dir):
    """
    Convert Aria IMU data to KITTI-compatible MATLAB format.
    
    Args:
        aria_processed_dir: Directory containing processed Aria sequences
        output_dir: Output directory for MATLAB files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sequences = ['016', '017', '018', '019']
    
    for seq in sequences:
        seq_path = Path(aria_processed_dir) / seq / 'imu_data.pt'
        
        if not seq_path.exists():
            print(f"Warning: IMU data not found for sequence {seq}")
            continue
            
        # Load PyTorch IMU data
        imu_data = torch.load(seq_path)
        
        # Convert to numpy and reshape
        # Aria format: [n_frames, 10, 6]
        # KITTI format: [n_samples, 6] where n_samples = (n_frames-1)*10 + 1
        n_frames = imu_data.shape[0]
        
        # Flatten the IMU data to match KITTI format
        # For n frames, we have (n-1)*10 + 1 IMU samples
        imu_flat = []
        for i in range(n_frames - 1):
            imu_flat.extend(imu_data[i].numpy())
        # Add the first sample from the last frame
        imu_flat.append(imu_data[-1, 0].numpy())
        
        imu_array = np.array(imu_flat)
        
        print(f"Sequence {seq}:")
        print(f"  Original shape: {imu_data.shape}")
        print(f"  Converted shape: {imu_array.shape}")
        print(f"  IMU range - Angular vel: [{imu_array[:, :3].min():.3f}, {imu_array[:, :3].max():.3f}]")
        print(f"  IMU range - Linear acc: [{imu_array[:, 3:].min():.3f}, {imu_array[:, 3:].max():.3f}]")
        
        # Save as MATLAB file
        output_path = os.path.join(output_dir, f"{seq}.mat")
        sio.savemat(output_path, {'imu_data_interp': imu_array})
        print(f"  Saved to: {output_path}\n")
        
    print("Conversion complete!")
    print("\nNote: KITTI expects IMU in the format:")
    print("  - Angular velocity (rad/s): columns 0-2")
    print("  - Linear acceleration (m/s²): columns 3-5")
    
    # Also check if KITTI sequences need dummy files
    kitti_sequences = ['05', '07', '10']
    print("\nCreating links for KITTI sequence names...")
    for kitti_seq in kitti_sequences:
        # Map KITTI sequences to Aria sequences
        if kitti_seq == '05':
            aria_seq = '016'
        elif kitti_seq == '07':
            aria_seq = '017'
        elif kitti_seq == '10':
            aria_seq = '018'  # or '019', but we'll use 018 for now
            
        src_path = os.path.join(output_dir, f"{aria_seq}.mat")
        dst_path = os.path.join(output_dir, f"{kitti_seq}.mat")
        
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            os.symlink(src_path, dst_path)
            print(f"  Created symlink: {kitti_seq}.mat -> {aria_seq}.mat")


if __name__ == "__main__":
    aria_processed_dir = "/home/external/VIFT_AEA/aria_processed"
    output_dir = "/home/external/VIFT_AEA/data/kitti_data/imus"
    
    convert_aria_imu_to_matlab(aria_processed_dir, output_dir)