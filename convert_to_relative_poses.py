#!/usr/bin/env python3
"""
Convert absolute poses to frame-to-frame relative poses
"""

import numpy as np
from pathlib import Path
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import shutil


def quaternion_multiply(q1, q2):
    """Multiply two quaternions (qw, qx, qy, qz format)"""
    # Convert from xyzw to wxyz for computation
    q1_wxyz = np.array([q1[3], q1[0], q1[1], q1[2]])
    q2_wxyz = np.array([q2[3], q2[0], q2[1], q2[2]])
    
    w1, x1, y1, z1 = q1_wxyz
    w2, x2, y2, z2 = q2_wxyz
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    # Convert back to xyzw
    return np.array([x, y, z, w])


def quaternion_conjugate(q):
    """Compute quaternion conjugate (qx, qy, qz, qw format)"""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def rotate_vector(v, q):
    """Rotate vector v by quaternion q"""
    # Convert to Rotation object for easier manipulation
    r = Rotation.from_quat(q)  # expects xyzw
    return r.apply(v)


def compute_relative_pose(pose1, pose2):
    """
    Compute relative pose from pose1 to pose2
    Both poses are [tx, ty, tz, qx, qy, qz, qw]
    Returns relative pose in same format
    """
    # Extract components
    t1, q1 = pose1[:3], pose1[3:]
    t2, q2 = pose2[:3], pose2[3:]
    
    # Compute relative rotation: q_rel = q1^{-1} * q2
    q1_inv = quaternion_conjugate(q1)
    q_rel = quaternion_multiply(q1_inv, q2)
    
    # Compute relative translation: t_rel = R1^{-1} * (t2 - t1)
    t_diff = t2 - t1
    t_rel = rotate_vector(t_diff, q1_inv)
    
    # Combine
    relative_pose = np.concatenate([t_rel, q_rel])
    return relative_pose


def convert_dataset_to_relative(data_dir, output_dir, split='train'):
    """Convert a dataset split to relative poses"""
    
    data_path = Path(data_dir) / split
    output_path = Path(output_dir) / split
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all samples
    gt_files = sorted(data_path.glob("*_gt.npy"))
    
    print(f"\nProcessing {split} split: {len(gt_files)} samples")
    
    for gt_file in tqdm(gt_files, desc=f"Converting {split}"):
        sample_id = gt_file.stem.split('_')[0]
        
        # Load data
        absolute_poses = np.load(gt_file)
        visual_features = np.load(data_path / f"{sample_id}_visual.npy")
        imu_features = np.load(data_path / f"{sample_id}_imu.npy")
        
        # Convert to relative poses
        num_frames = len(absolute_poses)
        relative_poses = np.zeros_like(absolute_poses)
        
        # First frame is relative to itself (zero translation, identity rotation)
        relative_poses[0] = np.array([0, 0, 0, 0, 0, 0, 1])
        
        # Compute frame-to-frame relative poses
        for i in range(1, num_frames):
            relative_poses[i] = compute_relative_pose(
                absolute_poses[i-1], 
                absolute_poses[i]
            )
        
        # Verify the relative poses are reasonable
        trans_mags = np.linalg.norm(relative_poses[:, :3], axis=1)
        print(f"\n  Sample {sample_id}:")
        print(f"    Translation magnitudes (m): mean={trans_mags.mean():.4f}, max={trans_mags.max():.4f}")
        
        # Save relative poses (keeping same format)
        np.save(output_path / f"{sample_id}_gt.npy", relative_poses)
        
        # Copy visual and IMU features
        np.save(output_path / f"{sample_id}_visual.npy", visual_features)
        np.save(output_path / f"{sample_id}_imu.npy", imu_features)
    
    print(f"✅ Converted {len(gt_files)} samples in {split} split")


def main():
    # Input and output directories
    input_dir = "data/aria_latent_data_pretrained"
    output_dir = "data/aria_latent_data_relative"
    
    print("Converting absolute poses to frame-to-frame relative poses")
    print("=" * 60)
    
    # Convert all splits
    for split in ['train', 'val', 'test']:
        if (Path(input_dir) / split).exists():
            convert_dataset_to_relative(input_dir, output_dir, split)
    
    print("\n✅ Conversion complete!")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()