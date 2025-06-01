import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import math

def quaternion_to_euler(q):
    """Convert quaternion to Euler angles in degrees"""
    r = R.from_quat(q)
    euler_rad = r.as_euler('xyz')
    return np.degrees(euler_rad)

def analyze_sample(idx, data_dir):
    """Analyze a single sample"""
    print(f"\n=== Analyzing Sample {idx} ===")
    
    # Load data
    gt_poses = np.load(os.path.join(data_dir, f"{idx}_gt.npy"))  # Shape: (seq_len, 7)
    
    print(f"Ground truth shape: {gt_poses.shape}")
    print(f"First 5 poses:")
    for i in range(min(5, len(gt_poses))):
        pos = gt_poses[i, :3]
        quat = gt_poses[i, 3:]
        euler = quaternion_to_euler(quat)
        print(f"  Pose {i}: pos={pos}, quat={quat}, euler_deg={euler}")
    
    # Check if first pose is at origin
    first_pos = gt_poses[0, :3]
    first_quat = gt_poses[0, 3:]
    print(f"\nFirst pose position: {first_pos}")
    print(f"First pose quaternion: {first_quat}")
    print(f"Is first pose at origin? pos_norm={np.linalg.norm(first_pos):.6f}")
    
    # Check if quaternion is identity (or close to it)
    identity_quat = np.array([0, 0, 0, 1])  # x, y, z, w format
    quat_diff = np.linalg.norm(first_quat - identity_quat)
    print(f"Quaternion difference from identity: {quat_diff:.6f}")
    
    # Analyze rotation magnitudes between consecutive frames
    print(f"\nRotation angles between consecutive frames (degrees):")
    rotation_angles = []
    for i in range(1, min(10, len(gt_poses))):
        q1 = gt_poses[i-1, 3:]
        q2 = gt_poses[i, 3:]
        
        # Compute relative rotation
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        r_rel = r1.inv() * r2
        
        # Get rotation angle
        angle_rad = r_rel.magnitude()
        angle_deg = np.degrees(angle_rad)
        rotation_angles.append(angle_deg)
        print(f"  Frame {i-1} to {i}: {angle_deg:.4f}°")
    
    if rotation_angles:
        print(f"\nRotation statistics:")
        print(f"  Mean: {np.mean(rotation_angles):.4f}°")
        print(f"  Max: {np.max(rotation_angles):.4f}°")
        print(f"  Min: {np.min(rotation_angles):.4f}°")
    
    # Check translation magnitudes
    print(f"\nTranslation magnitudes between consecutive frames (cm):")
    translation_mags = []
    for i in range(1, min(10, len(gt_poses))):
        trans = gt_poses[i, :3] - gt_poses[i-1, :3]
        mag = np.linalg.norm(trans)
        translation_mags.append(mag)
        print(f"  Frame {i-1} to {i}: {mag:.4f} cm")
    
    if translation_mags:
        print(f"\nTranslation statistics:")
        print(f"  Mean: {np.mean(translation_mags):.4f} cm")
        print(f"  Max: {np.max(translation_mags):.4f} cm")
        print(f"  Min: {np.min(translation_mags):.4f} cm")
    
    return gt_poses

def check_quaternion_format(samples_to_check, data_dir):
    """Check quaternion format consistency across samples"""
    print("\n=== Checking Quaternion Format ===")
    
    for idx in samples_to_check:
        gt_poses = np.load(os.path.join(data_dir, f"{idx}_gt.npy"))
        
        # Check quaternion norms
        for i in range(min(5, len(gt_poses))):
            quat = gt_poses[i, 3:]
            norm = np.linalg.norm(quat)
            if abs(norm - 1.0) > 0.01:
                print(f"WARNING: Sample {idx}, pose {i} has non-unit quaternion! Norm={norm}")

def main():
    data_dir = "/home/external/VIFT_AEA/aria_latent_data_pretrained/train"
    
    # Analyze several random samples
    samples_to_check = [0, 100, 500, 1000, 5000, 10000]
    
    all_first_poses = []
    for idx in samples_to_check:
        gt_poses = analyze_sample(idx, data_dir)
        all_first_poses.append(gt_poses[0])
    
    # Check quaternion format
    check_quaternion_format(samples_to_check, data_dir)
    
    # Check if all samples start at origin
    print("\n=== Summary of First Poses ===")
    for i, idx in enumerate(samples_to_check):
        pos = all_first_poses[i][:3]
        quat = all_first_poses[i][3:]
        print(f"Sample {idx}: pos_norm={np.linalg.norm(pos):.6f}, quat={quat}")

if __name__ == "__main__":
    main()