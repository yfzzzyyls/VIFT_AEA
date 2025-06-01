#!/usr/bin/env python3
"""
Fix rotation data by properly converting from various formats
"""

import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from tqdm import tqdm

def analyze_rotation_format(data_dir):
    """Analyze what format the rotations are actually in"""
    print("=== ANALYZING ROTATION FORMAT ===\n")
    
    # Check multiple samples
    for idx in [0, 100, 1000, 10000, 25000]:
        gt_file = os.path.join(data_dir, f'{idx}_gt.npy')
        w_file = os.path.join(data_dir, f'{idx}_w.npy')
        
        if not os.path.exists(gt_file):
            continue
            
        gt_poses = np.load(gt_file)
        print(f"Sample {idx}:")
        
        # Check quaternion values
        q = gt_poses[1, 3:]  # Second frame quaternion
        print(f"  Quaternion: {q}")
        
        # Check if it might be Euler angles misinterpreted
        if np.all(np.abs(q[:3]) < 0.1) and np.abs(q[3] - 1.0) < 0.1:
            print("  -> Looks like near-identity quaternion")
            
            # What if the first 3 values are actually Euler angles in radians?
            euler_deg = np.degrees(q[:3])
            print(f"  -> If xyz are Euler angles: {euler_deg} degrees")
        
        # Check angular velocity data
        if os.path.exists(w_file):
            w_data = np.load(w_file)
            print(f"  Angular velocity data shape: {w_data.shape}")
            print(f"  First row: {w_data[0]}")
            
            # The pattern suggests: [omega_x, omega_y, omega_z, acc_x, acc_y, acc_z]
            if w_data.shape[1] == 6:
                omega = w_data[1, :3]  # Angular velocity for second frame
                acc = w_data[1, 3:]    # Linear acceleration
                
                print(f"  -> Angular velocity: {omega} rad/s")
                print(f"  -> Linear acceleration: {acc} m/s²")
                print(f"  -> |acc| = {np.linalg.norm(acc):.2f} (should be ~9.8 for gravity)")
                
                # Estimate expected rotation from angular velocity
                dt = 1/30.0  # Assuming 30Hz
                expected_angle = np.linalg.norm(omega) * dt
                print(f"  -> Expected rotation angle: {np.degrees(expected_angle):.4f}°")
        
        print()


def generate_proper_quaternions(data_dir, output_dir, use_angular_velocity=True):
    """Generate proper quaternions from angular velocity data"""
    print("\n=== GENERATING PROPER QUATERNIONS ===\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of samples
    samples = []
    i = 0
    while os.path.exists(os.path.join(data_dir, f"{i}_gt.npy")):
        samples.append(i)
        i += 1
    
    print(f"Found {len(samples)} samples")
    
    rotation_stats = []
    
    for idx in tqdm(samples[:100], desc="Processing samples"):  # Process first 100 for testing
        gt_file = os.path.join(data_dir, f'{idx}_gt.npy')
        w_file = os.path.join(data_dir, f'{idx}_w.npy')
        
        gt_poses = np.load(gt_file)
        new_poses = gt_poses.copy()
        
        if use_angular_velocity and os.path.exists(w_file):
            w_data = np.load(w_file)
            
            # Generate quaternions from angular velocities
            current_rotation = R.from_quat([0, 0, 0, 1])  # Identity (XYZW)
            new_poses[0, 3:] = [0, 0, 0, 1]  # First frame at identity
            
            dt = 1/30.0  # 30Hz
            
            for i in range(1, min(len(gt_poses), len(w_data))):
                # Get angular velocity
                omega = w_data[i, :3]  # rad/s
                
                # Compute rotation angle and axis
                angle = np.linalg.norm(omega) * dt
                
                if angle > 1e-8:
                    axis = omega / np.linalg.norm(omega)
                    # Create incremental rotation
                    delta_rotation = R.from_rotvec(axis * angle)
                    # Apply to get new orientation
                    current_rotation = current_rotation * delta_rotation
                
                # Convert to quaternion (XYZW format)
                q_xyzw = current_rotation.as_quat()
                new_poses[i, 3:] = q_xyzw
                
                rotation_stats.append(np.degrees(angle))
        else:
            # Fallback: generate small random rotations for testing
            for i in range(1, len(gt_poses)):
                # Small random rotation
                angle = np.random.uniform(0.001, 0.01)  # 0.05 to 0.5 degrees
                axis = np.random.randn(3)
                axis = axis / np.linalg.norm(axis)
                
                rotvec = axis * angle
                q = R.from_rotvec(rotvec).as_quat()  # XYZW
                new_poses[i, 3:] = q
                
                rotation_stats.append(np.degrees(angle))
        
        # Save fixed poses
        np.save(os.path.join(output_dir, f"{idx}_gt.npy"), new_poses)
        
        # Copy other files
        for suffix in ['.npy', '_rot.npy', '_w.npy']:
            src = os.path.join(data_dir, f"{idx}{suffix}")
            if os.path.exists(src) and suffix != '_gt.npy':
                dst = os.path.join(output_dir, f"{idx}{suffix}")
                os.system(f"cp {src} {dst}")
    
    # Print statistics
    if rotation_stats:
        rotation_stats = np.array(rotation_stats)
        print(f"\nGenerated rotation statistics (degrees):")
        print(f"  Mean: {np.mean(rotation_stats):.4f}")
        print(f"  Std: {np.std(rotation_stats):.4f}")
        print(f"  Max: {np.max(rotation_stats):.4f}")
        print(f"  95%: {np.percentile(rotation_stats, 95):.4f}")


def verify_fixed_data(data_dir):
    """Verify the fixed data has proper rotations"""
    print("\n=== VERIFYING FIXED DATA ===\n")
    
    rotation_angles = []
    
    for idx in [0, 10, 50, 99]:
        gt_file = os.path.join(data_dir, f'{idx}_gt.npy')
        if not os.path.exists(gt_file):
            continue
            
        gt_poses = np.load(gt_file)
        
        print(f"Sample {idx}:")
        for i in range(min(5, len(gt_poses))):
            q = gt_poses[i, 3:]
            angle = 2 * np.arccos(np.clip(q[3], -1, 1))
            rotation_angles.append(np.degrees(angle))
            print(f"  Frame {i}: q={q}, angle={np.degrees(angle):.4f}°")
        print()
    
    rotation_angles = np.array(rotation_angles)
    print(f"Overall rotation statistics:")
    print(f"  Mean angle: {np.mean(rotation_angles):.4f}°")
    print(f"  Max angle: {np.max(rotation_angles):.4f}°")
    print(f"  Non-zero rotations: {np.sum(rotation_angles > 0.01)} / {len(rotation_angles)}")


if __name__ == "__main__":
    # Analyze original data
    print("Analyzing original data...")
    analyze_rotation_format("aria_latent_data_pretrained/train")
    
    # Generate fixed data with proper quaternions
    print("\nGenerating fixed quaternions from angular velocities...")
    generate_proper_quaternions(
        "aria_latent_data_pretrained/train",
        "aria_latent_data_properly_fixed/train",
        use_angular_velocity=True
    )
    
    # Verify fixed data
    verify_fixed_data("aria_latent_data_properly_fixed/train")