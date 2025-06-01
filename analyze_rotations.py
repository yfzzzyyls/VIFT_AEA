import numpy as np
import os

# Load the specified .npy files
files = [
    '/home/external/VIFT_AEA/aria_latent_data_properly_fixed/test/7_gt.npy',
    '/home/external/VIFT_AEA/aria_latent_data_properly_fixed/test/8_gt.npy',
    '/home/external/VIFT_AEA/aria_latent_data_properly_fixed/test/9_gt.npy',
    '/home/external/VIFT_AEA/aria_latent_data_properly_fixed/test/4_gt.npy',
    '/home/external/VIFT_AEA/aria_latent_data_properly_fixed/test/5_gt.npy'
]

# Function to calculate rotation angle from quaternion
def quaternion_to_angle(q):
    """Convert quaternion to rotation angle in degrees"""
    # q = [w, x, y, z]
    w = q[0]
    # Clamp w to valid range [-1, 1] to avoid numerical issues
    w = np.clip(w, -1.0, 1.0)
    # angle = 2 * arccos(w)
    angle_rad = 2 * np.arccos(np.abs(w))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Function to calculate rotation difference between two quaternions
def quaternion_difference(q1, q2):
    """Calculate the rotation angle between two quaternions"""
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Calculate relative rotation: q_diff = q2 * q1^(-1)
    # For unit quaternion, inverse is conjugate: q^(-1) = [w, -x, -y, -z]
    q1_inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
    
    # Quaternion multiplication
    w = q2[0]*q1_inv[0] - q2[1]*q1_inv[1] - q2[2]*q1_inv[2] - q2[3]*q1_inv[3]
    x = q2[0]*q1_inv[1] + q2[1]*q1_inv[0] + q2[2]*q1_inv[3] - q2[3]*q1_inv[2]
    y = q2[0]*q1_inv[2] - q2[1]*q1_inv[3] + q2[2]*q1_inv[0] + q2[3]*q1_inv[1]
    z = q2[0]*q1_inv[3] + q2[1]*q1_inv[2] - q2[2]*q1_inv[1] + q2[3]*q1_inv[0]
    
    q_diff = np.array([w, x, y, z])
    return quaternion_to_angle(q_diff)

# Analyze each file
for file_path in files:
    print(f"\n{'='*60}")
    print(f"Analyzing: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    # Load data
    data = np.load(file_path)
    print(f"Shape: {data.shape}")
    
    # Extract translations and quaternions
    translations = data[:, :3]
    quaternions = data[:, 3:7]
    
    # Print first few frames
    print(f"\nFirst 5 frames:")
    print(f"{'Frame':<6} {'Trans X':<10} {'Trans Y':<10} {'Trans Z':<10} {'Quat W':<10} {'Quat X':<10} {'Quat Y':<10} {'Quat Z':<10} {'Angle (deg)':<12}")
    print("-" * 108)
    
    for i in range(min(5, len(data))):
        angle = quaternion_to_angle(quaternions[i])
        print(f"{i:<6} {translations[i,0]:<10.6f} {translations[i,1]:<10.6f} {translations[i,2]:<10.6f} "
              f"{quaternions[i,0]:<10.6f} {quaternions[i,1]:<10.6f} {quaternions[i,2]:<10.6f} {quaternions[i,3]:<10.6f} "
              f"{angle:<12.6f}")
    
    # Analyze differences between consecutive frames
    print(f"\nConsecutive frame differences:")
    print(f"{'Frames':<10} {'Trans Diff (m)':<15} {'Rot Diff (deg)':<15} {'Trans X Diff':<12} {'Trans Y Diff':<12} {'Trans Z Diff':<12}")
    print("-" * 76)
    
    for i in range(min(10, len(data)-1)):
        trans_diff = translations[i+1] - translations[i]
        trans_diff_mag = np.linalg.norm(trans_diff)
        rot_diff = quaternion_difference(quaternions[i], quaternions[i+1])
        
        print(f"{i}->{i+1:<7} {trans_diff_mag:<15.6f} {rot_diff:<15.6f} {trans_diff[0]:<12.6f} {trans_diff[1]:<12.6f} {trans_diff[2]:<12.6f}")
    
    # Statistics
    print(f"\nStatistics for {os.path.basename(file_path)}:")
    
    # Translation statistics
    trans_diffs = []
    for i in range(len(data)-1):
        trans_diff = np.linalg.norm(translations[i+1] - translations[i])
        trans_diffs.append(trans_diff)
    
    trans_diffs = np.array(trans_diffs)
    print(f"Translation differences (meters):")
    print(f"  Mean: {np.mean(trans_diffs):.6f}")
    print(f"  Std:  {np.std(trans_diffs):.6f}")
    print(f"  Min:  {np.min(trans_diffs):.6f}")
    print(f"  Max:  {np.max(trans_diffs):.6f}")
    
    # Rotation statistics
    rot_diffs = []
    for i in range(len(data)-1):
        rot_diff = quaternion_difference(quaternions[i], quaternions[i+1])
        rot_diffs.append(rot_diff)
    
    rot_diffs = np.array(rot_diffs)
    print(f"\nRotation differences (degrees):")
    print(f"  Mean: {np.mean(rot_diffs):.6f}")
    print(f"  Std:  {np.std(rot_diffs):.6f}")
    print(f"  Min:  {np.min(rot_diffs):.6f}")
    print(f"  Max:  {np.max(rot_diffs):.6f}")
    
    # Check quaternion norms
    quat_norms = np.linalg.norm(quaternions, axis=1)
    print(f"\nQuaternion norms (should be ~1.0):")
    print(f"  Mean: {np.mean(quat_norms):.6f}")
    print(f"  Std:  {np.std(quat_norms):.6f}")
    print(f"  Min:  {np.min(quat_norms):.6f}")
    print(f"  Max:  {np.max(quat_norms):.6f}")