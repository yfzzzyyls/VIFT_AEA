#!/usr/bin/env python3
import numpy as np
import sys

# Load the data
poses = np.load('aria_latent_data_pretrained/train/0_gt.npy')
print(f'Shape: {poses.shape}')
print(f'\nFirst 5 poses:')
for i in range(min(5, len(poses))):
    print(f'Pose {i}: [{poses[i][0]:.6f}, {poses[i][1]:.6f}, {poses[i][2]:.6f}, {poses[i][3]:.6f}, {poses[i][4]:.6f}, {poses[i][5]:.6f}, {poses[i][6]:.6f}]')

# Check if first pose is at origin
print(f'\nFirst pose analysis:')
print(f'Translation: [{poses[0, 0]:.6f}, {poses[0, 1]:.6f}, {poses[0, 2]:.6f}]')
print(f'Quaternion: [{poses[0, 3]:.6f}, {poses[0, 4]:.6f}, {poses[0, 5]:.6f}, {poses[0, 6]:.6f}]')
print(f'Is at origin? Translation={np.allclose(poses[0, :3], 0, atol=1e-6)} Quaternion={np.allclose(poses[0, 3:], [0, 0, 0, 1], atol=1e-6)}')

# Check quaternion normalization
print(f'\nQuaternion normalization check:')
for i in range(min(5, len(poses))):
    q = poses[i, 3:]
    norm = np.linalg.norm(q)
    print(f'Pose {i} quaternion norm: {norm:.6f}')

# Check relative rotations between consecutive frames
print(f'\nRelative rotation analysis:')
for i in range(1, min(5, len(poses))):
    q1 = poses[i-1, 3:]
    q2 = poses[i, 3:]
    # Compute relative rotation magnitude
    dot = np.dot(q1, q2)
    angle = 2 * np.arccos(np.clip(abs(dot), -1, 1)) * 180 / np.pi
    print(f'Angle between pose {i-1} and {i}: {angle:.4f} degrees')

# Check translation magnitudes
print(f'\nTranslation analysis (in cm):')
for i in range(min(5, len(poses))):
    trans = poses[i, :3]
    mag = np.linalg.norm(trans)
    print(f'Pose {i} translation magnitude: {mag:.4f} cm')

# Check if we need to scale
print(f'\nChecking if data needs scaling:')
max_trans = np.max(np.abs(poses[:, :3]))
print(f'Max absolute translation value: {max_trans:.6f}')
if max_trans < 1.0:
    print('WARNING: Translations appear to be in meters, not centimeters!')