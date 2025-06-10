#!/usr/bin/env python3
"""Debug script to understand scaling issue."""

import json
import numpy as np

# Load original poses
with open('data/aria_processed/008/poses_quaternion.json', 'r') as f:
    poses = json.load(f)

# Extract first 100 frames with stride 20
frames = []
for i in range(0, min(100, len(poses)), 20):
    t = poses[i]['translation']
    q = poses[i]['quaternion']
    frames.append([t[0], t[1], t[2], q[0], q[1], q[2], q[3]])

frames = np.array(frames)
print(f"Absolute poses shape: {frames.shape}")
print("\nFirst 5 absolute translations (meters):")
for i in range(min(5, len(frames))):
    print(f"  Frame {i*20}: {frames[i, :3]} (norm: {np.linalg.norm(frames[i, :3]):.4f} m)")

# Compute relative poses manually
print("\nRelative translations (consecutive frames with stride 20):")
for i in range(1, min(5, len(frames))):
    rel_trans = frames[i, :3] - frames[i-1, :3]
    print(f"  Frame {(i-1)*20}->{i*20}: {rel_trans} (norm: {np.linalg.norm(rel_trans):.6f} m)")
    print(f"    In cm: {rel_trans * 100} (norm: {np.linalg.norm(rel_trans * 100):.4f} cm)")

# This is what we expect after proper relative pose computation and scaling
print("\nExpected relative pose magnitudes:")
print("- Walking speed: ~1.4 m/s = 140 cm/s")
print("- With stride 20 (1 second): ~140 cm relative motion")
print("- But we're seeing: ~0.001-0.005 cm which is 1000x too small!")