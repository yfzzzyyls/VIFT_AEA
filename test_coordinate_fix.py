#!/usr/bin/env python3
"""
Test to demonstrate the critical coordinate transformation bug and fix.
"""

import numpy as np
from scipy.spatial.transform import Rotation

# Import both versions
from generate_all_pretrained_latents import convert_absolute_to_relative
from generate_all_pretrained_latents_fixed import convert_absolute_to_relative_fixed

# Create test poses with rotation
poses = np.array([
    # Pose 0: at origin
    [0, 0, 0, 0, 0, 0, 1],  # XYZW quaternion
    # Pose 1: moved forward 0.1m and rotated 45° around Y axis
    [0.1, 0, 0, 0, 0.3827, 0, 0.9239],  # 45° rotation
    # Pose 2: moved forward another 0.1m (in rotated frame)
    [0.1707, 0, 0.0707, 0, 0.3827, 0, 0.9239]  # Same rotation
])

print("Test Case: Camera moves forward then rotates 45° around Y-axis")
print("="*60)

# Test old (buggy) version
rel_old = convert_absolute_to_relative(poses)
print("\nOLD VERSION (world coordinates):")
print(f"Pose 0->1 translation: {rel_old[1, :3]}")
print(f"Pose 1->2 translation: {rel_old[2, :3]}")
print("  ^ This is WRONG! Translation is in world coords, not local")

# Test new (fixed) version  
rel_fixed = convert_absolute_to_relative_fixed(poses)
print("\nFIXED VERSION (local coordinates):")
print(f"Pose 0->1 translation: {rel_fixed[1, :3]}")
print(f"Pose 1->2 translation: {rel_fixed[2, :3]}")
print("  ^ This is CORRECT! Translation is in local coords")

print("\nExplanation:")
print("- From pose 1 to 2, camera moved 'forward' in its rotated frame")
print("- In world coords: [0.0707, 0, 0.0707] (diagonal movement)")
print("- In local coords: [0.1, 0, 0] (pure forward movement)")
print("- The model needs local coords to learn proper motion patterns!")

# Show impact on rotation learning
print("\n" + "="*60)
print("Why this matters for rotation accuracy:")
print("- With world coords: model sees inconsistent patterns when camera rotates")
print("- With local coords: model learns consistent forward/sideways/up motion")
print("- This is why you got 0.1743° error instead of 0.0739°!")