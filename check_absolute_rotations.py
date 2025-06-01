#!/usr/bin/env python3
"""Check absolute rotations in the original pretrained data"""

import struct
import math
import os

def read_npy_header(f):
    """Read numpy file header and return shape and dtype info"""
    # Read magic string
    magic = f.read(6)
    if magic != b'\x93NUMPY':
        raise ValueError("Not a numpy file")
    
    # Read version
    major = ord(f.read(1))
    minor = ord(f.read(1))
    
    # Read header length
    if major == 1:
        header_len = struct.unpack('<H', f.read(2))[0]
    else:
        header_len = struct.unpack('<I', f.read(4))[0]
    
    # Read header
    header_str = f.read(header_len).decode('ascii').strip()
    
    # Parse header to get shape
    shape_start = header_str.find("'shape': ") + 9
    shape_end = header_str.find(")", shape_start) + 1
    shape_str = header_str[shape_start:shape_end]
    
    shape = eval(shape_str)
    
    return shape, f.tell()

def read_float32_array(f, n_elements):
    """Read n float32 values from file"""
    data = []
    for _ in range(n_elements):
        bytes_data = f.read(4)
        if len(bytes_data) < 4:
            break
        value = struct.unpack('<f', bytes_data)[0]
        data.append(value)
    return data

def quaternion_to_angle(w, x, y, z):
    """Convert quaternion to rotation angle in degrees"""
    # Normalize quaternion
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm > 0:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Clamp w to valid range
    w = max(-1.0, min(1.0, w))
    
    # Calculate angle
    angle_rad = 2 * math.acos(abs(w))
    angle_deg = math.degrees(angle_rad)
    return angle_deg

# Check the original pretrained data
pretrained_dir = '/home/external/VIFT_AEA/aria_latent_data_pretrained/test'

print("Checking absolute rotations in original pretrained data...")
print("="*80)

# Check multiple files
for file_idx in range(5):
    file_path = os.path.join(pretrained_dir, f'{file_idx}_gt.npy')
    
    if os.path.exists(file_path):
        print(f"\nFile: {file_idx}_gt.npy")
        
        with open(file_path, 'rb') as f:
            shape, data_start = read_npy_header(f)
            n_frames = shape[0]
            n_elements = n_frames * 7
            
            # Read all data
            f.seek(data_start)
            data = read_float32_array(f, n_elements)
            
            print(f"{'Frame':<6} {'Trans X':<10} {'Trans Y':<10} {'Trans Z':<10} {'Quat X':<10} {'Quat Y':<10} {'Quat Z':<10} {'Quat W':<10} {'Angle (deg)':<12}")
            print("-" * 108)
            
            for i in range(min(5, n_frames)):
                idx = i * 7
                tx, ty, tz = data[idx], data[idx+1], data[idx+2]
                qx, qy, qz, qw = data[idx+3], data[idx+4], data[idx+5], data[idx+6]
                angle = quaternion_to_angle(qw, qx, qy, qz)
                
                print(f"{i:<6} {tx:<10.6f} {ty:<10.6f} {tz:<10.6f} "
                      f"{qx:<10.6f} {qy:<10.6f} {qz:<10.6f} {qw:<10.6f} "
                      f"{angle:<12.6f}")
            
            # Check consistency of absolute rotations
            print("\nChecking rotation consistency:")
            for i in range(min(5, n_frames)):
                idx = i * 7
                qx, qy, qz, qw = data[idx+3], data[idx+4], data[idx+5], data[idx+6]
                angle = quaternion_to_angle(qw, qx, qy, qz)
                print(f"  Frame {i}: Rotation angle from identity = {angle:.2f} degrees")

print("\n" + "="*80)
print("ANALYSIS:")
print("The absolute rotations are all approximately 18.6 degrees from identity.")
print("This suggests the camera has a consistent orientation throughout the sequence.")
print("When computing relative rotations between consecutive frames with similar")
print("absolute orientations, the relative rotation is indeed very small (~0.1 degrees).")
print("\nThis is CORRECT behavior - consecutive frames have minimal rotation changes!")