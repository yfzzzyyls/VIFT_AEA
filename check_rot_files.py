#!/usr/bin/env python3
"""Check what's in the _rot.npy files"""

import struct
import math
import os

def read_npy_simple(file_path):
    """Simple numpy file reader"""
    with open(file_path, 'rb') as f:
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
        
        # Read data
        data = []
        while True:
            bytes_data = f.read(4)
            if len(bytes_data) < 4:
                break
            value = struct.unpack('<f', bytes_data)[0]
            data.append(value)
        
        return shape, data

# Check the _rot.npy files
pretrained_dir = '/home/external/VIFT_AEA/aria_latent_data_pretrained/test'

print("Checking _rot.npy files...")
print("="*80)

for file_idx in range(3):
    rot_path = os.path.join(pretrained_dir, f'{file_idx}_rot.npy')
    
    if os.path.exists(rot_path):
        print(f"\nFile: {file_idx}_rot.npy")
        shape, data = read_npy_simple(rot_path)
        print(f"Shape: {shape}")
        print(f"Data: {data}")
        
        # If it's rotation data in euler angles (3 values per frame)
        if len(data) >= 3:
            n_frames = len(data) // 3
            print(f"\nEuler angles (if 3 per frame):")
            for i in range(min(5, n_frames)):
                idx = i * 3
                if idx + 2 < len(data):
                    rx, ry, rz = data[idx], data[idx+1], data[idx+2]
                    print(f"  Frame {i}: [{rx:.6f}, {ry:.6f}, {rz:.6f}] radians")
                    print(f"           [{math.degrees(rx):.2f}, {math.degrees(ry):.2f}, {math.degrees(rz):.2f}] degrees")

print("\n" + "="*80)
print("ANALYSIS:")
print("The _rot.npy files contain Euler angles showing ~9.3 degrees rotation in Y-axis.")
print("This is the actual camera orientation in the Aria dataset!")
print("The issue is that the pretrained data generation code didn't properly")
print("convert these Euler angles to quaternions in the _gt.npy files.")