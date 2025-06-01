#!/usr/bin/env python3
"""Extended rotation analysis to understand the pattern"""

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
    
    # Parse header to get shape and dtype
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

def quaternion_difference(q1, q2):
    """Calculate rotation angle between two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    # Normalize
    norm1 = math.sqrt(w1*w1 + x1*x1 + y1*y1 + z1*z1)
    norm2 = math.sqrt(w2*w2 + x2*x2 + y2*y2 + z2*z2)
    
    if norm1 > 0:
        w1, x1, y1, z1 = w1/norm1, x1/norm1, y1/norm1, z1/norm1
    if norm2 > 0:
        w2, x2, y2, z2 = w2/norm2, x2/norm2, y2/norm2, z2/norm2
    
    # Calculate q2 * q1^(-1)
    w = w2*w1 + x2*x1 + y2*y1 + z2*z1
    x = -w2*x1 + x2*w1 - y2*z1 + z2*y1
    y = -w2*y1 + x2*z1 + y2*w1 - z2*x1
    z = -w2*z1 - x2*y1 + y2*x1 + z2*w1
    
    return quaternion_to_angle(w, x, y, z)

def analyze_longer_file(file_path):
    """Analyze a file with more frames"""
    with open(file_path, 'rb') as f:
        shape, data_start = read_npy_header(f)
        n_frames = shape[0]
        n_elements = n_frames * 7
        
        # Read all data
        f.seek(data_start)
        data = read_float32_array(f, n_elements)
        
        return data, n_frames

# Find a file with more frames
test_dir = '/home/external/VIFT_AEA/aria_latent_data_properly_fixed/test'

# List available files
print("Looking for larger files...")
for i in range(100, 120):  # Check files 100-119
    file_path = os.path.join(test_dir, f'{i}_gt.npy')
    if os.path.exists(file_path):
        try:
            data, n_frames = analyze_longer_file(file_path)
            if n_frames > 11:  # Found a longer sequence
                print(f"\nAnalyzing file {i}_gt.npy with {n_frames} frames")
                
                # Check the pattern of the first quaternion
                print("\nFirst quaternion (identity) pattern:")
                print(f"{'Frame':<6} {'Quat W':<10} {'Quat X':<10} {'Quat Y':<10} {'Quat Z':<10} {'Is Identity':<12}")
                print("-" * 60)
                
                identity_count = 0
                for j in range(min(n_frames, 20)):
                    idx = j * 7
                    qw, qx, qy, qz = data[idx+3], data[idx+4], data[idx+5], data[idx+6]
                    is_identity = (abs(qw) < 0.001 and abs(qx) < 0.001 and 
                                   abs(qy) < 0.001 and abs(qz-1.0) < 0.001)
                    if is_identity:
                        identity_count += 1
                    print(f"{j:<6} {qw:<10.6f} {qx:<10.6f} {qy:<10.6f} {qz:<10.6f} {'Yes' if is_identity else 'No':<12}")
                
                # Analyze the jump from identity to actual orientation
                print(f"\nFound {identity_count} identity quaternions at the start")
                
                # Find first non-identity frame
                first_non_identity = -1
                for j in range(n_frames):
                    idx = j * 7
                    qw, qx, qy, qz = data[idx+3], data[idx+4], data[idx+5], data[idx+6]
                    if not (abs(qw) < 0.001 and abs(qx) < 0.001 and 
                            abs(qy) < 0.001 and abs(qz-1.0) < 0.001):
                        first_non_identity = j
                        break
                
                if first_non_identity > 0:
                    print(f"\nFirst non-identity quaternion at frame {first_non_identity}")
                    idx = first_non_identity * 7
                    qw, qx, qy, qz = data[idx+3], data[idx+4], data[idx+5], data[idx+6]
                    angle = quaternion_to_angle(qw, qx, qy, qz)
                    print(f"Quaternion: [{qw:.6f}, {qx:.6f}, {qy:.6f}, {qz:.6f}]")
                    print(f"Rotation angle: {angle:.2f} degrees")
                    
                    # Check consistency after the jump
                    print("\nRotation differences after the initial jump:")
                    for k in range(min(10, n_frames - first_non_identity - 1)):
                        j = first_non_identity + k
                        idx1 = j * 7
                        idx2 = (j + 1) * 7
                        
                        q1 = (data[idx1+3], data[idx1+4], data[idx1+5], data[idx1+6])
                        q2 = (data[idx2+3], data[idx2+4], data[idx2+5], data[idx2+6])
                        rot_diff = quaternion_difference(q1, q2)
                        
                        print(f"  Frame {j}->{j+1}: {rot_diff:.6f} degrees")
                
                break
        except Exception as e:
            pass

# Also check the stride pattern
print("\n\nChecking stride pattern in sequences...")
print("Looking at how frames are sampled (stride=10):")

# Compare consecutive file indices
for base_idx in [0, 10, 20]:
    print(f"\nSequence starting at index {base_idx}:")
    for i in range(5):
        file_idx = base_idx + i
        file_path = os.path.join(test_dir, f'{file_idx}_gt.npy')
        if os.path.exists(file_path):
            try:
                data, n_frames = analyze_longer_file(file_path)
                # Get first non-identity quaternion
                for j in range(n_frames):
                    idx = j * 7
                    qw, qx, qy, qz = data[idx+3], data[idx+4], data[idx+5], data[idx+6]
                    if not (abs(qw) < 0.001 and abs(qx) < 0.001 and 
                            abs(qy) < 0.001 and abs(qz-1.0) < 0.001):
                        print(f"  File {file_idx}: First actual quaternion = [{qw:.6f}, {qx:.6f}, {qy:.6f}, {qz:.6f}]")
                        break
            except Exception as e:
                pass