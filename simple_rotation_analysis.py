#!/usr/bin/env python3
"""Simple rotation analysis without numpy dependency"""

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
    # Header format: {'descr': '<f4', 'fortran_order': False, 'shape': (N, 7), }
    shape_start = header_str.find("'shape': ") + 9
    shape_end = header_str.find(")", shape_start) + 1
    shape_str = header_str[shape_start:shape_end]
    
    # Extract numbers from shape
    shape = eval(shape_str)  # Safe here as we control the input
    
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
    # For unit quaternion, inverse is conjugate
    w = w2*w1 + x2*x1 + y2*y1 + z2*z1
    x = -w2*x1 + x2*w1 - y2*z1 + z2*y1
    y = -w2*y1 + x2*z1 + y2*w1 - z2*x1
    z = -w2*z1 - x2*y1 + y2*x1 + z2*w1
    
    return quaternion_to_angle(w, x, y, z)

def analyze_file(file_path):
    """Analyze a single .npy file"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    with open(file_path, 'rb') as f:
        shape, data_start = read_npy_header(f)
        print(f"Shape: {shape}")
        
        if len(shape) != 2 or shape[1] != 7:
            print("ERROR: Expected shape (N, 7) for pose data")
            return
        
        n_frames = shape[0]
        n_elements = n_frames * 7
        
        # Read all data
        f.seek(data_start)
        data = read_float32_array(f, n_elements)
        
        if len(data) != n_elements:
            print(f"ERROR: Expected {n_elements} elements, got {len(data)}")
            return
        
        # Print first 5 frames
        print(f"\nFirst 5 frames:")
        print(f"{'Frame':<6} {'Trans X':<10} {'Trans Y':<10} {'Trans Z':<10} {'Quat W':<10} {'Quat X':<10} {'Quat Y':<10} {'Quat Z':<10} {'Angle (deg)':<12}")
        print("-" * 108)
        
        for i in range(min(5, n_frames)):
            idx = i * 7
            tx, ty, tz = data[idx], data[idx+1], data[idx+2]
            qw, qx, qy, qz = data[idx+3], data[idx+4], data[idx+5], data[idx+6]
            angle = quaternion_to_angle(qw, qx, qy, qz)
            
            print(f"{i:<6} {tx:<10.6f} {ty:<10.6f} {tz:<10.6f} "
                  f"{qw:<10.6f} {qx:<10.6f} {qy:<10.6f} {qz:<10.6f} "
                  f"{angle:<12.6f}")
        
        # Analyze consecutive frame differences
        print(f"\nConsecutive frame differences:")
        print(f"{'Frames':<10} {'Trans Diff (m)':<15} {'Rot Diff (deg)':<15} {'Trans X Diff':<12} {'Trans Y Diff':<12} {'Trans Z Diff':<12}")
        print("-" * 76)
        
        trans_diffs = []
        rot_diffs = []
        
        for i in range(min(10, n_frames-1)):
            idx1 = i * 7
            idx2 = (i + 1) * 7
            
            # Translation difference
            dx = data[idx2] - data[idx1]
            dy = data[idx2+1] - data[idx1+1]
            dz = data[idx2+2] - data[idx1+2]
            trans_diff = math.sqrt(dx*dx + dy*dy + dz*dz)
            trans_diffs.append(trans_diff)
            
            # Rotation difference
            q1 = (data[idx1+3], data[idx1+4], data[idx1+5], data[idx1+6])
            q2 = (data[idx2+3], data[idx2+4], data[idx2+5], data[idx2+6])
            rot_diff = quaternion_difference(q1, q2)
            rot_diffs.append(rot_diff)
            
            print(f"{i}->{i+1:<7} {trans_diff:<15.6f} {rot_diff:<15.6f} {dx:<12.6f} {dy:<12.6f} {dz:<12.6f}")
        
        # Calculate statistics
        if trans_diffs:
            trans_mean = sum(trans_diffs) / len(trans_diffs)
            trans_min = min(trans_diffs)
            trans_max = max(trans_diffs)
            
            print(f"\nTranslation statistics (meters):")
            print(f"  Mean: {trans_mean:.6f}")
            print(f"  Min:  {trans_min:.6f}")
            print(f"  Max:  {trans_max:.6f}")
        
        if rot_diffs:
            rot_mean = sum(rot_diffs) / len(rot_diffs)
            rot_min = min(rot_diffs)
            rot_max = max(rot_diffs)
            
            print(f"\nRotation statistics (degrees):")
            print(f"  Mean: {rot_mean:.6f}")
            print(f"  Min:  {rot_min:.6f}")
            print(f"  Max:  {rot_max:.6f}")
        
        # Check quaternion norms
        print(f"\nQuaternion norms (should be ~1.0):")
        norms = []
        for i in range(min(20, n_frames)):
            idx = i * 7
            qw, qx, qy, qz = data[idx+3], data[idx+4], data[idx+5], data[idx+6]
            norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
            norms.append(norm)
        
        if norms:
            norm_mean = sum(norms) / len(norms)
            norm_min = min(norms)
            norm_max = max(norms)
            print(f"  Mean: {norm_mean:.6f}")
            print(f"  Min:  {norm_min:.6f}")
            print(f"  Max:  {norm_max:.6f}")

# Main execution
if __name__ == "__main__":
    files = [
        '/home/external/VIFT_AEA/aria_latent_data_properly_fixed/test/7_gt.npy',
        '/home/external/VIFT_AEA/aria_latent_data_properly_fixed/test/8_gt.npy',
        '/home/external/VIFT_AEA/aria_latent_data_properly_fixed/test/9_gt.npy',
        '/home/external/VIFT_AEA/aria_latent_data_properly_fixed/test/4_gt.npy',
        '/home/external/VIFT_AEA/aria_latent_data_properly_fixed/test/5_gt.npy'
    ]
    
    for file_path in files:
        try:
            analyze_file(file_path)
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")