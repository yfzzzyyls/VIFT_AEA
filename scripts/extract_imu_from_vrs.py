#!/usr/bin/env python3
"""
Extract real IMU data from Aria recording.vrs files
Requires Project Aria tools to be installed
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
import argparse
from typing import Dict, List, Optional

# Project Aria imports
try:
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
    from projectaria_tools.core.stream_id import StreamId
    ARIA_TOOLS_AVAILABLE = True
except ImportError:
    print("WARNING: Project Aria tools not installed. Install with:")
    print("pip install projectaria-tools")
    ARIA_TOOLS_AVAILABLE = False


class AriaIMUExtractor:
    """Extract real IMU data from Aria VRS recordings"""
    
    def __init__(self):
        if not ARIA_TOOLS_AVAILABLE:
            raise ImportError("Project Aria tools required but not installed")
    
    def extract_imu_from_vrs(self, vrs_path: str, output_path: str, 
                            start_time: Optional[float] = None,
                            end_time: Optional[float] = None) -> bool:
        """
        Extract IMU data from a VRS file.
        
        Args:
            vrs_path: Path to recording.vrs file
            output_path: Path to save IMU data
            start_time: Optional start time in seconds
            end_time: Optional end time in seconds
            
        Returns:
            bool: Success status
        """
        try:
            # Create data provider
            provider = data_provider.create_vrs_data_provider(vrs_path)
            if not provider:
                print(f"Failed to open VRS file: {vrs_path}")
                return False
            
            # Get IMU stream IDs
            # Aria has two IMUs: left and right
            imu_left_id = StreamId("1201-1")  # Left IMU
            imu_right_id = StreamId("1201-2")  # Right IMU
            
            # Check which streams are available
            available_streams = provider.get_available_stream_ids()
            print(f"Available streams: {[str(s) for s in available_streams]}")
            
            # Extract IMU data
            imu_data_list = []
            timestamps = []
            
            # Get stream metadata
            if imu_right_id in available_streams:
                imu_stream = imu_right_id
                print("Using right IMU stream")
            elif imu_left_id in available_streams:
                imu_stream = imu_left_id
                print("Using left IMU stream")
            else:
                print("No IMU streams found in VRS file")
                return False
            
            # Get IMU configuration
            imu_config = provider.get_imu_configuration(imu_stream)
            if imu_config:
                print(f"IMU sample rate: {imu_config.nominal_rate_hz} Hz")
            
            # Get time range
            if start_time is None:
                start_time = provider.get_first_time_ns(imu_stream, TimeDomain.DEVICE_TIME) / 1e9
            if end_time is None:
                end_time = provider.get_last_time_ns(imu_stream, TimeDomain.DEVICE_TIME) / 1e9
            
            print(f"Extracting IMU data from {start_time:.2f}s to {end_time:.2f}s")
            
            # Iterate through IMU samples
            time_ns = int(start_time * 1e9)
            end_time_ns = int(end_time * 1e9)
            
            while time_ns <= end_time_ns:
                # Get IMU data at this timestamp
                imu_data = provider.get_imu_data_by_time_ns(
                    imu_stream, 
                    time_ns,
                    TimeDomain.DEVICE_TIME,
                    TimeQueryOptions.CLOSEST
                )
                
                if imu_data:
                    # Extract accelerometer and gyroscope data
                    # Aria IMU format: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
                    sample = [
                        imu_data.accel_msec2[0],  # X acceleration (m/s²)
                        imu_data.accel_msec2[1],  # Y acceleration (m/s²)
                        imu_data.accel_msec2[2],  # Z acceleration (m/s²)
                        imu_data.gyro_radsec[0],  # X angular velocity (rad/s)
                        imu_data.gyro_radsec[1],  # Y angular velocity (rad/s)
                        imu_data.gyro_radsec[2],  # Z angular velocity (rad/s)
                    ]
                    
                    imu_data_list.append(sample)
                    timestamps.append(imu_data.capture_timestamp_ns / 1e9)
                
                # Move to next sample (1ms = 1000Hz)
                time_ns += int(1e6)  # 1ms in nanoseconds
            
            # Convert to numpy array
            imu_array = np.array(imu_data_list, dtype=np.float64)
            timestamps = np.array(timestamps, dtype=np.float64)
            
            print(f"Extracted {len(imu_array)} IMU samples")
            print(f"IMU shape: {imu_array.shape}")
            print(f"Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
            
            # Save as PyTorch tensor (compatible with existing pipeline)
            torch.save(torch.from_numpy(imu_array), output_path)
            
            # Also save timestamps
            timestamp_path = output_path.replace('.pt', '_timestamps.npy')
            np.save(timestamp_path, timestamps)
            
            print(f"Saved IMU data to: {output_path}")
            print(f"Saved timestamps to: {timestamp_path}")
            
            return True
            
        except Exception as e:
            print(f"Error extracting IMU: {e}")
            return False
    
    def process_sequence(self, sequence_path: Path, output_dir: Path) -> bool:
        """
        Process a single AriaEveryday sequence to extract real IMU data.
        
        Args:
            sequence_path: Path to sequence directory
            output_dir: Output directory
            
        Returns:
            bool: Success status
        """
        # Look for recording.vrs
        vrs_path = sequence_path / "recording.vrs"
        if not vrs_path.exists():
            print(f"No recording.vrs found in {sequence_path}")
            return False
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract IMU data
        imu_output = output_dir / "imu_data_real.pt"
        success = self.extract_imu_from_vrs(str(vrs_path), str(imu_output))
        
        if success:
            # Load and reshape for compatibility with existing pipeline
            imu_data = torch.load(imu_output)
            
            # The existing pipeline expects [num_frames, samples_per_frame, 6]
            # We need to reshape the continuous IMU stream
            # Assuming 20Hz video and 1000Hz IMU = 50 samples per frame
            samples_per_frame = 50
            num_frames = len(imu_data) // samples_per_frame
            
            # Reshape to match expected format
            imu_reshaped = imu_data[:num_frames * samples_per_frame].reshape(
                num_frames, samples_per_frame, 6
            )
            
            # Save reshaped data
            torch.save(imu_reshaped, output_dir / "imu_data.pt")
            print(f"Reshaped IMU data: {imu_reshaped.shape}")
            
            # Compare with synthetic IMU if it exists
            synthetic_path = output_dir / "imu_data_synthetic.pt"
            if synthetic_path.exists():
                synthetic_imu = torch.load(synthetic_path)
                print("\nComparison with synthetic IMU:")
                print(f"Real IMU shape: {imu_reshaped.shape}")
                print(f"Synthetic IMU shape: {synthetic_imu.shape}")
                print(f"Real IMU stats: mean={imu_reshaped.mean():.3f}, std={imu_reshaped.std():.3f}")
                print(f"Synthetic IMU stats: mean={synthetic_imu.mean():.3f}, std={synthetic_imu.std():.3f}")
        
        return success


def main():
    parser = argparse.ArgumentParser(description='Extract real IMU data from Aria VRS files')
    parser.add_argument('--sequence-dir', type=str, required=True,
                       help='Path to AriaEveryday sequence directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--vrs-file', type=str, default='recording.vrs',
                       help='Name of VRS file (default: recording.vrs)')
    
    args = parser.parse_args()
    
    if not ARIA_TOOLS_AVAILABLE:
        print("ERROR: Project Aria tools not installed")
        print("Install with: pip install projectaria-tools")
        return
    
    extractor = AriaIMUExtractor()
    
    sequence_path = Path(args.sequence_dir)
    output_path = Path(args.output_dir)
    
    success = extractor.process_sequence(sequence_path, output_path)
    
    if success:
        print("\n✅ Successfully extracted real IMU data")
    else:
        print("\n❌ Failed to extract IMU data")


if __name__ == "__main__":
    main()