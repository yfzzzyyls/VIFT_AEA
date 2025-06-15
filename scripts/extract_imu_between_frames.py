#!/usr/bin/env python3
"""
Extract IMU data properly aligned between frames for VIO training.
This ensures causal relationships and avoids temporal misalignment.
"""

import numpy as np
import torch
from typing import List, Dict, Optional
from pathlib import Path


def extract_imu_between_frames(
    imu_timestamps: np.ndarray,
    imu_data: np.ndarray,
    frame_timestamps: List[float],
    target_samples_per_interval: int = 50
) -> torch.Tensor:
    """
    Extract IMU data between consecutive frames with proper temporal alignment.
    
    Args:
        imu_timestamps: Array of IMU timestamps in seconds [N_imu]
        imu_data: Array of IMU measurements [N_imu, 6]
        frame_timestamps: List of frame timestamps in seconds
        target_samples_per_interval: Target number of IMU samples between frames
        
    Returns:
        torch.Tensor: IMU data [num_frames-1, samples_per_interval, 6]
    """
    num_frames = len(frame_timestamps)
    imu_between_frames = []
    
    for i in range(num_frames - 1):
        # Define interval between consecutive frames
        start_time = frame_timestamps[i]
        end_time = frame_timestamps[i + 1]
        
        # Find IMU samples within this interval
        mask = (imu_timestamps >= start_time) & (imu_timestamps < end_time)
        interval_imu = imu_data[mask]
        interval_timestamps = imu_timestamps[mask]
        
        if len(interval_imu) == 0:
            # No IMU data in interval - this shouldn't happen with 1000Hz IMU
            print(f"Warning: No IMU data between frames {i} and {i+1}")
            interval_imu = np.zeros((target_samples_per_interval, 6))
        elif len(interval_imu) < target_samples_per_interval:
            # Interpolate to get exact number of samples
            interval_imu = interpolate_imu(
                interval_imu, 
                interval_timestamps,
                start_time,
                end_time,
                target_samples_per_interval
            )
        elif len(interval_imu) > target_samples_per_interval:
            # Downsample to target number
            indices = np.linspace(0, len(interval_imu)-1, target_samples_per_interval, dtype=int)
            interval_imu = interval_imu[indices]
        
        imu_between_frames.append(torch.from_numpy(interval_imu).float())
    
    return torch.stack(imu_between_frames)


def interpolate_imu(
    imu_data: np.ndarray,
    timestamps: np.ndarray,
    start_time: float,
    end_time: float,
    num_samples: int
) -> np.ndarray:
    """
    Interpolate IMU data to get exact number of samples.
    """
    # Create target timestamps
    target_times = np.linspace(start_time, end_time, num_samples)
    
    # Interpolate each IMU channel
    interpolated = np.zeros((num_samples, 6))
    for i in range(6):
        interpolated[:, i] = np.interp(target_times, timestamps, imu_data[:, i])
    
    return interpolated


def convert_aria_imu_format(aria_imu_data: torch.Tensor, frame_timestamps: List[float]) -> torch.Tensor:
    """
    Convert Aria's per-frame IMU format to proper between-frames format.
    
    Args:
        aria_imu_data: Current Aria format [num_frames, 50, 6]
        frame_timestamps: Frame timestamps
        
    Returns:
        torch.Tensor: Between-frames format [num_frames-1, 50, 6]
    """
    num_frames = aria_imu_data.shape[0]
    samples_per_frame = aria_imu_data.shape[1]
    
    # For between-frames format, we need num_frames-1 intervals
    between_frames_imu = []
    
    for i in range(num_frames - 1):
        # Take the second half of frame i's IMU and first half of frame i+1's IMU
        # This approximates IMU data between the frames
        first_half = aria_imu_data[i, samples_per_frame//2:, :]  # 25 samples
        second_half = aria_imu_data[i+1, :samples_per_frame//2, :]  # 25 samples
        
        # Concatenate to form 50 samples between frames
        between_frames = torch.cat([first_half, second_half], dim=0)
        between_frames_imu.append(between_frames)
    
    return torch.stack(between_frames_imu)


def process_kitti_style_imu(imu_data: torch.Tensor, seq_len: int = 11) -> torch.Tensor:
    """
    Process IMU data for KITTI-style model input.
    
    Args:
        imu_data: Between-frames IMU data [num_intervals, 50, 6]
        seq_len: Sequence length for model
        
    Returns:
        torch.Tensor: KITTI format [(seq_len-1)*10+1, 6]
    """
    # KITTI expects 10 samples per interval
    # Downsample from 50 to 10 samples per interval
    downsampled = []
    
    for i in range(seq_len - 1):
        interval_data = imu_data[i]  # [50, 6]
        # Take every 5th sample to get 10 samples
        indices = np.arange(0, 50, 5)
        downsampled.append(interval_data[indices])
    
    # Add one final sample
    downsampled.append(imu_data[seq_len-2, -1:, :])  # Last sample
    
    return torch.cat(downsampled, dim=0)  # [(seq_len-1)*10+1, 6]


if __name__ == "__main__":
    # Example usage
    print("Example of proper IMU extraction between frames:")
    
    # Simulate 1000Hz IMU data
    imu_rate = 1000  # Hz
    duration = 1.0   # 1 second
    num_imu_samples = int(imu_rate * duration)
    imu_timestamps = np.linspace(0, duration, num_imu_samples)
    imu_data = np.random.randn(num_imu_samples, 6)
    
    # Simulate 20Hz camera frames
    frame_rate = 20  # Hz
    num_frames = int(frame_rate * duration) + 1
    frame_timestamps = np.linspace(0, duration, num_frames)
    
    # Extract IMU between frames
    imu_between = extract_imu_between_frames(
        imu_timestamps,
        imu_data,
        frame_timestamps.tolist(),
        target_samples_per_interval=50
    )
    
    print(f"IMU timestamps shape: {imu_timestamps.shape}")
    print(f"IMU data shape: {imu_data.shape}")
    print(f"Frame timestamps: {len(frame_timestamps)}")
    print(f"IMU between frames shape: {imu_between.shape}")
    print(f"Expected: [{num_frames-1}, 50, 6]")