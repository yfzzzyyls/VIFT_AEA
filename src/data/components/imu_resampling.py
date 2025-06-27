#!/usr/bin/env python3
"""
IMU resampling to exact 1kHz grid.
Eliminates timing jitter that causes integration drift.
"""

import numpy as np
import torch
from scipy import interpolate
from typing import List, Tuple, Optional


def resample_imu_to_fixed_rate(
    imu_data: List[np.ndarray],
    timestamps: List[float],
    target_rate: float = 1000.0,
    method: str = 'cubic'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample IMU data to fixed rate grid.
    
    Args:
        imu_data: List of IMU samples [N x 6]
        timestamps: List of timestamps in seconds [N]
        target_rate: Target sampling rate in Hz (default 1000)
        method: Interpolation method ('linear', 'cubic', 'quintic')
        
    Returns:
        resampled_data: Resampled IMU data
        resampled_timestamps: Regular timestamp grid
    """
    # Convert to numpy arrays
    data = np.array(imu_data)
    ts = np.array(timestamps)
    
    # Check if already close to target rate
    actual_rate = 1.0 / np.mean(np.diff(ts))
    if abs(actual_rate - target_rate) < 1.0:  # Within 1Hz
        return data, ts
        
    # Create regular timestamp grid
    dt = 1.0 / target_rate
    t_start = ts[0]
    t_end = ts[-1]
    t_regular = np.arange(t_start, t_end, dt)
    
    # Interpolate each channel
    resampled = np.zeros((len(t_regular), 6))
    
    for i in range(6):
        if method == 'linear':
            f = interpolate.interp1d(ts, data[:, i], kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
        elif method == 'cubic':
            f = interpolate.interp1d(ts, data[:, i], kind='cubic',
                                   bounds_error=False, fill_value='extrapolate')
        elif method == 'quintic':
            # Use UnivariateSpline for smoother interpolation
            f = interpolate.UnivariateSpline(ts, data[:, i], k=5, s=0)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
            
        resampled[:, i] = f(t_regular)
        
    return resampled, t_regular


def resample_variable_length_imu(
    imu_windows: List[torch.Tensor],
    dt_nominal: float = 0.001,
    jitter_std: float = 0.0001
) -> List[torch.Tensor]:
    """
    Resample variable-length IMU windows to reduce timing jitter.
    
    Args:
        imu_windows: List of variable-length IMU tensors
        dt_nominal: Nominal time step (1ms for 1kHz)
        jitter_std: Expected jitter standard deviation
        
    Returns:
        List of resampled IMU windows
    """
    resampled_windows = []
    
    for window in imu_windows:
        n_samples = len(window)
        
        # Create timestamps with simulated jitter
        t_nominal = np.arange(n_samples) * dt_nominal
        
        # If we're training, add synthetic jitter to learn robustness
        if window.requires_grad:
            jitter = np.random.normal(0, jitter_std, n_samples)
            jitter[0] = 0  # First sample has no jitter
            t_jittered = t_nominal + np.cumsum(jitter)
        else:
            t_jittered = t_nominal
            
        # Resample to regular grid
        window_np = window.detach().cpu().numpy()
        resampled_np, _ = resample_imu_to_fixed_rate(
            window_np, t_jittered, target_rate=1000.0, method='cubic'
        )
        
        # Convert back to tensor
        resampled = torch.tensor(resampled_np, dtype=window.dtype, device=window.device)
        resampled_windows.append(resampled)
        
    return resampled_windows


class IMUResampler(torch.nn.Module):
    """
    PyTorch module for IMU resampling in data pipeline.
    """
    
    def __init__(self, 
                 target_rate: float = 1000.0,
                 add_synthetic_jitter: bool = True,
                 jitter_std_ms: float = 0.1):
        """
        Initialize IMU resampler.
        
        Args:
            target_rate: Target sampling rate in Hz
            add_synthetic_jitter: Add jitter during training
            jitter_std_ms: Jitter standard deviation in milliseconds
        """
        super().__init__()
        self.target_rate = target_rate
        self.add_synthetic_jitter = add_synthetic_jitter
        self.jitter_std = jitter_std_ms / 1000.0  # Convert to seconds
        
    def forward(self, imu_data: torch.Tensor) -> torch.Tensor:
        """
        Resample IMU data.
        
        Args:
            imu_data: [B, T, K, 6] or [B, T, 6] IMU data
            
        Returns:
            Resampled IMU data (same shape)
        """
        if not self.training or not self.add_synthetic_jitter:
            return imu_data  # No resampling needed in eval mode
            
        original_shape = imu_data.shape
        device = imu_data.device
        
        # Handle variable-length format
        if len(original_shape) == 4:  # [B, T, K, 6]
            B, T, K = original_shape[:3]
            
            # Process each transition window
            resampled_list = []
            for b in range(B):
                transition_list = []
                for t in range(T):
                    window = imu_data[b, t]  # [K, 6]
                    
                    # Skip if too few samples
                    if K < 10:
                        transition_list.append(window)
                        continue
                        
                    # Resample this window
                    resampled = resample_variable_length_imu(
                        [window], 
                        dt_nominal=0.001,
                        jitter_std=self.jitter_std
                    )[0]
                    
                    # Ensure same length
                    if len(resampled) != K:
                        # Interpolate to match original length
                        indices = torch.linspace(0, len(resampled)-1, K)
                        resampled_matched = torch.zeros(K, 6, device=device)
                        for i in range(6):
                            resampled_matched[:, i] = torch.nn.functional.interpolate(
                                resampled[:, i].unsqueeze(0).unsqueeze(0),
                                size=K,
                                mode='linear',
                                align_corners=True
                            ).squeeze()
                    else:
                        resampled_matched = resampled
                        
                    transition_list.append(resampled_matched)
                    
                resampled_list.append(torch.stack(transition_list))
                
            return torch.stack(resampled_list)
            
        else:  # [B, T, 6]
            # For fixed-length format, process as single sequence
            B, T = original_shape[:2]
            
            resampled_list = []
            for b in range(B):
                sequence = imu_data[b]  # [T, 6]
                resampled = resample_variable_length_imu(
                    [sequence],
                    dt_nominal=0.001,
                    jitter_std=self.jitter_std
                )[0]
                
                # Ensure same length
                if len(resampled) != T:
                    resampled = torch.nn.functional.interpolate(
                        resampled.T.unsqueeze(0),  # [1, 6, T]
                        size=T,
                        mode='linear',
                        align_corners=True
                    ).squeeze(0).T  # [T, 6]
                    
                resampled_list.append(resampled)
                
            return torch.stack(resampled_list)


def compute_timing_error(timestamps: np.ndarray, 
                        nominal_rate: float = 1000.0) -> dict:
    """
    Compute timing error statistics.
    
    Args:
        timestamps: Array of timestamps
        nominal_rate: Expected sampling rate
        
    Returns:
        Dictionary of timing statistics
    """
    dt_actual = np.diff(timestamps)
    dt_nominal = 1.0 / nominal_rate
    
    timing_error = dt_actual - dt_nominal
    jitter = timing_error - np.mean(timing_error)
    
    stats = {
        'mean_rate_hz': 1.0 / np.mean(dt_actual),
        'rate_std_hz': np.std(1.0 / dt_actual),
        'timing_error_mean_us': np.mean(timing_error) * 1e6,
        'timing_error_std_us': np.std(timing_error) * 1e6,
        'jitter_rms_us': np.sqrt(np.mean(jitter**2)) * 1e6,
        'max_jitter_us': np.max(np.abs(jitter)) * 1e6
    }
    
    return stats