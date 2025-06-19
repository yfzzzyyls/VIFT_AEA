"""Shared IMU utilities to ensure consistency between training and evaluation."""

import torch


def remove_gravity(imu_window: torch.Tensor, num_transitions: int = 20) -> torch.Tensor:
    """Remove gravity bias from IMU accelerometer data
    
    This function computes a per-transition bias and removes it from accelerometer readings.
    NOTE: The model was trained with raw IMU data (gravity included), so this should 
    generally NOT be used during evaluation to maintain train/test consistency.
    
    Args:
        imu_window: [B,N,6] tensor with (ax,ay,az,gx,gy,gz) where N = num_transitions * samples_per_transition
        num_transitions: Number of transitions in the window (default: 20 for 21-frame sequences)
    
    Returns:
        IMU tensor with gravity-bias removed from accelerometer
    """
    # Extract accelerometer data
    accel = imu_window[..., :3]  # [B, N, 3]
    
    # Dynamically determine samples per transition
    total_samples = accel.shape[1]
    samples_per_transition = total_samples // num_transitions
    
    if total_samples % num_transitions != 0:
        raise ValueError(f"Total samples {total_samples} not divisible by {num_transitions} transitions")
    
    # Reshape to compute per-transition bias
    # Average across all samples within each transition
    # Use contiguous() to ensure safe reshape on potentially non-contiguous tensors
    accel_reshaped = accel.contiguous().view(accel.shape[0], num_transitions, samples_per_transition, 3)
    bias = accel_reshaped.mean(dim=2, keepdim=True)  # [B, num_transitions, 1, 3]
    
    # Expand bias for each sample in each transition (no memory copy)
    bias_expanded = bias.expand(-1, num_transitions, samples_per_transition, -1).contiguous().view(accel.shape)
    
    # Remove bias from accelerometer
    accel_corrected = accel - bias_expanded
    
    # Concatenate back with gyroscope data
    return torch.cat([accel_corrected, imu_window[..., 3:]], dim=-1)