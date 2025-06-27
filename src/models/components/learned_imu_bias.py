"""
Learned IMU Bias Correction Module.
Based on Deep IMU-Bias paper - reduces bias drift by 20-30%.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class LearnedIMUBias(nn.Module):
    """
    GRU-based IMU bias correction module.
    
    Architecture from Deep IMU-Bias paper:
    - Input: Recent IMU measurements (6D) + current bias estimate (6D)
    - GRU: 12 → 16 hidden units
    - Output: Bias correction Δb (6D)
    
    Called at 20-50Hz to adaptively correct bias drift.
    """
    
    def __init__(self, 
                 hidden_dim: int = 16,
                 window_size: int = 10,  # Number of recent IMU samples
                 device: str = 'cuda'):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.device = device
        
        # Input: window_size * 6 (IMU) + 6 (current bias) = 66 for window=10
        input_dim = window_size * 6 + 6
        
        # Single-layer GRU as per paper
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Output projection to bias correction
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),
            nn.Tanh()  # Limit corrections to reasonable range
        )
        
        # Scale factor for bias corrections (learned)
        self.bias_scale = nn.Parameter(torch.tensor([0.01, 0.01, 0.01,  # Accel bias
                                                     0.001, 0.001, 0.001]))  # Gyro bias
        
        # Hidden state
        self.hidden = None
        
        # IMU buffer for window
        self.imu_buffer = []
        
        self.to(device)
        
    def reset_hidden(self, batch_size: int = 1):
        """Reset GRU hidden state."""
        self.hidden = torch.zeros(1, batch_size, self.hidden_dim).to(self.device)
        self.imu_buffer = []
        
    def add_imu_measurement(self, acc: np.ndarray, gyro: np.ndarray):
        """Add IMU measurement to buffer."""
        imu = np.concatenate([acc, gyro])
        self.imu_buffer.append(imu)
        
        # Keep only recent measurements
        if len(self.imu_buffer) > self.window_size:
            self.imu_buffer.pop(0)
            
    def predict_bias_correction(self, 
                               current_bias: np.ndarray,
                               return_numpy: bool = True) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
        """
        Predict bias correction given current bias estimate.
        
        Args:
            current_bias: Current bias estimate [acc_bias, gyro_bias] (6D)
            return_numpy: Return as numpy array (for integration with GTSAM)
            
        Returns:
            bias_correction: Δb to add to current bias (6D)
            uncertainty: Optional uncertainty estimate
        """
        if len(self.imu_buffer) < self.window_size:
            # Not enough data yet
            if return_numpy:
                return np.zeros(6), None
            else:
                return torch.zeros(6).to(self.device), None
                
        # Prepare input
        imu_window = np.array(self.imu_buffer[-self.window_size:])  # [window, 6]
        imu_flat = imu_window.flatten()  # [window * 6]
        
        # Concatenate with current bias
        input_vec = np.concatenate([imu_flat, current_bias])  # [window*6 + 6]
        
        # Convert to torch
        x = torch.tensor(input_vec, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, input_dim]
        
        # GRU forward
        if self.hidden is None:
            self.reset_hidden()
            
        gru_out, self.hidden = self.gru(x, self.hidden)
        
        # Predict correction
        correction = self.output_net(gru_out.squeeze(0))  # [1, 6]
        
        # Scale correction
        scaled_correction = correction * self.bias_scale
        
        if return_numpy:
            return scaled_correction.squeeze(0).detach().cpu().numpy(), None
        else:
            return scaled_correction.squeeze(0), None
            
    def forward(self, imu_window: torch.Tensor, current_bias: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            imu_window: [B, T, window_size, 6] IMU measurements
            current_bias: [B, T, 6] current bias estimates
            
        Returns:
            bias_corrections: [B, T, 6] predicted corrections
        """
        B, T, W, _ = imu_window.shape
        
        # Flatten IMU window
        imu_flat = imu_window.view(B, T, W * 6)  # [B, T, window*6]
        
        # Concatenate with bias
        x = torch.cat([imu_flat, current_bias], dim=-1)  # [B, T, window*6 + 6]
        
        # GRU forward
        gru_out, _ = self.gru(x)  # [B, T, hidden]
        
        # Predict corrections
        corrections = self.output_net(gru_out)  # [B, T, 6]
        
        # Scale corrections
        scaled_corrections = corrections * self.bias_scale
        
        return scaled_corrections


class BiasedIMUPreintegration:
    """
    Wrapper around IMU preintegration that includes learned bias correction.
    Injects bias corrections into the preintegration process.
    """
    
    def __init__(self, 
                 base_preintegrator,
                 bias_corrector: LearnedIMUBias,
                 update_rate: float = 50.0):  # Hz
        """
        Args:
            base_preintegrator: Base IMU preintegration module
            bias_corrector: Learned bias correction network
            update_rate: How often to update bias (Hz)
        """
        self.base = base_preintegrator
        self.bias_corrector = bias_corrector
        self.update_period = 1.0 / update_rate
        
        # Timing
        self.last_bias_update = 0.0
        self.current_time = 0.0
        
        # Current bias estimate
        self.current_bias = np.zeros(6)
        
    def integrate_measurement(self, acc: np.ndarray, gyro: np.ndarray, dt: float):
        """Integrate measurement with learned bias correction."""
        # Add to bias corrector buffer
        self.bias_corrector.add_imu_measurement(acc, gyro)
        
        # Update time
        self.current_time += dt
        
        # Check if we should update bias
        if self.current_time - self.last_bias_update >= self.update_period:
            # Get bias correction
            with torch.no_grad():
                delta_bias, _ = self.bias_corrector.predict_bias_correction(
                    self.current_bias, return_numpy=True
                )
                
            # Update bias estimate
            self.current_bias += delta_bias
            self.last_bias_update = self.current_time
            
            # Clamp bias to reasonable values
            self.current_bias[:3] = np.clip(self.current_bias[:3], -0.5, 0.5)  # Accel
            self.current_bias[3:] = np.clip(self.current_bias[3:], -0.1, 0.1)  # Gyro
            
        # Apply bias correction
        acc_corrected = acc - self.current_bias[:3]
        gyro_corrected = gyro - self.current_bias[3:]
        
        # Integrate with corrected measurements
        self.base.integrate_measurement(acc_corrected, gyro_corrected, dt)
        
    def predict(self, R0: np.ndarray, p0: np.ndarray, v0: np.ndarray):
        """Predict using base preintegrator."""
        return self.base.predict(R0, p0, v0)
        
    def get_current_bias(self) -> np.ndarray:
        """Get current bias estimate."""
        return self.current_bias.copy()