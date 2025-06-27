"""
Adaptive Process and Measurement Noise Estimation.
Learns state-dependent Q and R matrices for improved EKF consistency.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict


class AdaptiveNoiseModel(nn.Module):
    """
    MLP-based adaptive noise estimation for VIO.
    
    Maps recent IMU statistics and visual quality metrics to:
    - Process noise Q scaling factors
    - Measurement noise R scaling factors
    
    Based on uncertainty-aware VIO papers showing 15% RMSE reduction.
    """
    
    def __init__(self,
                 imu_window: int = 20,  # Number of recent IMU samples
                 hidden_dim: int = 64,
                 device: str = 'cuda'):
        super().__init__()
        
        self.imu_window = imu_window
        self.device = device
        
        # Input features:
        # - IMU variance (6D: acc + gyro)
        # - IMU magnitude stats (6D: mean + std for acc/gyro)
        # - Motion intensity (1D: norm of angular velocity)
        # - Visual feature count (1D)
        # - Mean correlation confidence (1D)
        input_dim = 6 + 6 + 1 + 1 + 1  # = 15
        
        # Process noise network (for Q)
        self.q_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 15),  # 15D state vector
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Measurement noise network (for R)
        self.r_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 3D for visual measurements
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Scale bounds (clamped to [0.5, 2.0]× nominal as per your spec)
        self.scale_min = 0.5
        self.scale_max = 2.0
        
        # Nominal noise values (to be scaled)
        self.register_buffer('nominal_q', torch.ones(15) * 1e-4)
        self.register_buffer('nominal_r', torch.ones(3) * 1e-2)
        
        # IMU buffer
        self.imu_buffer = []
        
        self.to(device)
        
    def add_imu_measurement(self, acc: np.ndarray, gyro: np.ndarray):
        """Add IMU measurement to buffer for statistics."""
        imu = np.concatenate([acc, gyro])
        self.imu_buffer.append(imu)
        
        if len(self.imu_buffer) > self.imu_window:
            self.imu_buffer.pop(0)
            
    def compute_imu_statistics(self) -> Dict[str, np.ndarray]:
        """Compute IMU statistics from buffer."""
        if len(self.imu_buffer) < 2:
            return {
                'variance': np.zeros(6),
                'mean_magnitude': np.zeros(6),
                'std_magnitude': np.zeros(6),
                'angular_intensity': 0.0
            }
            
        imu_array = np.array(self.imu_buffer)  # [N, 6]
        
        # Variance of each channel
        variance = np.var(imu_array, axis=0)
        
        # Magnitude statistics
        acc_mag = np.linalg.norm(imu_array[:, :3], axis=1)
        gyro_mag = np.linalg.norm(imu_array[:, 3:], axis=1)
        
        mean_magnitude = np.array([
            np.mean(acc_mag), np.mean(acc_mag), np.mean(acc_mag),
            np.mean(gyro_mag), np.mean(gyro_mag), np.mean(gyro_mag)
        ])
        
        std_magnitude = np.array([
            np.std(acc_mag), np.std(acc_mag), np.std(acc_mag),
            np.std(gyro_mag), np.std(gyro_mag), np.std(gyro_mag)
        ])
        
        # Angular motion intensity
        angular_intensity = np.mean(gyro_mag)
        
        return {
            'variance': variance,
            'mean_magnitude': mean_magnitude,
            'std_magnitude': std_magnitude,
            'angular_intensity': angular_intensity
        }
        
    def predict_noise_scales(self,
                           feature_count: int = 0,
                           mean_correlation: float = 0.0,
                           return_numpy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict Q and R scaling factors.
        
        Args:
            feature_count: Number of tracked visual features
            mean_correlation: Mean correlation confidence [0, 1]
            return_numpy: Return as numpy arrays
            
        Returns:
            q_scales: Process noise scaling factors (15D)
            r_scales: Measurement noise scaling factors (3D)
        """
        # Compute IMU statistics
        stats = self.compute_imu_statistics()
        
        # Build input vector
        input_vec = np.concatenate([
            stats['variance'],           # 6D
            stats['mean_magnitude'],     # 6D
            [stats['angular_intensity']], # 1D
            [feature_count / 100.0],     # 1D (normalized)
            [mean_correlation]           # 1D
        ])
        
        # Convert to torch
        x = torch.tensor(input_vec, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(0)  # [1, 15]
        
        # Predict scales
        with torch.no_grad():
            q_raw = self.q_net(x).squeeze(0)  # [15]
            r_raw = self.r_net(x).squeeze(0)  # [3]
            
        # Map to [scale_min, scale_max]
        q_scales = self.scale_min + (self.scale_max - self.scale_min) * q_raw
        r_scales = self.scale_min + (self.scale_max - self.scale_min) * r_raw
        
        if return_numpy:
            return q_scales.cpu().numpy(), r_scales.cpu().numpy()
        else:
            return q_scales, r_scales
            
    def get_scaled_noise_matrices(self,
                                 feature_count: int = 0,
                                 mean_correlation: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get scaled Q and R matrices.
        
        Returns:
            Q: Scaled process noise covariance (15×15)
            R: Scaled measurement noise covariance (3×3)
        """
        q_scales, r_scales = self.predict_noise_scales(feature_count, mean_correlation)
        
        # Build diagonal matrices
        Q = np.diag(self.nominal_q.cpu().numpy() * q_scales)
        R = np.diag(self.nominal_r.cpu().numpy() * r_scales)
        
        return Q, R
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            features: [B, T, 15] input features
            
        Returns:
            q_scales: [B, T, 15] process noise scales
            r_scales: [B, T, 3] measurement noise scales
        """
        B, T, _ = features.shape
        
        # Flatten for processing
        features_flat = features.view(B * T, -1)  # [B*T, 15]
        
        # Predict scales
        q_raw = self.q_net(features_flat)  # [B*T, 15]
        r_raw = self.r_net(features_flat)  # [B*T, 3]
        
        # Map to [scale_min, scale_max]
        q_scales = self.scale_min + (self.scale_max - self.scale_min) * q_raw
        r_scales = self.scale_min + (self.scale_max - self.scale_min) * r_raw
        
        # Reshape
        q_scales = q_scales.view(B, T, 15)
        r_scales = r_scales.view(B, T, 3)
        
        return q_scales, r_scales


class AdaptiveMSCKFState:
    """
    MSCKF state with adaptive noise models.
    Wraps existing MSCKF state with learned Q/R estimation.
    """
    
    def __init__(self,
                 base_msckf_state,
                 noise_model: AdaptiveNoiseModel):
        """
        Args:
            base_msckf_state: Base MSCKF state manager
            noise_model: Adaptive noise estimation network
        """
        self.base = base_msckf_state
        self.noise_model = noise_model
        
        # Track visual quality metrics
        self.last_feature_count = 0
        self.last_correlation_confidence = 0.0
        
    def propagate_imu(self, R_new, p_new, v_new, Phi, timestamp):
        """Propagate with adaptive process noise."""
        # Get adaptive Q
        Q, _ = self.noise_model.get_scaled_noise_matrices(
            self.last_feature_count,
            self.last_correlation_confidence
        )
        
        # Use adaptive Q for propagation
        self.base.propagate_imu(R_new, p_new, v_new, Phi, Q, timestamp)
        
    def update_visual_metrics(self, feature_count: int, mean_correlation: float):
        """Update visual quality metrics for noise adaptation."""
        self.last_feature_count = feature_count
        self.last_correlation_confidence = mean_correlation
        
    def get_measurement_noise(self) -> np.ndarray:
        """Get adaptive measurement noise."""
        _, R = self.noise_model.get_scaled_noise_matrices(
            self.last_feature_count,
            self.last_correlation_confidence
        )
        return R
    
    def __getattr__(self, name):
        """Forward all other attributes to base MSCKF state."""
        return getattr(self.base, name)