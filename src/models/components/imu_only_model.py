"""IMU-only model for pose prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vsvio import Inertial_encoder


class IMUOnlyModel(nn.Module):
    """IMU-only model that uses only IMU data to predict relative poses."""
    
    def __init__(self, config=None):
        super().__init__()
        
        # Default configuration
        if config is None:
            class Config:
                # IMU parameters
                i_f_len = 256  # IMU feature dimension
                imu_dropout = 0.2
                
                # Model parameters
                hidden_dim = 512
                num_layers = 3
                dropout = 0.1
                
                # Output
                output_dim = 7  # 3 trans + 4 quat
                
            config = Config()
        
        self.config = config
        
        # IMU encoder from VIFT
        self.imu_encoder = Inertial_encoder(config)
        
        # Additional processing layers for temporal modeling
        self.temporal_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.i_f_len if i == 0 else config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            )
            for i in range(config.num_layers)
        ])
        
        # Output projection for multi-step prediction
        self.pose_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )
        
        # Initialize quaternion bias to favor identity rotation
        with torch.no_grad():
            self.pose_predictor[-1].bias[3:6].fill_(0.0)  # qx, qy, qz = 0
            self.pose_predictor[-1].bias[6].fill_(1.0)    # qw = 1
            self.pose_predictor[-1].weight.data *= 0.01   # Small weights
    
    def forward(self, batch):
        """Forward pass using only IMU data.
        
        Args:
            batch: Dictionary containing:
                - 'imu': IMU data [B, 110, 6] (11 frames × 10 Hz)
        
        Returns:
            Dictionary with:
                - 'poses': Predicted poses [B, 10, 7] (10 transitions)
        """
        imu = batch['imu']  # [B, 110, 6]
        batch_size = imu.shape[0]
        
        # The encoder expects [B, seq_len, 11, 6] where seq_len is number of windows
        # For each of the 11 frames, we need 11 IMU samples (not 10)
        # But we have 110 samples total for 10 transitions
        # So we'll create overlapping windows of 11 samples each
        
        imu_windows = []
        for i in range(11):
            if i < 10:
                # For frames 0-9, take 11 samples starting from i*10
                start_idx = i * 10
                end_idx = min(start_idx + 11, 110)
                window = imu[:, start_idx:end_idx, :]
                # Pad if necessary
                if window.shape[1] < 11:
                    pad_size = 11 - window.shape[1]
                    window = torch.cat([window, window[:, -1:, :].repeat(1, pad_size, 1)], dim=1)
            else:
                # For the last frame, use the last 11 samples
                window = imu[:, -11:, :]
            imu_windows.append(window)
        
        imu_windows = torch.stack(imu_windows, dim=1)  # [B, 11, 11, 6]
        
        # Get IMU features using the encoder
        imu_features = self.imu_encoder(imu_windows)  # [B, 11, 256]
        
        # Process through temporal layers
        features = imu_features
        for layer in self.temporal_layers:
            features = layer(features)  # [B, 11, hidden_dim]
        
        # Predict poses for transitions (first 10 timesteps predict next frame)
        transition_features = features[:, :10, :]  # [B, 10, hidden_dim]
        
        # Apply pose predictor to each timestep
        batch_size, seq_len, feat_dim = transition_features.shape
        transition_features_flat = transition_features.reshape(-1, feat_dim)  # [B*10, hidden_dim]
        poses_flat = self.pose_predictor(transition_features_flat)  # [B*10, 7]
        poses = poses_flat.reshape(batch_size, seq_len, 7)  # [B, 10, 7]
        
        # Normalize quaternion part
        trans = poses[:, :, :3]  # [B, 10, 3]
        quat = poses[:, :, 3:]   # [B, 10, 4]
        quat = F.normalize(quat, p=2, dim=-1)
        
        # Combine normalized poses
        normalized_poses = torch.cat([trans, quat], dim=-1)  # [B, 10, 7]
        
        return {
            'poses': normalized_poses  # [B, 10, 7]
        }
    
    def get_feature_dim(self):
        """Return the feature dimension for compatibility."""
        return self.config.hidden_dim