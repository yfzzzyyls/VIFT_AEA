#!/usr/bin/env python3
"""
Pure IMU encoder model for 7-DoF pose prediction.
Standalone implementation without TransformerVIO dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PureIMUEncoder(nn.Module):
    """
    Pure IMU encoder: 1D CNN → 7-DoF pose predictions
    
    Architecture:
    - 3-layer 1D CNN for temporal feature extraction
    - Direct prediction heads for translation and rotation
    - No transformer, no visual features, no dependencies
    
    Input: IMU data [B, seq_len-1, samples_per_interval, 6]
    Output: Relative poses [B, seq_len-1, 7] (3 trans + 4 quat)
    """
    
    def __init__(self, 
                 seq_len: int = 21,
                 samples_per_interval: int = 50,  # 50 samples at 1kHz = 50ms
                 imu_channels: int = 6,
                 feat_dim: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        
        self.seq_len = seq_len
        self.n_steps = seq_len - 1  # 20 transitions for seq_len=21
        self.samples_per_interval = samples_per_interval
        self.feat_dim = feat_dim
        
        # Temporal Convolutional Network with dilated convolutions
        # Inspired by IONet/Extended-IONet for better long-range dependencies
        # Input: [B*n_steps, samples_per_interval, 6]
        # Output: [B*n_steps, feat_dim]
        
        # Initial projection
        self.input_proj = nn.Conv1d(imu_channels, 64, kernel_size=1)
        
        # TCN blocks with residual connections and increasing dilation
        self.tcn_blocks = nn.ModuleList([
            self._make_tcn_block(64, 128, kernel_size=5, dilation=1, dropout=dropout),
            self._make_tcn_block(128, 128, kernel_size=5, dilation=2, dropout=dropout),
            self._make_tcn_block(128, 256, kernel_size=5, dilation=4, dropout=dropout),
            self._make_tcn_block(256, 256, kernel_size=5, dilation=8, dropout=dropout),
        ])
        
        # Final projection to feature dimension
        self.final_conv = nn.Sequential(
            nn.Conv1d(256, feat_dim, kernel_size=1),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Prediction heads
        hidden_dim = feat_dim // 2  # 128
        
        # Shared layer
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Translation head
        self.trans_head = nn.Linear(hidden_dim, 3)
        
        # Rotation head (quaternion)
        self.rot_head = nn.Linear(hidden_dim, 4)
        
        # Initialize weights
        self._initialize_weights()
        
        # Learnable uncertainty weights for loss balancing
        self.s_t = nn.Parameter(torch.zeros(()))  # Translation log-variance
        self.s_r = nn.Parameter(torch.zeros(()))  # Rotation log-variance
    
    def _make_tcn_block(self, in_channels, out_channels, kernel_size=5, dilation=1, dropout=0.2):
        """Create a TCN block with residual connection."""
        
        class TCNBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
                super().__init__()
                padding = (kernel_size - 1) * dilation // 2
                
                self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                      padding=padding, dilation=dilation)
                self.bn1 = nn.BatchNorm1d(out_channels)
                self.dropout1 = nn.Dropout(dropout)
                
                self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                      padding=padding, dilation=dilation)
                self.bn2 = nn.BatchNorm1d(out_channels)
                self.dropout2 = nn.Dropout(dropout)
                
                # Residual connection
                self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
                self.relu = nn.ReLU(inplace=True)
                
            def forward(self, x):
                residual = x
                
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.dropout1(out)
                
                out = self.conv2(out)
                out = self.bn2(out)
                
                if self.downsample is not None:
                    residual = self.downsample(x)
                
                out += residual
                out = self.relu(out)
                out = self.dropout2(out)
                
                return out
        
        return TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        # CNN layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Linear layers
        for m in [self.shared[0], self.trans_head]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        
        # Quaternion head - initialize to favor identity rotation
        nn.init.xavier_uniform_(self.rot_head.weight, gain=0.1)  # Increased from 0.01
        nn.init.constant_(self.rot_head.bias, 0)
        # Initialize one component to be slightly positive to avoid zero quaternions
        with torch.no_grad():
            self.rot_head.bias[3] = 0.1
    
    def extract_imu_features(self, imu_windows):
        """
        Extract features from IMU windows using TCN architecture.
        
        Args:
            imu_windows: [B, n_steps, samples_per_interval, 6]
        
        Returns:
            features: [B, n_steps, feat_dim]
        """
        B, n_steps, samples, channels = imu_windows.shape
        
        # Reshape for CNN processing
        imu_flat = imu_windows.reshape(B * n_steps, samples, channels)
        
        # Transpose for Conv1d (needs [B*n_steps, channels, samples])
        imu_flat = imu_flat.transpose(1, 2)  # [B*n_steps, 6, samples]
        
        # Initial projection
        x = self.input_proj(imu_flat)  # [B*n_steps, 64, samples]
        
        # Apply TCN blocks with residual connections
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)  # Progressive: 64->128->128->256->256
        
        # Final projection
        x = self.final_conv(x)  # [B*n_steps, feat_dim, samples]
        
        # Global pooling
        features = self.global_pool(x)  # [B*n_steps, feat_dim, 1]
        features = features.squeeze(-1)  # [B*n_steps, feat_dim]
        
        # Reshape back to [B, n_steps, feat_dim]
        features = features.reshape(B, n_steps, -1)
        
        return features
    
    def _robust_geodesic_loss(self, pred_quat, gt_quat):
        """Numerically stable quaternion geodesic loss with improved handling."""
        # Normalize quaternions with better epsilon
        pred_quat = F.normalize(pred_quat.view(-1, 4), p=2, dim=-1, eps=1e-6)
        gt_quat = F.normalize(gt_quat.view(-1, 4), p=2, dim=-1, eps=1e-6)
        
        # Compute dot product (handle double cover)
        dot = (pred_quat * gt_quat).sum(-1)
        
        # Choose the closer quaternion (handle double cover)
        # If dot < 0, the opposite quaternion is closer
        mask = dot < 0
        dot = torch.where(mask, -dot, dot)
        
        # Clamp to avoid numerical issues with acos
        dot = dot.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        
        # Two options for computing angle:
        # Option 1: Direct acos (can be unstable near ±1)
        # angle = 2.0 * torch.acos(dot)
        
        # Option 2: atan2 formulation (more stable)
        # For unit quaternions: 1 - dot^2 = sin^2(θ/2)
        angle = 2.0 * torch.atan2(torch.sqrt(1.0 - dot * dot), dot)
        
        return angle.mean()
    
    def forward(self, batch, epoch: int = 0):
        """
        Forward pass.
        
        Args:
            batch: Dictionary with 'imu' and optionally 'gt_poses'
            epoch: Current epoch for curriculum learning
        
        Returns:
            Dictionary with predictions and losses
        """
        imu = batch['imu']  # [B, n_steps * samples_per_interval, 6]
        B = imu.shape[0]
        
        # Note: IMU saturation is now handled in the training script
        # to allow for better control and debugging
        
        # Validate and reshape IMU data
        total_samples = imu.shape[1]
        expected_samples = self.n_steps * self.samples_per_interval
        
        if total_samples == expected_samples:
            # Expected case: exact match
            samples_per_interval = self.samples_per_interval
        elif total_samples % self.n_steps == 0:
            # Flexible case: different sampling rate
            samples_per_interval = total_samples // self.n_steps
            if samples_per_interval not in [10, 11, 40, 50]:  # Common values
                print(f"Note: Using {samples_per_interval} IMU samples per interval")
        else:
            raise ValueError(f"IMU samples {total_samples} not compatible with {self.n_steps} steps")
        
        # Reshape to windows
        imu_windows = imu.reshape(B, self.n_steps, samples_per_interval, 6)
        
        # Extract features
        features = self.extract_imu_features(imu_windows)  # [B, n_steps, feat_dim]
        
        # Apply prediction heads
        B, n_steps, _ = features.shape
        features_flat = features.reshape(-1, self.feat_dim)  # [B*n_steps, feat_dim]
        
        # Shared processing
        shared_feat = self.shared(features_flat)  # [B*n_steps, hidden_dim]
        
        # Predictions
        trans = self.trans_head(shared_feat)  # [B*n_steps, 3]
        quat = self.rot_head(shared_feat)     # [B*n_steps, 4]
        
        # Reshape and normalize with better numerical stability
        trans = trans.reshape(B, n_steps, 3)
        quat = quat.reshape(B, n_steps, 4)
        
        # Add small epsilon before normalization to prevent zero quaternions
        quat = quat + 1e-8
        
        # Check for NaN before normalization
        if torch.isnan(quat).any():
            print(f"⚠️ NaN detected in quaternion predictions before normalization!")
            # Replace NaN with small random values
            quat = torch.where(torch.isnan(quat), torch.randn_like(quat) * 0.01, quat)
        
        quat = F.normalize(quat, p=2, dim=-1, eps=1e-6)
        
        # Combine
        poses = torch.cat([trans, quat], dim=-1)  # [B, n_steps, 7]
        
        # Compute loss if ground truth provided
        if 'gt_poses' in batch:
            gt_poses = batch['gt_poses']  # [B, n_steps, 7]
            
            # Get optional sample weights for anomalous data
            sample_weights = batch.get('anomalous_mask', None)
            if sample_weights is not None:
                # Convert mask to weights (invert: True->0, False->1)
                sample_weights = (~sample_weights).float()  # [B]
            
            # Split predictions and ground truth
            pred_trans = trans
            pred_rot = quat
            gt_trans = gt_poses[..., :3]
            gt_rot = gt_poses[..., 3:]
            
            # Scale-normalized translation loss with gradient clipping for small motions
            trans_norms = gt_trans.norm(dim=-1)  # [B, n_steps]
            small_motion_mask = trans_norms < 0.01  # 1cm threshold, [B, n_steps]
            
            # Use scale normalization
            scale_norm = gt_trans.flatten(0, 1).norm(dim=-1).mean().clamp(min=1.0)
            
            # Compute loss with potential gradient clipping
            if small_motion_mask.any():
                # Flatten for indexing
                pred_trans_flat = pred_trans.reshape(-1, 3)
                gt_trans_flat = gt_trans.reshape(-1, 3)
                small_motion_mask_flat = small_motion_mask.reshape(-1)
                
                # For small motions, use MSE loss (smoother gradients)
                if small_motion_mask_flat.any():
                    trans_loss_small = F.mse_loss(
                        pred_trans_flat[small_motion_mask_flat] / scale_norm,
                        gt_trans_flat[small_motion_mask_flat] / scale_norm,
                        reduction='mean'
                    )
                else:
                    trans_loss_small = 0.0
                    
                if (~small_motion_mask_flat).any():
                    trans_loss_large = F.smooth_l1_loss(
                        pred_trans_flat[~small_motion_mask_flat] / scale_norm,
                        gt_trans_flat[~small_motion_mask_flat] / scale_norm,
                        reduction='mean'
                    )
                else:
                    trans_loss_large = 0.0
                
                # Weighted combination
                small_weight = small_motion_mask_flat.float().mean()
                trans_loss_raw = small_weight * trans_loss_small + (1 - small_weight) * trans_loss_large
            else:
                trans_loss_raw = F.smooth_l1_loss(
                    pred_trans / scale_norm, 
                    gt_trans / scale_norm, 
                    reduction='mean'
                )
            
            # Geodesic rotation loss
            rot_loss_raw = self._robust_geodesic_loss(pred_rot, gt_rot) * 10.0
            
            # Homoscedastic uncertainty weighting
            s_t = torch.clamp(self.s_t, -2., 2.)
            s_r = torch.clamp(self.s_r, -2., 2.)
            
            loss = (torch.exp(-s_t) * trans_loss_raw + 
                    torch.exp(-s_r) * rot_loss_raw + 
                    s_t + s_r + 
                    1e-4 * (s_t**2 + s_r**2))
            
            # Path length regularization (curriculum learning)
            pred_path_length = pred_trans.norm(dim=-1).mean(dim=1)
            gt_path_length = gt_trans.norm(dim=-1).mean(dim=1)
            path_loss = F.mse_loss(pred_path_length, gt_path_length)
            
            # Optional velocity smoothness loss (helps prevent scale drift)
            if self.n_steps > 2:  # Need at least 3 frames for velocity differences
                # Translation velocity smoothness
                pred_vel = pred_trans[:, 1:] - pred_trans[:, :-1]  # [B, n_steps-1, 3]
                pred_accel = pred_vel[:, 1:] - pred_vel[:, :-1]     # [B, n_steps-2, 3]
                trans_smooth_loss = pred_accel.abs().mean()
                
                # Rotation velocity smoothness (simplified angular velocity)
                # Compute dot products between consecutive quaternions
                q_dots = (pred_rot[:, 1:] * pred_rot[:, :-1]).sum(dim=-1).abs()  # [B, n_steps-1]
                q_dots = torch.clamp(q_dots, -1.0, 1.0)
                angular_diffs = 2.0 * torch.acos(q_dots)  # Angular differences
                angular_vel = angular_diffs[:, 1:] - angular_diffs[:, :-1]  # Angular acceleration
                rot_smooth_loss = angular_vel.abs().mean()
                
                velocity_loss = trans_smooth_loss + 0.1 * rot_smooth_loss
            else:
                velocity_loss = torch.tensor(0.0, device=pred_trans.device)
            
            # Curriculum weight
            path_weight = min(0.02 + (0.08 * epoch / 5.0), 0.1)
            velocity_weight = 0.01  # Small weight for velocity smoothness
            loss = loss + path_weight * path_loss + velocity_weight * velocity_loss
            
            # Apply per-sample weights if provided
            if sample_weights is not None:
                # Reshape loss to per-sample if needed
                if loss.dim() == 0:  # Already reduced
                    loss = loss * sample_weights.mean()
                else:
                    loss = (loss * sample_weights).mean()
            
            return {
                'poses': poses,
                'total_loss': loss,
                'trans_loss_raw': trans_loss_raw,
                'rot_loss_raw': rot_loss_raw,
                'path_loss': path_loss,
                'velocity_loss': velocity_loss,
                's_t': self.s_t.detach(),
                's_r': self.s_r.detach(),
                's_t_c': s_t.detach(),
                's_r_c': s_r.detach()
            }
        else:
            return {'poses': poses}


class PureIMUEncoderWithScale(PureIMUEncoder):
    """Pure IMU encoder with constrained learnable global scale parameter."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize raw scale parameter to 0 (will map to scale=1.0)
        self.scale_raw = nn.Parameter(torch.zeros(1))
    
    def get_scale(self):
        """Get constrained scale value: 1 + 0.1 * tanh(scale_raw)."""
        # This constrains scale to range [0.9, 1.1]
        scale = 1.0 + 0.1 * torch.tanh(self.scale_raw)
        # Additional hard clamp for extra safety
        return torch.clamp(scale, 0.9, 1.1)
    
    def forward(self, batch, epoch: int = 0):
        """Forward pass with constrained scale learning."""
        # Enforce scale bounds on parameter directly
        self.scale_raw.data.clamp_(-2.0, 2.0)  # This maps to [0.8, 1.2] after tanh
        
        # Get base predictions
        predictions = super().forward(batch, epoch)
        
        # Apply learned scale to translations
        if 'poses' in predictions:
            poses = predictions['poses']
            poses_scaled = poses.clone()
            
            # Get constrained scale
            scale = self.get_scale()
            poses_scaled[..., :3] = poses[..., :3] * scale
            
            predictions['poses'] = poses_scaled
            predictions['learned_scale'] = scale.item()
            predictions['scale_raw'] = self.scale_raw.item()
        
        return predictions