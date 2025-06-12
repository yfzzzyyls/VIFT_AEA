"""
Simple VIFT model with 7DoF output - following original architecture
Just changes the output from 6DoF to 7DoF (translation + quaternion)
"""

import torch
import torch.nn as nn
from .components.feature_encoder import FeatureEncoder
from .components.imu_encoder import IMUEncoder
from .components.pose_transformer import PoseTransformer


class VIFTSimple7DoF(nn.Module):
    """
    Original VIFT architecture with 7DoF output
    No separate heads - just a unified transformer as in the paper
    """
    
    def __init__(
        self,
        visual_dim=512,
        imu_dim=256,
        hidden_dim=128,
        num_heads=4,
        num_layers=6,
        dropout=0.1,
        sequence_length=10
    ):
        super().__init__()
        
        self.sequence_length = sequence_length
        
        # Feature encoders (already pre-extracted in our case)
        self.visual_encoder = nn.Linear(visual_dim, hidden_dim)
        self.imu_encoder = nn.Linear(imu_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, sequence_length * 2, hidden_dim))
        
        # Main transformer (as in original VIFT)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Output projection to 7DoF (3 translation + 4 quaternion)
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 7)  # 7DoF output
        )
        
        # Initialize output layer
        self._init_output_layer()
        
    def _init_output_layer(self):
        """Initialize output to predict small motions and identity rotation"""
        # Initialize final layer with small weights
        nn.init.xavier_uniform_(self.pose_head[-1].weight, gain=0.1)
        
        # Initialize bias: small translation, identity quaternion
        if self.pose_head[-1].bias is not None:
            self.pose_head[-1].bias.data.zero_()
            # Set quaternion w to 1 (identity rotation)
            self.pose_head[-1].bias.data[6] = 1.0
            
    def forward(self, batch):
        """
        Forward pass
        Args:
            batch: dict with 'visual_features' and 'imu_features'
        Returns:
            dict with 'poses' containing 7DoF predictions
        """
        visual_features = batch['visual_features']  # (B, seq_len, visual_dim)
        imu_features = batch['imu_features']       # (B, seq_len, imu_dim)
        
        B, seq_len = visual_features.shape[:2]
        
        # Encode features
        visual_encoded = self.visual_encoder(visual_features)  # (B, seq_len, hidden_dim)
        imu_encoded = self.imu_encoder(imu_features)          # (B, seq_len, hidden_dim)
        
        # Concatenate visual and IMU features
        combined = torch.cat([visual_encoded, imu_encoded], dim=1)  # (B, seq_len*2, hidden_dim)
        
        # Add positional encoding
        combined = combined + self.pos_encoder
        
        # Pass through transformer
        transformed = self.transformer(combined)  # (B, seq_len*2, hidden_dim)
        
        # Take only the visual positions (first half) for pose prediction
        pose_features = transformed[:, :seq_len]  # (B, seq_len, hidden_dim)
        
        # Predict poses
        poses = self.pose_head(pose_features)  # (B, seq_len, 7)
        
        # Normalize quaternions
        translations = poses[..., :3]
        quaternions = poses[..., 3:]
        quaternions = nn.functional.normalize(quaternions, p=2, dim=-1)
        
        # Combine normalized output
        normalized_poses = torch.cat([translations, quaternions], dim=-1)
        
        return {'poses': normalized_poses}


class VIFTSimple7DoFwithLoss(nn.Module):
    """VIFT model with integrated loss computation"""
    
    def __init__(self, model_config):
        super().__init__()
        self.model = VIFTSimple7DoF(**model_config)
        
    def forward(self, batch):
        """Forward pass with loss computation"""
        # Get predictions
        output = self.model(batch)
        pred_poses = output['poses']
        
        # Get ground truth
        gt_poses = batch['poses']
        
        # Compute losses
        pred_trans = pred_poses[..., :3]
        pred_rot = pred_poses[..., 3:]
        gt_trans = gt_poses[..., :3]
        gt_rot = gt_poses[..., 3:]
        
        # Translation loss (L1)
        trans_loss = nn.functional.l1_loss(pred_trans, gt_trans)
        
        # Rotation loss (MSE on normalized quaternions)
        pred_rot_norm = nn.functional.normalize(pred_rot, p=2, dim=-1)
        gt_rot_norm = nn.functional.normalize(gt_rot, p=2, dim=-1)
        
        # Handle quaternion ambiguity
        dot = (pred_rot_norm * gt_rot_norm).sum(dim=-1, keepdim=True)
        gt_rot_norm = torch.where(dot < 0, -gt_rot_norm, gt_rot_norm)
        
        rot_loss = nn.functional.mse_loss(pred_rot_norm, gt_rot_norm)
        
        # Combined loss
        total_loss = trans_loss + 10.0 * rot_loss  # Weight rotation more
        
        return {
            'loss': total_loss,
            'trans_loss': trans_loss,
            'rot_loss': rot_loss,
            'predictions': pred_poses
        }