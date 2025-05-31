"""
Multi-Head VIO Architecture - All Frames Prediction Version
Modified to predict poses for all frames in the sequence, similar to original VIFT.
Specialized processing heads for rotation and translation with different
attention patterns optimized for AR/VR motion characteristics.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import lightning as L
from torchmetrics import MeanAbsoluteError

from .components.pose_transformer_new import PoseTransformer
from .components.imu_encoder import IMUEncoder
from .components.feature_encoder import ImageFeatureEncoder
from ..metrics.arvr_loss import ARVRAdaptiveLoss


class RotationSpecializedHead(nn.Module):
    """
    Specialized head for rotation prediction with angular velocity focus.
    Modified to predict rotations for all frames in the sequence.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Rotation-specific feature processing
        self.rotation_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Angular velocity specific transformer
        self.angular_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Rotation output layers - applied to each timestep
        self.rotation_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4)  # Quaternion output
        )
        
        # Angular velocity prediction (auxiliary task) - for each timestep
        self.angular_velocity_output = nn.Linear(hidden_dim, 3)
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Modified to predict rotations for all frames.
        
        Args:
            features: [B, seq_len, input_dim]
        
        Returns:
            Dictionary with rotation and angular velocity predictions for all frames
        """
        # Process features for rotation
        rot_features = self.rotation_processor(features)  # [B, seq_len, hidden_dim]
        
        # Apply rotation-specific attention with causal mask
        attended_features = self.angular_transformer(rot_features)  # [B, seq_len, hidden_dim]
        
        # Predict for all frames
        B, seq_len, hidden_dim = attended_features.shape
        
        # Reshape for batch processing
        all_features = attended_features.reshape(B * seq_len, hidden_dim)
        
        # Predict rotation and angular velocity for all frames
        rotation_pred = self.rotation_output(all_features)  # [B*seq_len, 4]
        angular_velocity_pred = self.angular_velocity_output(all_features)  # [B*seq_len, 3]
        
        # Reshape back to sequence format
        rotation_pred = rotation_pred.reshape(B, seq_len, 4)
        angular_velocity_pred = angular_velocity_pred.reshape(B, seq_len, 3)
        
        # Normalize quaternions
        rotation_pred = rotation_pred / (torch.norm(rotation_pred, dim=-1, keepdim=True) + 1e-8)
        
        return {
            'rotation': rotation_pred,  # [B, seq_len, 4]
            'angular_velocity': angular_velocity_pred,  # [B, seq_len, 3]
            'rotation_features': attended_features  # [B, seq_len, hidden_dim]
        }


class TranslationSpecializedHead(nn.Module):
    """
    Specialized head for translation prediction with linear velocity focus.
    Modified to predict translations for all frames in the sequence.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Translation-specific feature processing
        self.translation_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Linear velocity specific transformer
        self.velocity_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Translation output layers - applied to each timestep
        self.translation_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # XYZ output
        )
        
        # Linear velocity prediction (auxiliary task) - for each timestep
        self.linear_velocity_output = nn.Linear(hidden_dim, 3)
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Modified to predict translations for all frames.
        
        Args:
            features: [B, seq_len, input_dim]
        
        Returns:
            Dictionary with translation and linear velocity predictions for all frames
        """
        # Process features for translation
        trans_features = self.translation_processor(features)  # [B, seq_len, hidden_dim]
        
        # Apply translation-specific attention with causal mask
        attended_features = self.velocity_transformer(trans_features)  # [B, seq_len, hidden_dim]
        
        # Predict for all frames
        B, seq_len, hidden_dim = attended_features.shape
        
        # Reshape for batch processing
        all_features = attended_features.reshape(B * seq_len, hidden_dim)
        
        # Predict translation and linear velocity for all frames
        translation_pred = self.translation_output(all_features)  # [B*seq_len, 3]
        linear_velocity_pred = self.linear_velocity_output(all_features)  # [B*seq_len, 3]
        
        # Reshape back to sequence format
        translation_pred = translation_pred.reshape(B, seq_len, 3)
        linear_velocity_pred = linear_velocity_pred.reshape(B, seq_len, 3)
        
        return {
            'translation': translation_pred,  # [B, seq_len, 3]
            'linear_velocity': linear_velocity_pred,  # [B, seq_len, 3]
            'translation_features': attended_features  # [B, seq_len, hidden_dim]
        }


class MultiHeadVIOModel(L.LightningModule):
    """
    Multi-Head VIO model with specialized processing for rotation and translation.
    Modified to predict poses for all frames in the sequence.
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        num_shared_layers: int = 4,
        num_specialized_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        rotation_weight: float = 1.0,
        translation_weight: float = 1.0,
        velocity_weight: float = 0.3,
        sequence_length: int = 11
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Feature encoders
        self.feature_encoder = ImageFeatureEncoder(
            input_dim=feature_dim,
            output_dim=feature_dim,
            hidden_dim=512,
            num_layers=3,
            dropout=dropout
        )
        
        self.imu_encoder = IMUEncoder(
            input_dim=6,
            output_dim=feature_dim // 2,
            hidden_dim=128,
            dropout=dropout
        )
        
        # Combined feature dimension
        combined_dim = feature_dim + feature_dim // 2
        
        # Shared transformer for initial processing
        self.shared_processor = PoseTransformer(
            input_dim=combined_dim,
            hidden_dim=hidden_dim,
            num_layers=num_shared_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_sequence_length=sequence_length + 5  # Some buffer
        )
        
        # Specialized heads
        self.rotation_head = RotationSpecializedHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_specialized_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.translation_head = TranslationSpecializedHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_specialized_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Optional cross-modal fusion
        self.cross_modal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Loss function
        self.arvr_loss = ARVRAdaptiveLoss()
        
        # Metrics
        self.train_rot_mae = MeanAbsoluteError()
        self.train_trans_mae = MeanAbsoluteError()
        self.val_rot_mae = MeanAbsoluteError()
        self.val_trans_mae = MeanAbsoluteError()
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass predicting poses for all frames.
        
        Args:
            batch: Dictionary with 'images', 'imus', 'poses'
        
        Returns:
            Dictionary with rotation and translation predictions for all frames
        """
        # Encode visual features
        B, seq_len, feature_dim = batch['images'].shape
        # Flatten for feature encoder that expects [B*seq_len, feature_dim]
        images_flat = batch['images'].reshape(B * seq_len, feature_dim)
        visual_features_flat = self.feature_encoder(images_flat)  # [B*seq_len, feature_dim]
        visual_features = visual_features_flat.reshape(B, seq_len, -1)  # [B, seq_len, feature_dim]
        
        # Encode IMU features
        imu_features = self.imu_encoder(batch['imus'])  # [B, seq_len, feature_dim//2]
        
        # Combine features
        combined_features = torch.cat([visual_features, imu_features], dim=-1)
        
        # Shared transformer processing
        shared_features = self.shared_processor(combined_features)  # [B, seq_len, hidden_dim]
        
        # Get predictions from specialized heads
        rotation_outputs = self.rotation_head(shared_features)
        translation_outputs = self.translation_head(shared_features)
        
        # Optional: Cross-modal fusion (applied to each timestep)
        B, seq_len, hidden_dim = shared_features.shape
        
        # Concatenate rotation and translation features
        rot_features = rotation_outputs['rotation_features']  # [B, seq_len, hidden_dim]
        trans_features = translation_outputs['translation_features']  # [B, seq_len, hidden_dim]
        
        # Reshape for fusion
        combined = torch.cat([rot_features, trans_features], dim=-1)  # [B, seq_len, hidden_dim*2]
        combined_flat = combined.reshape(B * seq_len, hidden_dim * 2)
        fused_flat = self.cross_modal_fusion(combined_flat)
        fused_features = fused_flat.reshape(B, seq_len, hidden_dim)  # [B, seq_len, hidden_dim]
        
        # Add fusion residual to predictions (small contribution)
        fusion_weight = 0.1
        
        return {
            'rotation': rotation_outputs['rotation'],  # [B, seq_len, 4]
            'translation': translation_outputs['translation'],  # [B, seq_len, 3]
            'angular_velocity': rotation_outputs['angular_velocity'],  # [B, seq_len, 3]
            'linear_velocity': translation_outputs['linear_velocity']  # [B, seq_len, 3]
        }
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                     batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for all frames.
        
        Args:
            predictions: Model predictions for all frames
            batch: Ground truth data
        
        Returns:
            Dictionary with individual loss components
        """
        # Target poses for all frames (excluding first frame as it's reference)
        target_rotation = batch['poses'][:, 1:, 3:7]  # [B, seq_len-1, 4]
        target_translation = batch['poses'][:, 1:, :3]  # [B, seq_len-1, 3]
        
        # Predictions (excluding first frame to match targets)
        pred_rotation = predictions['rotation'][:, 1:, :]  # [B, seq_len-1, 4]
        pred_translation = predictions['translation'][:, 1:, :]  # [B, seq_len-1, 3]
        
        # Compute losses using MSE (AR/VR loss expects 3D tensors, we have 4D quaternions)
        rotation_loss = nn.functional.mse_loss(pred_rotation.reshape(-1, 4), 
                                               target_rotation.reshape(-1, 4))
        translation_loss = nn.functional.mse_loss(pred_translation.reshape(-1, 3), 
                                                 target_translation.reshape(-1, 3))
        
        # Velocity losses (computed from consecutive frame differences)
        if predictions['angular_velocity'].shape[1] > 1:
            # Compute target velocities from ground truth
            target_angular_vel = target_rotation[:, 1:] - target_rotation[:, :-1]  # [B, seq_len-2, 4]
            pred_angular_vel = predictions['angular_velocity'][:, 2:, :]  # [B, seq_len-2, 3]
            
            target_linear_vel = target_translation[:, 1:] - target_translation[:, :-1]  # [B, seq_len-2, 3]
            pred_linear_vel = predictions['linear_velocity'][:, 2:, :]  # [B, seq_len-2, 3]
            
            # Only use first 3 components of angular velocity (ignore w component)
            angular_vel_loss = nn.functional.mse_loss(pred_angular_vel, target_angular_vel[:, :, :3])
            linear_vel_loss = nn.functional.mse_loss(pred_linear_vel, target_linear_vel)
        else:
            angular_vel_loss = torch.tensor(0.0, device=rotation_loss.device)
            linear_vel_loss = torch.tensor(0.0, device=rotation_loss.device)
        
        # Total loss
        total_loss = (self.hparams.rotation_weight * rotation_loss + 
                     self.hparams.translation_weight * translation_loss +
                     self.hparams.velocity_weight * (angular_vel_loss + linear_vel_loss))
        
        return {
            'total_loss': total_loss,
            'rotation_loss': rotation_loss,
            'translation_loss': translation_loss,
            'angular_velocity_loss': angular_vel_loss,
            'linear_velocity_loss': linear_vel_loss
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with all-frame prediction."""
        predictions = self(batch)
        losses = self.compute_loss(predictions, batch)
        
        # Log losses
        for name, loss in losses.items():
            self.log(f'train/{name}', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Update metrics (using mean across all frames)
        pred_rot = predictions['rotation'][:, 1:, :].reshape(-1, 4).contiguous()
        target_rot = batch['poses'][:, 1:, 3:7].reshape(-1, 4).contiguous()
        self.train_rot_mae(pred_rot, target_rot)
        
        pred_trans = predictions['translation'][:, 1:, :].reshape(-1, 3).contiguous()
        target_trans = batch['poses'][:, 1:, :3].reshape(-1, 3).contiguous()
        self.train_trans_mae(pred_trans, target_trans)
        
        self.log('train/rotation_mae', self.train_rot_mae, on_step=True, on_epoch=True)
        self.log('train/translation_mae', self.train_trans_mae, on_step=True, on_epoch=True)
        
        # Log current learning rate (handled by LearningRateMonitor callback now)
        # But we can still log it here for consistency with original VIFT
        if self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('train/lr', lr, on_step=True, on_epoch=False, prog_bar=False)
        
        return losses['total_loss']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step with all-frame prediction."""
        predictions = self(batch)
        losses = self.compute_loss(predictions, batch)
        
        # Log losses
        for name, loss in losses.items():
            self.log(f'val/{name}', loss, on_epoch=True)
        
        # Update metrics (using mean across all frames)
        pred_rot = predictions['rotation'][:, 1:, :].reshape(-1, 4).contiguous()
        target_rot = batch['poses'][:, 1:, 3:7].reshape(-1, 4).contiguous()
        self.val_rot_mae(pred_rot, target_rot)
        
        pred_trans = predictions['translation'][:, 1:, :].reshape(-1, 3).contiguous()
        target_trans = batch['poses'][:, 1:, :3].reshape(-1, 3).contiguous()
        self.val_trans_mae(pred_trans, target_trans)
        
        self.log('val/rotation_mae', self.val_rot_mae, on_epoch=True)
        self.log('val/translation_mae', self.val_trans_mae, on_epoch=True)
    
    def configure_optimizers(self):
        """Configure optimizer with parameter groups."""
        # Different learning rates for different components
        param_groups = [
            {'params': self.feature_encoder.parameters(), 'lr': self.hparams.learning_rate * 0.5},
            {'params': self.imu_encoder.parameters(), 'lr': self.hparams.learning_rate * 0.5},
            {'params': self.shared_processor.parameters(), 'lr': self.hparams.learning_rate},
            {'params': self.rotation_head.parameters(), 'lr': self.hparams.learning_rate * 1.5},
            {'params': self.translation_head.parameters(), 'lr': self.hparams.learning_rate * 1.5},
            {'params': self.cross_modal_fusion.parameters(), 'lr': self.hparams.learning_rate}
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay
        )
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,  # Adjust based on expected epochs
            eta_min=self.hparams.learning_rate * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/total_loss'
            }
        }