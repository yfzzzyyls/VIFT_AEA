"""
Multi-Head VIO Architecture
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
    Uses attention patterns optimized for rotational motion.
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
        
        # Rotation output layers
        self.rotation_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4)  # Quaternion output
        )
        
        # Angular velocity prediction (auxiliary task)
        self.angular_velocity_output = nn.Linear(hidden_dim, 3)
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, seq_len, input_dim]
        
        Returns:
            Dictionary with rotation and angular velocity predictions
        """
        # Process features for rotation
        rot_features = self.rotation_processor(features)  # [B, seq_len, hidden_dim]
        
        # Apply rotation-specific attention
        attended_features = self.angular_transformer(rot_features)  # [B, seq_len, hidden_dim]
        
        # Use last frame for prediction
        final_features = attended_features[:, -1, :]  # [B, hidden_dim]
        
        # Predict rotation and angular velocity
        rotation_pred = self.rotation_output(final_features)  # [B, 4]
        angular_velocity_pred = self.angular_velocity_output(final_features)  # [B, 3]
        
        # Normalize quaternion
        rotation_pred = rotation_pred / (torch.norm(rotation_pred, dim=-1, keepdim=True) + 1e-8)
        
        return {
            'rotation': rotation_pred,
            'angular_velocity': angular_velocity_pred,
            'rotation_features': final_features
        }


class TranslationSpecializedHead(nn.Module):
    """
    Specialized head for translation prediction with linear velocity focus.
    Uses attention patterns optimized for translational motion.
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
        
        # Translation output layers
        self.translation_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # XYZ output
        )
        
        # Linear velocity prediction (auxiliary task)
        self.linear_velocity_output = nn.Linear(hidden_dim, 3)
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, seq_len, input_dim]
        
        Returns:
            Dictionary with translation and linear velocity predictions
        """
        # Process features for translation
        trans_features = self.translation_processor(features)  # [B, seq_len, hidden_dim]
        
        # Apply translation-specific attention
        attended_features = self.velocity_transformer(trans_features)  # [B, seq_len, hidden_dim]
        
        # Use last frame for prediction
        final_features = attended_features[:, -1, :]  # [B, hidden_dim]
        
        # Predict translation and linear velocity
        translation_pred = self.translation_output(final_features)  # [B, 3]
        linear_velocity_pred = self.linear_velocity_output(final_features)  # [B, 3]
        
        return {
            'translation': translation_pred,
            'linear_velocity': linear_velocity_pred,
            'translation_features': final_features
        }


class MultiHeadVIOModel(L.LightningModule):
    """
    Multi-head VIO model with specialized processing for rotation and translation.
    """
    
    def __init__(
        self,
        sequence_length: int = 11,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        num_transformer_layers: int = 4,
        num_attention_heads: int = 8,
        head_layers: int = 3,
        dropout: float = 0.1,
        use_auxiliary_tasks: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        rotation_weight: float = 1.0,
        translation_weight: float = 1.0,
        velocity_weight: float = 0.3
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.use_auxiliary_tasks = use_auxiliary_tasks
        
        # Feature encoders
        self.feature_encoder = ImageFeatureEncoder(output_dim=feature_dim)
        self.imu_encoder = IMUEncoder(output_dim=feature_dim // 2)
        
        # Shared feature processing
        combined_dim = feature_dim + feature_dim // 2
        self.shared_processor = PoseTransformer(
            input_dim=combined_dim,
            hidden_dim=hidden_dim,
            num_layers=num_transformer_layers,
            num_heads=num_attention_heads,
            max_sequence_length=sequence_length,
            dropout=dropout
        )
        
        # Specialized heads
        self.rotation_head = RotationSpecializedHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=head_layers,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        self.translation_head = TranslationSpecializedHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=head_layers,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Cross-modal fusion (optional)
        self.cross_modal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Loss functions
        self.arvr_loss = ARVRAdaptiveLoss()
        
        # Metrics
        self.train_rot_mae = MeanAbsoluteError()
        self.train_trans_mae = MeanAbsoluteError()
        self.val_rot_mae = MeanAbsoluteError()
        self.val_trans_mae = MeanAbsoluteError()
        
        # Loss weights
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
        self.velocity_weight = velocity_weight
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with specialized heads.
        
        Args:
            batch: Dictionary containing 'images', 'imus', 'poses'
        
        Returns:
            Dictionary with rotation and translation predictions
        """
        # Encode visual and IMU features
        visual_features = self._encode_visual_features(batch['images'])
        imu_features = self.imu_encoder(batch['imus'])
        
        # Combine features
        combined_features = torch.cat([visual_features, imu_features], dim=-1)
        
        # Shared processing
        shared_features = self.shared_processor(combined_features)
        
        # Specialized head processing
        rotation_output = self.rotation_head(shared_features)
        translation_output = self.translation_head(shared_features)
        
        # Optional cross-modal fusion
        if hasattr(self, 'use_cross_modal_fusion') and self.use_cross_modal_fusion:
            fused_features = torch.cat([
                rotation_output['rotation_features'],
                translation_output['translation_features']
            ], dim=-1)
            fused_features = self.cross_modal_fusion(fused_features)
            
            # Refined predictions
            rotation_refined = self.rotation_head.rotation_output(fused_features)
            translation_refined = self.translation_head.translation_output(fused_features)
            
            rotation_output['rotation'] = rotation_refined / (torch.norm(rotation_refined, dim=-1, keepdim=True) + 1e-8)
            translation_output['translation'] = translation_refined
        
        return {
            'rotation': rotation_output['rotation'],
            'translation': translation_output['translation'],
            'angular_velocity': rotation_output.get('angular_velocity'),
            'linear_velocity': translation_output.get('linear_velocity'),
            'rotation_features': rotation_output['rotation_features'],
            'translation_features': translation_output['translation_features']
        }
    
    def _encode_visual_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode visual features for the sequence.
        Handles both pre-extracted features and raw images.
        """
        batch_size, seq_len = images.shape[:2]
        
        # Check if input is already features (3D) or raw images (5D)
        if len(images.shape) == 3:
            # Already extracted features [B, seq_len, feature_dim]
            visual_features = images
            # Apply feature encoder if dimensions don't match
            if visual_features.shape[-1] != self.feature_dim:
                features_flat = visual_features.view(-1, visual_features.shape[-1])
                encoded_flat = self.feature_encoder(features_flat)
                visual_features = encoded_flat.view(batch_size, seq_len, -1)
        else:
            # Raw images [B, seq_len, C, H, W]
            images_flat = images.view(-1, *images.shape[2:])
            features_flat = self.feature_encoder(images_flat)
            visual_features = features_flat.view(batch_size, seq_len, -1)
        
        return visual_features
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with multi-head losses."""
        predictions = self(batch)
        
        # Target poses
        target_rotation = batch['poses'][:, -1, 3:7]
        target_translation = batch['poses'][:, -1, :3]
        
        # Main losses using standard MSE (ARVR loss expects different format)
        rotation_loss = nn.functional.mse_loss(predictions['rotation'], target_rotation)
        translation_loss = nn.functional.mse_loss(predictions['translation'], target_translation)
        
        total_loss = self.rotation_weight * rotation_loss + self.translation_weight * translation_loss
        
        # Auxiliary velocity losses (if enabled)
        if self.use_auxiliary_tasks and 'angular_velocity' in predictions:
            # Calculate target velocities from pose sequence
            if batch['poses'].shape[1] > 1:
                target_angular_vel = self._calculate_angular_velocity(batch['poses'])
                target_linear_vel = self._calculate_linear_velocity(batch['poses'])
                
                ang_vel_loss = nn.functional.mse_loss(predictions['angular_velocity'], target_angular_vel)
                lin_vel_loss = nn.functional.mse_loss(predictions['linear_velocity'], target_linear_vel)
                
                total_loss += self.velocity_weight * (ang_vel_loss + lin_vel_loss)
                
                self.log('train/angular_velocity_loss', ang_vel_loss)
                self.log('train/linear_velocity_loss', lin_vel_loss)
        
        # Log metrics
        self.train_rot_mae(predictions['rotation'].reshape(-1), target_rotation.reshape(-1))
        self.train_trans_mae(predictions['translation'].reshape(-1), target_translation.reshape(-1))
        
        self.log('train/rotation_loss', rotation_loss, prog_bar=True)
        self.log('train/translation_loss', translation_loss, prog_bar=True)
        self.log('train/total_loss', total_loss, prog_bar=True)
        self.log('train/rotation_mae', self.train_rot_mae, prog_bar=True)
        self.log('train/translation_mae', self.train_trans_mae, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step."""
        predictions = self(batch)
        
        target_rotation = batch['poses'][:, -1, 3:7]
        target_translation = batch['poses'][:, -1, :3]
        
        rotation_loss = nn.functional.mse_loss(predictions['rotation'], target_rotation)
        translation_loss = nn.functional.mse_loss(predictions['translation'], target_translation)
        total_loss = self.rotation_weight * rotation_loss + self.translation_weight * translation_loss
        
        self.val_rot_mae(predictions['rotation'].reshape(-1), target_rotation.reshape(-1))
        self.val_trans_mae(predictions['translation'].reshape(-1), target_translation.reshape(-1))
        
        self.log('val/rotation_loss', rotation_loss, prog_bar=True)
        self.log('val/translation_loss', translation_loss, prog_bar=True)
        self.log('val/total_loss', total_loss, prog_bar=True)
        self.log('val/rotation_mae', self.val_rot_mae, prog_bar=True)
        self.log('val/translation_mae', self.val_trans_mae, prog_bar=True)
    
    def _calculate_angular_velocity(self, poses: torch.Tensor) -> torch.Tensor:
        """Calculate angular velocity from pose sequence."""
        # Simplified angular velocity calculation
        # In practice, you'd want proper quaternion differentiation
        rotations = poses[:, :, 3:7]  # [B, seq_len, 4]
        if rotations.shape[1] < 2:
            return torch.zeros(rotations.shape[0], 3, device=rotations.device)
        
        # Simple finite difference approximation
        current_rot = rotations[:, -1]  # [B, 4]
        prev_rot = rotations[:, -2]     # [B, 4]
        
        # This is a simplified calculation - proper implementation would use
        # quaternion logarithm and handle the manifold properly
        angular_vel = (current_rot[:, :3] - prev_rot[:, :3]) * 30.0  # Assume 30 FPS
        return angular_vel
    
    def _calculate_linear_velocity(self, poses: torch.Tensor) -> torch.Tensor:
        """Calculate linear velocity from pose sequence."""
        translations = poses[:, :, :3]  # [B, seq_len, 3]
        if translations.shape[1] < 2:
            return torch.zeros(translations.shape[0], 3, device=translations.device)
        
        # Finite difference
        current_trans = translations[:, -1]  # [B, 3]
        prev_trans = translations[:, -2]     # [B, 3]
        linear_vel = (current_trans - prev_trans) * 30.0  # Assume 30 FPS
        return linear_vel
    
    def configure_optimizers(self):
        """Configure optimizer with separate learning rates for heads."""
        # Different learning rates for different components
        shared_params = list(self.shared_processor.parameters()) + \
                       list(self.feature_encoder.parameters()) + \
                       list(self.imu_encoder.parameters())
        
        head_params = list(self.rotation_head.parameters()) + \
                     list(self.translation_head.parameters())
        
        optimizer = torch.optim.AdamW([
            {'params': shared_params, 'lr': self.hparams.lr},
            {'params': head_params, 'lr': self.hparams.lr * 1.5}  # Higher LR for heads
        ], weight_decay=self.hparams.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/total_loss",
            },
        }