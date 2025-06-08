"""
MultiHeadVIOModel - Multi-head Visual-Inertial Odometry model with proper weight initialization
Uses separate visual and IMU processing streams with specialized heads for rotation and translation
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import lightning as L
from torchmetrics import MeanAbsoluteError
import numpy as np

from .components.pose_transformer_new import PoseTransformer
from ..metrics.arvr_loss_wrapper import ARVRLossWrapper


class QuaternionMAE(nn.Module):
    """Proper quaternion MAE that computes geodesic distance"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        """
        Compute geodesic distance between quaternions
        Args:
            pred: [N, 4] predicted quaternions (XYZW)
            target: [N, 4] target quaternions (XYZW)
        Returns:
            Scalar MAE in radians
        """
        # Normalize quaternions
        pred = pred / (torch.norm(pred, dim=-1, keepdim=True) + 1e-8)
        target = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-8)
        
        # Compute dot product
        dot = torch.sum(pred * target, dim=-1)
        
        # Clamp to avoid numerical issues
        dot = torch.clamp(dot, -1.0, 1.0)
        
        # Geodesic distance
        angle = 2 * torch.acos(torch.abs(dot))
        
        return angle.mean()


class MultiModalPoseTransformer(nn.Module):
    """Multi-modal transformer for visual-inertial fusion"""
    
    def __init__(
        self,
        visual_dim: int = 512,
        imu_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_sequence_length: int = 15
    ):
        super().__init__()
        
        # Modal-specific projections
        self.visual_projection = nn.Linear(visual_dim, hidden_dim)
        self.imu_projection = nn.Linear(imu_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_sequence_length, hidden_dim) * 0.02
        )
        
        # Modal type embeddings
        self.visual_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.imu_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Transformer with proper initialization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Initialize projections properly
        nn.init.xavier_uniform_(self.visual_projection.weight)
        nn.init.xavier_uniform_(self.imu_projection.weight)
        nn.init.zeros_(self.visual_projection.bias)
        nn.init.zeros_(self.imu_projection.bias)
        
    def forward(self, visual_features: torch.Tensor, imu_features: torch.Tensor) -> torch.Tensor:
        B, seq_len, _ = visual_features.shape
        
        # Project features
        visual_proj = self.visual_projection(visual_features)
        imu_proj = self.imu_projection(imu_features)
        
        # Add modal type embeddings
        visual_proj = visual_proj + self.visual_type_embedding
        imu_proj = imu_proj + self.imu_type_embedding
        
        # Add positional encoding
        visual_proj = visual_proj + self.positional_encoding[:, :seq_len]
        imu_proj = imu_proj + self.positional_encoding[:, :seq_len]
        
        # Interleave visual and IMU features
        combined = torch.stack([visual_proj, imu_proj], dim=2).reshape(B, seq_len * 2, -1)
        
        # Apply transformer
        output = self.transformer(combined)
        
        # Take mean of visual and IMU outputs for each timestep
        visual_out = output[:, 0::2]  # Even indices
        imu_out = output[:, 1::2]      # Odd indices
        fused = (visual_out + imu_out) / 2
        
        return fused


class RotationSpecializedHead(nn.Module):
    """Specialized head for rotation prediction with PROPER initialization"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Rotation-specific processing
        self.rotation_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Angular transformer
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
        
        # Output layer for quaternion - no activation function!
        self.rotation_output = nn.Linear(hidden_dim, 4)
        
        # CRITICAL FIX: Use Xavier initialization instead of zeros!
        nn.init.xavier_uniform_(self.rotation_output.weight, gain=0.1)  # Small gain for stability
        nn.init.zeros_(self.rotation_output.bias)
        # Bias initialization for identity quaternion in XYZW format (model's output format)
        self.rotation_output.bias.data = torch.tensor([0.0, 0.0, 0.0, 1.0])
        
        self.angular_velocity_output = nn.Linear(hidden_dim, 3)
        nn.init.xavier_uniform_(self.angular_velocity_output.weight)
        nn.init.zeros_(self.angular_velocity_output.bias)
        
        # Initialize processor properly
        for module in self.rotation_processor.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(module.bias)
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, seq_len, _ = features.shape
        
        # Process features
        rot_features = self.rotation_processor(features)
        attended_features = self.angular_transformer(rot_features)
        
        # Reshape for batch processing
        all_features = attended_features.reshape(B * seq_len, -1)
        
        # Predictions - no ReLU on quaternion output!
        rotation_pred = self.rotation_output(all_features)
        angular_velocity_pred = self.angular_velocity_output(all_features)
        
        # Reshape back
        rotation_pred = rotation_pred.reshape(B, seq_len, 4)
        angular_velocity_pred = angular_velocity_pred.reshape(B, seq_len, 3)
        
        # Normalize quaternions (already in XYZW format from output layer)
        rotation_pred = rotation_pred / (torch.norm(rotation_pred, dim=-1, keepdim=True) + 1e-8)
        
        return {
            'rotation': rotation_pred,  # Already in XYZW format
            'angular_velocity': angular_velocity_pred,
            'rotation_features': attended_features
        }


class TranslationSpecializedHead(nn.Module):
    """Specialized head for translation prediction"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Translation-specific processing
        self.translation_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Initialize processor layers with Kaiming
        for module in self.translation_processor.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(module.bias)
        
        # Spatial cross-attention
        self.spatial_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Translation transformer
        self.translation_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Output layers
        self.translation_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # XYZ translation
        )
        
        self.velocity_output = nn.Linear(hidden_dim, 3)
        
        # Initialize output layers
        for module in self.translation_output.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        nn.init.xavier_uniform_(self.velocity_output.weight)
        nn.init.zeros_(self.velocity_output.bias)
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, seq_len, _ = features.shape
        
        # Process features
        trans_features = self.translation_processor(features)
        
        # Apply spatial attention
        attended_features, _ = self.spatial_cross_attention(
            trans_features, trans_features, trans_features
        )
        attended_features = attended_features + trans_features
        
        # Apply transformer
        refined_features = self.translation_transformer(attended_features)
        
        # Reshape for batch processing
        all_features = refined_features.reshape(B * seq_len, -1)
        
        # Predictions
        translation_pred = self.translation_output(all_features)
        velocity_pred = self.velocity_output(all_features)
        
        # Reshape back
        translation_pred = translation_pred.reshape(B, seq_len, 3)
        velocity_pred = velocity_pred.reshape(B, seq_len, 3)
        
        return {
            'translation': translation_pred,
            'linear_velocity': velocity_pred,
            'translation_features': refined_features
        }


class MultiHeadVIOModel(L.LightningModule):
    """
    Multi-head VIO model with separate visual and IMU processing
    Features: Proper weight initialization, geodesic loss, and specialized heads
    """
    
    def __init__(
        self,
        visual_dim: int = 512,
        imu_dim: int = 256,
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
        sequence_length: int = 10
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Multi-modal transformer for shared processing
        self.shared_processor = MultiModalPoseTransformer(
            visual_dim=visual_dim,
            imu_dim=imu_dim,
            hidden_dim=hidden_dim,
            num_layers=num_shared_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_sequence_length=sequence_length + 5
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
        
        # Cross-modal fusion layer
        self.cross_modal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initialize fusion layer
        for module in self.cross_modal_fusion.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Loss function with improved settings
        self.arvr_loss = ARVRLossWrapper(use_log_scale=True, use_weighted_loss=False)
        
        # Metrics - use proper quaternion MAE
        self.train_rot_mae = QuaternionMAE()
        self.train_trans_mae = MeanAbsoluteError()
        self.val_rot_mae = QuaternionMAE()
        self.val_trans_mae = MeanAbsoluteError()
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with separate visual and IMU features
        
        Args:
            batch: Dictionary with 'visual_features', 'imu_features', 'poses'
        
        Returns:
            Dictionary with rotation and translation predictions
        """
        # Extract separate features
        visual_features = batch.get('visual_features', None)
        imu_features = batch.get('imu_features', None)
        
        # Handle backward compatibility
        if visual_features is None or imu_features is None:
            # Fall back to concatenated features
            combined_features = batch['images']  # [B, seq_len, 768]
            visual_features = combined_features[..., :512]
            imu_features = combined_features[..., 512:]
        
        # Process through shared transformer
        shared_features = self.shared_processor(visual_features, imu_features)
        
        # Get predictions from specialized heads
        rotation_outputs = self.rotation_head(shared_features)
        translation_outputs = self.translation_head(shared_features)
        
        # Optional cross-modal fusion
        B, seq_len, hidden_dim = shared_features.shape
        
        rot_features = rotation_outputs['rotation_features']
        trans_features = translation_outputs['translation_features']
        
        combined = torch.cat([rot_features, trans_features], dim=-1)
        combined_flat = combined.reshape(B * seq_len, -1)
        fused_flat = self.cross_modal_fusion(combined_flat)
        fused_features = fused_flat.reshape(B, seq_len, hidden_dim)
        
        return {
            'rotation': rotation_outputs['rotation'],
            'translation': translation_outputs['translation'],
            'angular_velocity': rotation_outputs['angular_velocity'],
            'linear_velocity': translation_outputs['linear_velocity'],
            'fused_features': fused_features
        }
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                     batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute loss for all frames"""
        # Target poses (all are transitions now, no need to skip first)
        target_rotation = batch['poses'][:, :, 3:7]
        target_translation = batch['poses'][:, :, :3]
        
        # Predictions
        pred_rotation = predictions['rotation']
        pred_translation = predictions['translation']
        
        # Flatten for loss computation
        B, seq_len, _ = pred_rotation.shape
        
        pred_rot = pred_rotation.reshape(-1, 4).contiguous()
        target_rot = target_rotation.reshape(-1, 4).contiguous()
        pred_trans = pred_translation.reshape(-1, 3).contiguous()
        target_trans = target_translation.reshape(-1, 3).contiguous()
        
        # Compute losses
        loss_dict = self.arvr_loss(
            pred_rotation=pred_rot,
            target_rotation=target_rot,
            pred_translation=pred_trans,
            target_translation=target_trans
        )
        
        # Weight losses
        loss_dict['rotation_loss'] *= self.hparams.rotation_weight
        loss_dict['translation_loss'] *= self.hparams.translation_weight
        
        return loss_dict
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        loss_dict = self.compute_loss(predictions, batch)
        
        total_loss = loss_dict.get('total_loss', sum(v for k, v in loss_dict.items() if k != 'total_loss'))
        
        # Log losses with actual values (not just 0.0000)
        self.log('train/total_loss', total_loss, prog_bar=True)
        for key, value in loss_dict.items():
            if key != 'total_loss':
                self.log(f'train/{key}', value)
        
        # Log gradient norms for debugging
        if batch_idx % 100 == 0:  # Every 100 batches
            total_norm = 0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2.)
            self.log('train/grad_norm', total_norm)
        
        # Update metrics - All frames are transitions now
        with torch.no_grad():
            pred_rot = predictions['rotation'].reshape(-1, 4).contiguous()
            target_rot = batch['poses'][:, :, 3:7].reshape(-1, 4).contiguous()
            pred_trans = predictions['translation'].reshape(-1, 3).contiguous()
            target_trans = batch['poses'][:, :, :3].reshape(-1, 3).contiguous()
            
            # Use custom quaternion MAE
            rot_mae = self.train_rot_mae(pred_rot, target_rot)
            self.train_trans_mae(pred_trans, target_trans)
            
            # Log in radians and degrees
            self.log('train/rot_mae_rad', rot_mae, prog_bar=True)
            self.log('train/rot_mae_deg', torch.rad2deg(rot_mae), prog_bar=False)
            self.log('train/trans_mae', self.train_trans_mae, prog_bar=True)
            
            # Log prediction statistics
            if batch_idx % 100 == 0:
                self.log('train/pred_trans_mean', pred_trans.abs().mean())
                self.log('train/pred_trans_std', pred_trans.std())
                self.log('train/target_trans_mean', target_trans.abs().mean())
                self.log('train/target_trans_std', target_trans.std())
                
                # Log rotation statistics
                self.log('train/pred_rot_std', pred_rot.std())
                self.log('train/target_rot_std', target_rot.std())
                
                # Check if predictions are changing
                rot_diff = (pred_rot[1:] - pred_rot[:-1]).abs().mean()
                self.log('train/pred_rot_variation', rot_diff)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        loss_dict = self.compute_loss(predictions, batch)
        
        total_loss = loss_dict.get('total_loss', sum(v for k, v in loss_dict.items() if k != 'total_loss'))
        
        # Log losses
        self.log('val/total_loss', total_loss, prog_bar=True)
        for key, value in loss_dict.items():
            if key != 'total_loss':
                self.log(f'val/{key}', value)
        
        # Update metrics - All frames are transitions
        with torch.no_grad():
            pred_rot = predictions['rotation'].reshape(-1, 4).contiguous()
            target_rot = batch['poses'][:, :, 3:7].reshape(-1, 4).contiguous()
            pred_trans = predictions['translation'].reshape(-1, 3).contiguous()
            target_trans = batch['poses'][:, :, :3].reshape(-1, 3).contiguous()
            
            rot_mae = self.val_rot_mae(pred_rot, target_rot)
            self.val_trans_mae(pred_trans, target_trans)
            
            self.log('val/rot_mae_rad', rot_mae, prog_bar=True)
            self.log('val/rot_mae_deg', torch.rad2deg(rot_mae), prog_bar=False)
            self.log('val/trans_mae', self.val_trans_mae, prog_bar=True)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.05,
            anneal_strategy='cos'
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }