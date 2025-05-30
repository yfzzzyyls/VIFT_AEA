"""
Multi-Scale Temporal VIO Model
Processes multiple temporal scales (short, medium, long-term) to capture
different types of motion dependencies in AR/VR scenarios.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import lightning as L
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from .components.pose_transformer_new import PoseTransformer
from .components.imu_encoder import IMUEncoder
from .components.feature_encoder import ImageFeatureEncoder


class MultiScaleTemporalVIO(L.LightningModule):
    """
    Multi-scale temporal VIO model that processes different sequence lengths
    to capture short-term dynamics, medium-term patterns, and long-term context.
    """
    
    def __init__(
        self,
        sequence_lengths: List[int] = [7, 11, 15],  # Short, medium, long-term
        feature_dim: int = 768,
        hidden_dim: int = 256,
        num_transformer_layers: int = 4,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        use_scale_weights: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.sequence_lengths = sequence_lengths
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.use_scale_weights = use_scale_weights
        
        # Feature encoder (shared across scales)
        self.feature_encoder = ImageFeatureEncoder(output_dim=feature_dim)
        self.imu_encoder = IMUEncoder(output_dim=feature_dim // 2)
        
        # Multi-scale transformers
        self.scale_transformers = nn.ModuleDict()
        for i, seq_len in enumerate(sequence_lengths):
            scale_name = f"scale_{seq_len}"
            self.scale_transformers[scale_name] = PoseTransformer(
                input_dim=feature_dim + feature_dim // 2,  # Visual + IMU features
                hidden_dim=hidden_dim,
                num_layers=num_transformer_layers,
                num_heads=num_attention_heads,
                max_sequence_length=seq_len,
                dropout=dropout
            )
        
        # Scale fusion network
        self.scale_fusion = nn.Sequential(
            nn.Linear(len(sequence_lengths) * hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output heads
        self.rotation_head = nn.Linear(hidden_dim, 4)  # Quaternion
        self.translation_head = nn.Linear(hidden_dim, 3)  # XYZ
        
        # Scale attention weights (learnable)
        if use_scale_weights:
            self.scale_attention = nn.Parameter(torch.ones(len(sequence_lengths)))
        
        # Metrics
        self.train_rot_mae = MeanAbsoluteError()
        self.train_trans_mae = MeanAbsoluteError()
        self.val_rot_mae = MeanAbsoluteError()
        self.val_trans_mae = MeanAbsoluteError()
        
        # Loss function with AR/VR specific weighting
        self.rotation_weight = 1.0
        self.translation_weight = 1.0
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass processing multiple temporal scales.
        
        Args:
            batch: Dictionary containing:
                - 'images': [B, max_seq_len, C, H, W]
                - 'imus': [B, max_seq_len, imu_dim]
                - 'poses': [B, max_seq_len, 7] (target poses)
        
        Returns:
            Dictionary with predictions for each scale and fused output
        """
        batch_size = batch['images'].shape[0]
        max_seq_len = batch['images'].shape[1]
        
        # Encode features for full sequence
        visual_features = self._encode_visual_features(batch['images'])  # [B, max_seq_len, feat_dim]
        imu_features = self.imu_encoder(batch['imus'])  # [B, max_seq_len, feat_dim//2]
        
        # Combine visual and IMU features
        combined_features = torch.cat([visual_features, imu_features], dim=-1)
        
        # Process each temporal scale
        scale_outputs = {}
        scale_features = []
        
        for i, seq_len in enumerate(self.sequence_lengths):
            scale_name = f"scale_{seq_len}"
            
            # Extract subsequence (use last seq_len frames)
            if max_seq_len >= seq_len:
                start_idx = max_seq_len - seq_len
                scale_input = combined_features[:, start_idx:, :]
            else:
                # Pad if sequence is shorter than required
                padding = torch.zeros(batch_size, seq_len - max_seq_len, combined_features.shape[-1])
                if combined_features.is_cuda:
                    padding = padding.cuda()
                scale_input = torch.cat([padding, combined_features], dim=1)
            
            # Process through scale-specific transformer
            scale_output = self.scale_transformers[scale_name](scale_input)
            scale_outputs[scale_name] = scale_output
            
            # Extract final hidden state for fusion
            scale_features.append(scale_output[:, -1, :])  # [B, hidden_dim]
        
        # Fuse multi-scale features
        if self.use_scale_weights:
            # Apply learned attention weights
            attention_weights = torch.softmax(self.scale_attention, dim=0)
            weighted_features = []
            for i, features in enumerate(scale_features):
                weighted_features.append(attention_weights[i] * features)
            fused_features = torch.stack(weighted_features, dim=1).sum(dim=1)  # [B, hidden_dim]
        else:
            # Simple concatenation and fusion
            concatenated = torch.cat(scale_features, dim=-1)  # [B, len(scales) * hidden_dim]
            fused_features = self.scale_fusion(concatenated)  # [B, hidden_dim]
        
        # Generate final predictions
        rotation_pred = self.rotation_head(fused_features)  # [B, 4]
        translation_pred = self.translation_head(fused_features)  # [B, 3]
        
        # Normalize quaternion
        rotation_pred = rotation_pred / (torch.norm(rotation_pred, dim=-1, keepdim=True) + 1e-8)
        
        return {
            'rotation': rotation_pred,
            'translation': translation_pred,
            'scale_outputs': scale_outputs,
            'scale_attention_weights': torch.softmax(self.scale_attention, dim=0) if self.use_scale_weights else None
        }
    
    def _encode_visual_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode visual features for the entire sequence.
        
        Args:
            images: [B, seq_len, feature_dim] (pre-extracted features) or [B, seq_len, C, H, W] (raw images)
        
        Returns:
            visual_features: [B, seq_len, feature_dim]
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
            images_flat = images.view(-1, *images.shape[2:])  # [B*seq_len, C, H, W]
            features_flat = self.feature_encoder(images_flat)  # [B*seq_len, feature_dim]
            visual_features = features_flat.view(batch_size, seq_len, -1)  # [B, seq_len, feature_dim]
        
        return visual_features
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with multi-scale loss."""
        predictions = self(batch)
        
        # Get target poses (last frame)
        target_rotation = batch['poses'][:, -1, 3:7]  # Quaternion
        target_translation = batch['poses'][:, -1, :3]  # XYZ
        
        # Calculate losses
        rotation_loss = nn.functional.mse_loss(predictions['rotation'], target_rotation)
        translation_loss = nn.functional.mse_loss(predictions['translation'], target_translation)
        
        # Total loss
        total_loss = self.rotation_weight * rotation_loss + self.translation_weight * translation_loss
        
        # Log metrics
        self.train_rot_mae(predictions['rotation'].reshape(-1), target_rotation.reshape(-1))
        self.train_trans_mae(predictions['translation'].reshape(-1), target_translation.reshape(-1))
        
        self.log('train/rotation_loss', rotation_loss, prog_bar=True)
        self.log('train/translation_loss', translation_loss, prog_bar=True)
        self.log('train/total_loss', total_loss, prog_bar=True)
        self.log('train/rotation_mae', self.train_rot_mae, prog_bar=True)
        self.log('train/translation_mae', self.train_trans_mae, prog_bar=True)
        
        # Log scale attention weights if enabled
        if self.use_scale_weights and predictions['scale_attention_weights'] is not None:
            weights = predictions['scale_attention_weights']
            for i, seq_len in enumerate(self.sequence_lengths):
                self.log(f'train/scale_weight_{seq_len}', weights[i], prog_bar=False)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
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
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/total_loss",
            },
        }


class AdaptiveMultiScaleVIO(MultiScaleTemporalVIO):
    """
    Enhanced version with adaptive scale selection based on motion patterns.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Motion pattern classifier
        self.motion_classifier = nn.Sequential(
            nn.Linear(self.feature_dim + self.feature_dim // 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, len(self.sequence_lengths)),  # Probability for each scale
            nn.Softmax(dim=-1)
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Enhanced forward with adaptive scale selection."""
        # Get base multi-scale output
        base_output = super().forward(batch)
        
        # Classify motion pattern
        visual_features = self._encode_visual_features(batch['images'])
        imu_features = self.imu_encoder(batch['imus'])
        combined_features = torch.cat([visual_features, imu_features], dim=-1)
        
        # Use last frame features for motion classification
        motion_features = combined_features[:, -1, :]
        motion_weights = self.motion_classifier(motion_features)  # [B, num_scales]
        
        # Adaptively weight scale features
        scale_features = []
        for i, seq_len in enumerate(self.sequence_lengths):
            scale_name = f"scale_{seq_len}"
            scale_output = base_output['scale_outputs'][scale_name]
            scale_features.append(scale_output[:, -1, :])  # [B, hidden_dim]
        
        # Apply adaptive weights
        adaptive_features = torch.zeros_like(scale_features[0])  # [B, hidden_dim]
        for i, features in enumerate(scale_features):
            adaptive_features += motion_weights[:, i:i+1] * features
        
        # Generate adaptive predictions
        adaptive_rotation = self.rotation_head(adaptive_features)
        adaptive_translation = self.translation_head(adaptive_features)
        adaptive_rotation = adaptive_rotation / (torch.norm(adaptive_rotation, dim=-1, keepdim=True) + 1e-8)
        
        base_output.update({
            'adaptive_rotation': adaptive_rotation,
            'adaptive_translation': adaptive_translation,
            'motion_weights': motion_weights
        })
        
        return base_output