"""
VIFT model modified to output quaternions instead of Euler angles
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import lightning as L
from torchmetrics import MeanAbsoluteError
import numpy as np

from .components.pose_transformer import PoseTransformer
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


class VIFTQuaternionModel(L.LightningModule):
    """
    VIFT model that outputs quaternions (7 values: tx, ty, tz, qx, qy, qz, qw)
    instead of Euler angles (6 values: tx, ty, tz, rx, ry, rz)
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        embedding_dim: int = 128,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        rotation_weight: float = 1.0,
        translation_weight: float = 1.0,
        optimizer_type: str = 'adamw',
        scheduler_type: str = 'onecycle',
        warmup_steps: int = 500
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Create the original VIFT model backbone
        self.pose_transformer = PoseTransformer(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Replace the output layer to output 7 values (translation + quaternion)
        # Original outputs 6 (translation + euler), we need 7 (translation + quaternion)
        self.output_projection = nn.Linear(embedding_dim, 7)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        # Initialize quaternion part to identity [0, 0, 0, 1]
        self.output_projection.bias.data[3:6] = 0.0  # qx, qy, qz
        self.output_projection.bias.data[6] = 1.0    # qw
        
        # Loss function
        self.arvr_loss = ARVRLossWrapper(use_log_scale=False, use_weighted_loss=False)
        
        # Metrics
        self.train_rot_mae = QuaternionMAE()
        self.train_trans_mae = MeanAbsoluteError()
        self.val_rot_mae = QuaternionMAE()
        self.val_trans_mae = MeanAbsoluteError()
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass that outputs quaternions
        """
        # Extract features (handle both separate and combined features)
        if 'visual_features' in batch and 'imu_features' in batch:
            visual_features = batch['visual_features']
            imu_features = batch['imu_features']
            B, seq_len, _ = visual_features.shape
            combined_features = torch.cat([visual_features, imu_features], dim=-1)
        else:
            combined_features = batch['images']
            B, seq_len, _ = combined_features.shape
        
        # Create the batch tuple expected by VIFT forward method
        batch_tuple = (combined_features, None, None)
        
        # Use VIFT's forward but intercept before final projection
        # The PoseTransformer returns [B, seq_len-1, 6] for transitions
        # We need to modify this behavior
        
        # Get embeddings from pose transformer
        x = combined_features
        
        # Apply input projection
        x = self.pose_transformer.input_fc(x)  # [B, seq_len, embedding_dim]
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.pose_transformer.position_encoder(x, positions)
        
        # Pass through transformer
        x = x.permute(1, 0, 2)  # [seq_len, batch, embedding_dim]
        x = self.pose_transformer.transformer(x)
        x = x.permute(1, 0, 2)  # [batch, seq_len, embedding_dim]
        
        # For transitions, we need seq_len-1 outputs
        # Take differences between consecutive frames
        x_transitions = x[:, 1:] - x[:, :-1]  # [B, seq_len-1, embedding_dim]
        
        # Apply our quaternion output projection
        output = self.output_projection(x_transitions)  # [B, seq_len-1, 7]
        
        # Split translation and quaternion
        translation = output[:, :, :3]
        quaternion = output[:, :, 3:7]
        
        # Normalize quaternions
        quaternion = quaternion / (torch.norm(quaternion, dim=-1, keepdim=True) + 1e-8)
        
        return {
            'translation': translation,
            'rotation': quaternion,
            'full_output': output
        }
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                     batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute loss using quaternion predictions"""
        # Target poses
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
        
        # Log losses
        self.log('train/total_loss', total_loss, prog_bar=True)
        for key, value in loss_dict.items():
            if key != 'total_loss':
                self.log(f'train/{key}', value)
        
        # Update metrics
        with torch.no_grad():
            pred_trans = predictions['translation'].reshape(-1, 3).contiguous()
            target_trans = batch['poses'][:, :, :3].reshape(-1, 3).contiguous()
            pred_rot = predictions['rotation'].reshape(-1, 4).contiguous()
            target_rot = batch['poses'][:, :, 3:7].reshape(-1, 4).contiguous()
            
            self.train_trans_mae(pred_trans, target_trans)
            rot_mae = self.train_rot_mae(pred_rot, target_rot)
            
            self.log('train/trans_mae', self.train_trans_mae, prog_bar=True)
            self.log('train/rot_mae_rad', rot_mae, prog_bar=True)
            self.log('train/rot_mae_deg', torch.rad2deg(rot_mae), prog_bar=False)
        
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
        
        # Update metrics
        with torch.no_grad():
            pred_trans = predictions['translation'].reshape(-1, 3).contiguous()
            target_trans = batch['poses'][:, :, :3].reshape(-1, 3).contiguous()
            pred_rot = predictions['rotation'].reshape(-1, 4).contiguous()
            target_rot = batch['poses'][:, :, 3:7].reshape(-1, 4).contiguous()
            
            self.val_trans_mae(pred_trans, target_trans)
            rot_mae = self.val_rot_mae(pred_rot, target_rot)
            
            self.log('val/trans_mae', self.val_trans_mae, prog_bar=True)
            self.log('val/rot_mae_rad', rot_mae, prog_bar=True)
            self.log('val/rot_mae_deg', torch.rad2deg(rot_mae), prog_bar=False)
        
        return total_loss
    
    def configure_optimizers(self):
        # Select optimizer
        if self.hparams.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-7
            )
        elif self.hparams.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9
            )
        else:  # adamw
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-7
            )
        
        # Select scheduler
        if self.hparams.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.estimated_stepping_batches,
                eta_min=1e-6
            )
        elif self.hparams.scheduler_type == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=self.trainer.estimated_stepping_batches
            )
        elif self.hparams.scheduler_type == 'constant':
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1.0
            )
        else:  # onecycle
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate * 10,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=10,
                final_div_factor=100
            )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }