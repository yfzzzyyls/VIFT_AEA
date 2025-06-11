import torch
import torch.nn as nn
import lightning as L
from typing import Dict, Optional
from torchmetrics import MeanAbsoluteError
import numpy as np
from scipy.spatial.transform import Rotation

from .components.pose_transformer_quaternion import PoseTransformerQuaternion
from ..metrics.arvr_loss_wrapper import ARVRLossWrapper


class VIFTQuaternionModelV4(L.LightningModule):
    """
    VIFT model with quaternion output and separate learning rates for translation/rotation.
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
        rotation_lr_multiplier: float = 10.0,  # Rotation LR = base_lr * this
        weight_decay: float = 1e-5,
        rotation_weight: float = 1.0,
        translation_weight: float = 1.0,
        optimizer_type: str = 'adamw',
        scheduler_type: str = 'cosine',
        warmup_steps: int = 500
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Create the quaternion transformer
        self.model = PoseTransformerQuaternion(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Loss function
        self.loss_fn = ARVRLossWrapper(use_log_scale=False, use_weighted_loss=False)
        
        # Metrics
        self.train_trans_mae = MeanAbsoluteError()
        self.val_trans_mae = MeanAbsoluteError()
        self.train_rot_mae = MeanAbsoluteError()
        self.val_rot_mae = MeanAbsoluteError()
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass"""
        # Extract separate features
        visual_features = batch.get('visual_features', None)
        imu_features = batch.get('imu_features', None)
        
        # Handle backward compatibility
        if visual_features is None or imu_features is None:
            features = batch['images']
            visual_features = features[..., :512]
            imu_features = features[..., 512:]
        
        # Concatenate visual and IMU features
        features = torch.cat([visual_features, imu_features], dim=-1)
        
        # Create dummy batch tuple for compatibility
        batch_tuple = (features, None, None)
        
        # Forward through model
        predictions = self.model(batch_tuple, None)
        
        return predictions
    
    def compute_quaternion_angle_error(self, pred_quat, target_quat):
        """Compute angular error between predicted and target quaternions in degrees"""
        # Normalize quaternions
        pred_quat = pred_quat / (torch.norm(pred_quat, dim=-1, keepdim=True) + 1e-8)
        target_quat = target_quat / (torch.norm(target_quat, dim=-1, keepdim=True) + 1e-8)
        
        # Compute dot product
        dot = torch.sum(pred_quat * target_quat, dim=-1)
        dot = torch.clamp(dot, -1.0, 1.0)
        
        # Angular error in radians
        angle_rad = 2 * torch.acos(torch.abs(dot))
        
        # Convert to degrees
        angle_deg = torch.rad2deg(angle_rad)
        
        return angle_deg.mean()
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Get predictions
        predictions = self(batch)
        
        # Get ground truth
        target = batch.get('relative_poses', batch.get('poses'))
        
        # Reshape for loss computation
        pred_flat = predictions.reshape(-1, 7)
        target_flat = target.reshape(-1, 7)
        
        # Split translation and rotation
        pred_trans = pred_flat[:, :3]
        pred_rot = pred_flat[:, 3:]
        target_trans = target_flat[:, :3]
        target_rot = target_flat[:, 3:]
        
        # Compute losses
        loss_dict = self.loss_fn(
            pred_rotation=pred_rot,
            target_rotation=target_rot,
            pred_translation=pred_trans,
            target_translation=target_trans
        )
        
        # Weighted total loss
        loss_dict['rotation_loss'] *= self.hparams.rotation_weight
        loss_dict['translation_loss'] *= self.hparams.translation_weight
        
        total_loss = loss_dict['total_loss']
        
        # Log losses
        self.log('train/total_loss', total_loss, prog_bar=True)
        self.log('train/trans_loss', loss_dict['translation_loss'])
        self.log('train/rot_loss', loss_dict['rotation_loss'])
        
        # Update metrics
        with torch.no_grad():
            self.train_trans_mae(pred_trans.contiguous(), target_trans.contiguous())
            angle_error = self.compute_quaternion_angle_error(pred_rot, target_rot)
            self.train_rot_mae.update(angle_error, torch.zeros_like(angle_error))
            
            self.log('train/trans_mae_cm', self.train_trans_mae, prog_bar=True)
            self.log('train/rot_mae_deg', self.train_rot_mae, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Get predictions
        predictions = self(batch)
        
        # Get ground truth
        target = batch.get('relative_poses', batch.get('poses'))
        
        # Compute loss (same as training)
        pred_flat = predictions.reshape(-1, 7)
        target_flat = target.reshape(-1, 7)
        
        pred_trans = pred_flat[:, :3]
        pred_rot = pred_flat[:, 3:]
        target_trans = target_flat[:, :3]
        target_rot = target_flat[:, 3:]
        
        loss_dict = self.loss_fn(
            pred_rotation=pred_rot,
            target_rotation=target_rot,
            pred_translation=pred_trans,
            target_translation=target_trans
        )
        
        loss_dict['rotation_loss'] *= self.hparams.rotation_weight
        loss_dict['translation_loss'] *= self.hparams.translation_weight
        
        total_loss = loss_dict['total_loss']
        
        # Log losses
        self.log('val/total_loss', total_loss, prog_bar=True)
        self.log('val/trans_loss', loss_dict['translation_loss'])
        self.log('val/rot_loss', loss_dict['rotation_loss'])
        
        # Update metrics
        with torch.no_grad():
            self.val_trans_mae(pred_trans.contiguous(), target_trans.contiguous())
            angle_error = self.compute_quaternion_angle_error(pred_rot, target_rot)
            self.val_rot_mae.update(angle_error, torch.zeros_like(angle_error))
            
            self.log('val/trans_mae_cm', self.val_trans_mae, prog_bar=True)
            self.log('val/rot_mae_deg', self.val_rot_mae, prog_bar=True)
        
        return total_loss
    
    def configure_optimizers(self):
        # Create parameter groups with different learning rates
        
        # Identify rotation-related parameters
        rotation_params = []
        translation_params = []
        
        # The last layer outputs both translation and rotation
        # We'll use a heuristic: parameters that affect output indices 3-6 are rotation-related
        for name, param in self.named_parameters():
            if 'fc2' in name and ('weight' in name or 'bias' in name):
                # This is the output layer - split it conceptually
                # In practice, we'll apply higher LR to encourage rotation learning
                rotation_params.append(param)
            else:
                translation_params.append(param)
        
        # Create parameter groups
        param_groups = [
            {'params': translation_params, 'lr': self.hparams.learning_rate},
            {'params': rotation_params, 'lr': self.hparams.learning_rate * self.hparams.rotation_lr_multiplier}
        ]
        
        # Create optimizer with parameter groups
        if self.hparams.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                param_groups,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.999)
            )
        else:  # sgd
            optimizer = torch.optim.SGD(
                param_groups,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9
            )
        
        # Scheduler
        if self.hparams.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs
            )
        elif self.hparams.scheduler_type == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[self.hparams.learning_rate, self.hparams.learning_rate * self.hparams.rotation_lr_multiplier],
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                anneal_strategy='cos'
            )
        else:  # constant
            return optimizer
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/total_loss',
                'interval': 'epoch' if self.hparams.scheduler_type != 'onecycle' else 'step'
            }
        }