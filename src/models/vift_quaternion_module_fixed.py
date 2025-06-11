"""
Fixed VIFT Quaternion Module with bias correction and improved training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as L
from typing import Dict, Any, Optional, Tuple
import numpy as np

from src.models.components.pose_transformer_quaternion import PoseTransformerQuaternion
from src.metrics.arvr_loss_wrapper import ARVRLossWrapper


class VIFTQuaternionModuleFixed(L.LightningModule):
    """Fixed version with bias correction and improved loss"""
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        rotation_weight: float = 100.0,  # Increased rotation weight
        bias_weight: float = 0.1,  # New: penalize systematic bias
        variance_weight: float = 0.1,  # New: encourage output variance
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = PoseTransformerQuaternion(
            input_dim=input_dim,
            embedding_dim=hidden_dim,  # PoseTransformerQuaternion uses embedding_dim, not hidden_dim
            num_layers=num_layers,
            nhead=num_heads,  # PoseTransformerQuaternion uses nhead, not num_heads
            dropout=dropout
        )
        
        # Loss function - no log scale for clearer training signal
        self.loss_fn = ARVRLossWrapper(
            use_log_scale=False,
            use_weighted_loss=False
        )
        
        # Track running statistics for bias correction
        self.register_buffer('running_mean_trans', torch.zeros(3))
        self.register_buffer('running_mean_rot', torch.zeros(4))
        self.register_buffer('update_count', torch.tensor(0))
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with bias correction"""
        # Concatenate visual and IMU features
        visual_features = batch['visual_features']  # [B, T, 512]
        imu_features = batch['imu_features']       # [B, T, 512]
        combined_features = torch.cat([visual_features, imu_features], dim=-1)  # [B, T, 1024]
        
        # Get raw predictions from model
        predictions = self.model(combined_features, batch.get('poses'))  # [B, T, 7]
        
        # Split predictions into translation and rotation
        translation = predictions[:, :, :3]  # [B, T, 3]
        rotation = predictions[:, :, 3:]     # [B, T, 4]
        
        # Apply bias correction during inference
        if not self.training and self.update_count > 100:
            translation = translation - self.running_mean_trans.unsqueeze(0).unsqueeze(0)
        
        outputs = {
            'translation': translation,
            'rotation': rotation
        }
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute loss with additional regularization terms"""
        # Get predictions and ground truth
        pred_trans = outputs['translation']  # [B, T, 3]
        pred_rot = outputs['rotation']      # [B, T, 4]
        gt_poses = batch['poses']           # [B, T, 7]
        gt_trans = gt_poses[:, :, :3]
        gt_rot = gt_poses[:, :, 3:]
        
        # Flatten for loss computation
        B, T = pred_trans.shape[:2]
        pred_trans_flat = pred_trans.view(B * T, 3)
        pred_rot_flat = pred_rot.view(B * T, 4)
        gt_trans_flat = gt_trans.view(B * T, 3)
        gt_rot_flat = gt_rot.view(B * T, 4)
        
        # Basic translation and rotation losses
        loss_dict = self.loss_fn(pred_rot_flat, gt_rot_flat, pred_trans_flat, gt_trans_flat)
        trans_loss = loss_dict['translation_loss']
        rot_loss = loss_dict['rotation_loss']
        
        # 1. Bias regularization - penalize systematic bias
        trans_bias = torch.mean(pred_trans, dim=(0, 1))  # Average over batch and time
        bias_loss = torch.norm(trans_bias)
        
        # 2. Variance regularization - encourage diverse predictions
        trans_var = torch.var(pred_trans, dim=1).mean()  # Variance over time
        rot_var = torch.var(pred_rot, dim=1).mean()
        # Negative variance loss to maximize variance
        variance_loss = -torch.log(trans_var + 1e-6) - 0.1 * torch.log(rot_var + 1e-6)
        
        # 3. Motion consistency - predictions should vary with input
        if pred_trans.shape[1] > 1:
            trans_diff = torch.diff(pred_trans, dim=1)
            motion_var = torch.var(trans_diff, dim=0).mean()
            consistency_loss = -torch.log(motion_var + 1e-6)
        else:
            consistency_loss = torch.tensor(0.0, device=pred_trans.device)
        
        # Total loss with weights
        total_loss = (
            trans_loss + 
            self.hparams.rotation_weight * rot_loss +
            self.hparams.bias_weight * bias_loss +
            self.hparams.variance_weight * variance_loss +
            0.1 * consistency_loss
        )
        
        # Update running statistics
        if self.training:
            with torch.no_grad():
                self.update_count += 1
                alpha = 0.01  # Exponential moving average
                batch_mean_trans = torch.mean(pred_trans.detach(), dim=(0, 1))
                batch_mean_rot = torch.mean(pred_rot.detach(), dim=(0, 1))
                self.running_mean_trans = (1 - alpha) * self.running_mean_trans + alpha * batch_mean_trans
                self.running_mean_rot = (1 - alpha) * self.running_mean_rot + alpha * batch_mean_rot
        
        return {
            'total_loss': total_loss,
            'trans_loss': trans_loss,
            'rot_loss': rot_loss,
            'bias_loss': bias_loss,
            'variance_loss': variance_loss,
            'consistency_loss': consistency_loss
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Apply data augmentation
        batch = self.augment_batch(batch)
        
        outputs = self.forward(batch)
        losses = self.compute_loss(outputs, batch)
        
        # Log all losses
        for name, loss in losses.items():
            self.log(f'train/{name}', loss, on_step=True, on_epoch=True, prog_bar=(name == 'total_loss'))
        
        # Compute and log metrics
        with torch.no_grad():
            pred_trans = outputs['translation']
            gt_trans = batch['poses'][:, :, :3]
            trans_mae = torch.mean(torch.abs(pred_trans - gt_trans))
            
            # Log bias magnitude
            trans_bias = torch.mean(pred_trans, dim=(0, 1))
            bias_magnitude = torch.norm(trans_bias)
            
            # Log prediction variance
            trans_var = torch.var(pred_trans, dim=1).mean()
            
            self.log('train/trans_mae_cm', trans_mae, on_step=True, on_epoch=True)
            self.log('train/bias_magnitude', bias_magnitude, on_step=True, on_epoch=True)
            self.log('train/pred_variance', trans_var, on_step=True, on_epoch=True)
        
        return losses['total_loss']
    
    def augment_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply data augmentation to prevent bias"""
        if not self.training:
            return batch
            
        # Random rotation augmentation
        if torch.rand(1).item() < 0.5:
            batch = self.apply_random_rotation(batch)
        
        # Random scaling augmentation
        if torch.rand(1).item() < 0.3:
            scale = torch.rand(1).item() * 0.4 + 0.8  # 0.8 to 1.2
            batch['poses'][:, :, :3] *= scale
            
        return batch
    
    def apply_random_rotation(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply random rotation to entire sequence"""
        # Generate random rotation quaternion
        angle = torch.rand(1).item() * 2 * np.pi
        axis = F.normalize(torch.randn(3), dim=0)
        q = torch.cat([
            axis * torch.sin(angle / 2),
            torch.tensor([torch.cos(angle / 2)])
        ])
        
        # Apply rotation to translations and rotations
        # This is simplified - full implementation would properly transform quaternions
        # For now, just add noise to prevent overfitting to specific directions
        noise = torch.randn_like(batch['poses'][:, :, :3]) * 0.1
        batch['poses'][:, :, :3] = batch['poses'][:, :, :3] + noise
        
        return batch
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.forward(batch)
        losses = self.compute_loss(outputs, batch)
        
        # Log all losses
        for name, loss in losses.items():
            self.log(f'val/{name}', loss, on_epoch=True, prog_bar=(name == 'total_loss'))
        
        # Compute and log metrics
        with torch.no_grad():
            pred_trans = outputs['translation']
            gt_trans = batch['poses'][:, :, :3]
            trans_mae = torch.mean(torch.abs(pred_trans - gt_trans))
            
            # Check for bias
            trans_bias = torch.mean(pred_trans, dim=(0, 1))
            bias_magnitude = torch.norm(trans_bias)
            
            self.log('val/trans_mae_cm', trans_mae, on_epoch=True)
            self.log('val/bias_magnitude', bias_magnitude, on_epoch=True)
        
        return losses['total_loss']
    
    def configure_optimizers(self):
        # Use AdamW with gradient clipping
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.98)
        )
        
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / self.hparams.warmup_steps
            else:
                progress = (step - self.hparams.warmup_steps) / (self.hparams.max_steps - self.hparams.warmup_steps)
                return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }