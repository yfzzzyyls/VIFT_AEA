"""
VIFT model modified to output quaternions - Correctly following the paper architecture
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import lightning as L
from torchmetrics import MeanAbsoluteError
import numpy as np
import math


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as described in the VIFT paper"""
    
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        
    def forward(self, seq_length):
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * 
                            -(math.log(10000.0) / self.embedding_dim))
        
        pos_embedding = torch.zeros(seq_length, self.embedding_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        
        return pos_embedding.unsqueeze(0)  # [1, seq_length, embedding_dim]


class VIFTQuaternionCorrect(L.LightningModule):
    """
    VIFT model that correctly follows the paper architecture
    Key differences:
    1. Uses causal mask in transformer
    2. Applies 2-layer MLP after transformer (not fc2 from PoseTransformer)
    3. Outputs N poses for N+1 inputs (sliding window)
    4. Uses quaternions instead of Euler angles
    """
    
    def __init__(
        self,
        input_dim: int = 768,          # 512 visual + 256 IMU
        embedding_dim: int = 128,      # Hidden dimension
        num_layers: int = 6,           # Number of transformer layers
        nhead: int = 8,                # Number of attention heads
        dim_feedforward: int = 512,    # Feed-forward dimension in transformer
        dropout: float = 0.1,
        window_size: int = 11,         # N+1 in the paper (10 transitions + 1)
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        rotation_weight: float = 10.0,
        translation_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Linear projection for visual-inertial features (Section 3.2)
        self.linear_projection = nn.Linear(input_dim, embedding_dim)
        
        # Sinusoidal positional encoding (Section 3.2)
        self.positional_encoding = SinusoidalPositionalEncoding(embedding_dim)
        
        # Transformer encoder with causal mask (Section 3.2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',  # Standard transformer uses ReLU
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 2-layer MLP for pose output (Section 3.2)
        # Changed to output 7 values (3 translation + 4 quaternion)
        self.pose_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 7)  # 7DoF output
        )
        
        # Initialize output layer
        self._initialize_weights()
        
        # Metrics
        self.train_trans_mae = MeanAbsoluteError()
        self.val_trans_mae = MeanAbsoluteError()
        
    def _initialize_weights(self):
        """Initialize weights following best practices"""
        # Initialize linear projection
        nn.init.xavier_uniform_(self.linear_projection.weight)
        nn.init.zeros_(self.linear_projection.bias)
        
        # Initialize MLP
        for layer in self.pose_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Set quaternion bias to identity [0, 0, 0, 1]
        self.pose_mlp[-1].bias.data[3:6] = 0.0  # qx, qy, qz
        self.pose_mlp[-1].bias.data[6] = 1.0    # qw
    
    def generate_causal_mask(self, seq_length, device):
        """Generate causal mask for transformer (lower triangular)"""
        mask = torch.triu(
            torch.full((seq_length, seq_length), float('-inf'), device=device),
            diagonal=1
        )
        return mask
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass following VIFT paper exactly
        Input: N+1 measurements
        Output: N relative poses
        """
        # Extract features
        if 'visual_features' in batch and 'imu_features' in batch:
            visual_features = batch['visual_features']
            imu_features = batch['imu_features']
            B, seq_length, _ = visual_features.shape
            
            # Concatenate visual and IMU features
            combined_features = torch.cat([visual_features, imu_features], dim=-1)
        else:
            combined_features = batch['images']
            B, seq_length, _ = combined_features.shape
        
        # Step 1: Linear projection (Section 3.2)
        x = self.linear_projection(combined_features)  # [B, seq_length, embedding_dim]
        
        # Step 2: Add sinusoidal positional encoding (Section 3.2)
        pos_encoding = self.positional_encoding(seq_length).to(x.device)
        x = x + pos_encoding
        
        # Step 3: Pass through transformer with causal mask (Section 3.2)
        causal_mask = self.generate_causal_mask(seq_length, x.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        
        # Step 4: Apply 2-layer MLP to get pose outputs (Section 3.2)
        # "After transformer layers, we apply 2-layer MLP to every feature to obtain pose output"
        poses = self.pose_mlp(x)  # [B, seq_length, 7]
        
        # Step 5: Extract N poses from N+1 inputs
        # "Ultimately, we obtain N relative poses for N + 1 input images"
        # We use poses from positions 1 to N (0-indexed: 1 to seq_length-1)
        # This represents transitions: (0->1), (1->2), ..., (N-1->N)
        relative_poses = poses[:, 1:, :]  # [B, seq_length-1, 7]
        
        # Split translation and quaternion
        translation = relative_poses[:, :, :3]
        quaternion = relative_poses[:, :, 3:7]
        
        # Normalize quaternions
        quaternion = nn.functional.normalize(quaternion, p=2, dim=-1)
        
        return {
            'translation': translation,
            'rotation': quaternion,
            'poses': relative_poses  # Full 7DoF output
        }
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                     batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss following VIFT paper
        For quaternions, we use geodesic distance instead of RPMG
        """
        # Get predictions
        pred_translation = predictions['translation']
        pred_rotation = predictions['rotation']
        
        # Get ground truth - already contains N transitions for N+1 inputs
        # Ground truth should be [B, N, 7] where N = seq_length - 1
        gt_poses = batch['poses']
        if gt_poses.shape[1] == pred_translation.shape[1] + 1:
            # If GT has N+1 frames, extract N transitions
            gt_translation = gt_poses[:, 1:, :3]
            gt_rotation = gt_poses[:, 1:, 3:7]
        else:
            # GT already contains transitions
            gt_translation = gt_poses[:, :, :3]
            gt_rotation = gt_poses[:, :, 3:7]
        
        # Flatten for loss computation
        pred_trans_flat = pred_translation.reshape(-1, 3)
        gt_trans_flat = gt_translation.reshape(-1, 3)
        pred_rot_flat = pred_rotation.reshape(-1, 4)
        gt_rot_flat = gt_rotation.reshape(-1, 4)
        
        # Translation loss (L1 as mentioned in paper)
        trans_loss = nn.functional.l1_loss(pred_trans_flat, gt_trans_flat)
        
        # Rotation loss - quaternion geodesic distance
        # Since we use quaternions instead of rotation matrices, 
        # we compute geodesic distance directly
        pred_rot_norm = nn.functional.normalize(pred_rot_flat, p=2, dim=-1)
        gt_rot_norm = nn.functional.normalize(gt_rot_flat, p=2, dim=-1)
        
        # Handle quaternion double cover
        dot = (pred_rot_norm * gt_rot_norm).sum(dim=-1)
        dot = torch.abs(dot).clamp(-1.0, 1.0)
        
        # Geodesic distance
        rot_loss = torch.mean(2.0 * torch.acos(dot))
        
        # Apply weights (α in Equation 4)
        weighted_trans_loss = trans_loss * self.hparams.translation_weight
        weighted_rot_loss = rot_loss * self.hparams.rotation_weight
        
        total_loss = weighted_trans_loss + weighted_rot_loss
        
        return {
            'total_loss': total_loss,
            'translation_loss': weighted_trans_loss,
            'rotation_loss': weighted_rot_loss,
            'trans_loss_raw': trans_loss,
            'rot_loss_raw': rot_loss
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        loss_dict = self.compute_loss(predictions, batch)
        
        # Log losses
        self.log('train/total_loss', loss_dict['total_loss'], prog_bar=True)
        self.log('train/trans_loss', loss_dict['trans_loss_raw'])
        self.log('train/rot_loss', loss_dict['rot_loss_raw'])
        
        # Log MAE metrics
        with torch.no_grad():
            self.train_trans_mae(predictions['translation'].reshape(-1, 3),
                               batch['poses'][:, 1:, :3].reshape(-1, 3))
            self.log('train/trans_mae', self.train_trans_mae, prog_bar=True)
        
        return loss_dict['total_loss']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        loss_dict = self.compute_loss(predictions, batch)
        
        # Log losses
        self.log('val/total_loss', loss_dict['total_loss'], prog_bar=True)
        self.log('val/trans_loss', loss_dict['trans_loss_raw'])
        self.log('val/rot_loss', loss_dict['rot_loss_raw'])
        
        # Log MAE metrics
        with torch.no_grad():
            self.val_trans_mae(predictions['translation'].reshape(-1, 3),
                             batch['poses'][:, 1:, :3].reshape(-1, 3))
            self.log('val/trans_mae', self.val_trans_mae, prog_bar=True)
        
        return loss_dict['total_loss']
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }