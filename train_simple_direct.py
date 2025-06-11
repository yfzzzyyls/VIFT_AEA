#!/usr/bin/env python3
"""
Simple direct training approach for VIFT model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from pathlib import Path

class SimpleVIFTModel(nn.Module):
    """Simple VIFT model architecture"""
    
    def __init__(self, visual_dim=512, imu_dim=256, hidden_dim=256):
        super().__init__()
        
        # Feature projection
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.imu_proj = nn.Linear(imu_dim, hidden_dim)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Output heads
        self.translation_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3)
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, visual_features, imu_features):
        # Project features
        visual_proj = self.visual_proj(visual_features)
        imu_proj = self.imu_proj(imu_features)
        
        # Fuse features
        fused = torch.cat([visual_proj, imu_proj], dim=-1)
        fused = self.fusion(fused)
        
        # Transformer processing
        encoded = self.transformer(fused)
        
        # Output predictions
        translation = self.translation_head(encoded)
        rotation = self.rotation_head(encoded)
        
        # Normalize quaternions
        rotation = F.normalize(rotation, p=2, dim=-1)
        
        return translation, rotation

class SimpleVIFTModule(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # Create model
        self.model = SimpleVIFTModel()
        
        # Loss weights (found from data analysis)
        self.trans_weight = 1.0
        self.rot_weight = 10.0  # Rotation values are smaller, need higher weight
        
    def forward(self, batch):
        visual = batch['visual_features']
        imu = batch['imu_features']
        return self.model(visual, imu)
    
    def compute_loss(self, pred_trans, pred_rot, gt_trans, gt_rot):
        """Compute balanced loss"""
        # Translation loss (Huber for robustness)
        trans_loss = F.huber_loss(pred_trans, gt_trans, delta=2.0)
        
        # Rotation loss (quaternion distance)
        # Ensure quaternions are normalized
        pred_rot_norm = F.normalize(pred_rot, p=2, dim=-1)
        gt_rot_norm = F.normalize(gt_rot, p=2, dim=-1)
        
        # Quaternion distance: 1 - |<q1, q2>|
        dot_product = torch.abs(torch.sum(pred_rot_norm * gt_rot_norm, dim=-1))
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        rot_loss = torch.mean(1.0 - dot_product)
        
        return trans_loss, rot_loss
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        pred_trans, pred_rot = self.forward(batch)
        
        # Ground truth
        gt_poses = batch['poses']
        gt_trans = gt_poses[:, :, :3]
        gt_rot = gt_poses[:, :, 3:]
        
        # Compute losses
        trans_loss, rot_loss = self.compute_loss(pred_trans, pred_rot, gt_trans, gt_rot)
        total_loss = self.trans_weight * trans_loss + self.rot_weight * rot_loss
        
        # Metrics
        with torch.no_grad():
            trans_mae = torch.mean(torch.abs(pred_trans - gt_trans))
            trans_rmse = torch.sqrt(torch.mean((pred_trans - gt_trans)**2))
            
            # Translation magnitude
            pred_mag = torch.mean(torch.norm(pred_trans, dim=-1))
            gt_mag = torch.mean(torch.norm(gt_trans, dim=-1))
            
        # Log everything
        self.log('train/trans_loss', trans_loss)
        self.log('train/rot_loss', rot_loss)
        self.log('train/total_loss', total_loss)
        self.log('train/trans_mae_cm', trans_mae)
        self.log('train/trans_rmse_cm', trans_rmse)
        self.log('train/pred_magnitude', pred_mag)
        self.log('train/gt_magnitude', gt_mag)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        pred_trans, pred_rot = self.forward(batch)
        
        # Ground truth
        gt_poses = batch['poses']
        gt_trans = gt_poses[:, :, :3]
        gt_rot = gt_poses[:, :, 3:]
        
        # Compute losses
        trans_loss, rot_loss = self.compute_loss(pred_trans, pred_rot, gt_trans, gt_rot)
        total_loss = self.trans_weight * trans_loss + self.rot_weight * rot_loss
        
        # Metrics
        trans_mae = torch.mean(torch.abs(pred_trans - gt_trans))
        trans_rmse = torch.sqrt(torch.mean((pred_trans - gt_trans)**2))
        
        # Log
        self.log('val/trans_loss', trans_loss)
        self.log('val/rot_loss', rot_loss)
        self.log('val/total_loss', total_loss)
        self.log('val/trans_mae_cm', trans_mae)
        self.log('val/trans_rmse_cm', trans_rmse)
        
        return total_loss
    
    def configure_optimizers(self):
        # Use AdamW with warm restart
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Initial restart interval
            T_mult=2,  # Multiply interval by 2 after each restart
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

# Dataset
class SeparateFeatureDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.samples = []
        
        for visual_file in sorted(self.data_dir.glob("*_visual.npy")):
            idx = visual_file.stem.split("_")[0]
            imu_file = self.data_dir / f"{idx}_imu.npy"
            gt_file = self.data_dir / f"{idx}_gt.npy"
            
            if imu_file.exists() and gt_file.exists():
                self.samples.append({
                    'visual': visual_file,
                    'imu': imu_file,
                    'gt': gt_file
                })
        
        print(f"Found {len(self.samples)} samples in {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        visual = torch.from_numpy(np.load(sample['visual'])).float()
        imu = torch.from_numpy(np.load(sample['imu'])).float()
        poses = torch.from_numpy(np.load(sample['gt'])).float()
        
        return {
            'visual_features': visual,
            'imu_features': imu,
            'poses': poses
        }

def collate_fn(batch):
    return {
        'visual_features': torch.stack([b['visual_features'] for b in batch]),
        'imu_features': torch.stack([b['imu_features'] for b in batch]),
        'poses': torch.stack([b['poses'] for b in batch])
    }

if __name__ == "__main__":
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create datasets
    train_dataset = SeparateFeatureDataset('aria_latent_data_cm/train')
    val_dataset = SeparateFeatureDataset('aria_latent_data_cm/val')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    model = SimpleVIFTModule()
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=50,
        callbacks=[
            ModelCheckpoint(
                dirpath='simple_direct_model',
                filename='epoch_{epoch:03d}_mae_{val/trans_mae_cm:.4f}',
                monitor='val/trans_mae_cm',
                mode='min',
                save_top_k=5,
                save_last=True
            ),
            RichProgressBar()
        ],
        log_every_n_steps=10,
        val_check_interval=0.5,
        gradient_clip_val=1.0,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1  # Use single GPU to avoid DDP issues
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)