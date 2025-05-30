"""
Simplified Aria Data Module for AR/VR Training
Works with the current latent data format.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L
from pathlib import Path
from typing import Optional
import random


class SimpleAriaDataset(Dataset):
    """Simple dataset for pre-processed Aria latent data."""
    
    def __init__(self, data_dir: str, max_samples: Optional[int] = None):
        self.data_dir = Path(data_dir)
        
        # Find all feature files
        self.feature_files = sorted(list(self.data_dir.glob("*.npy")))
        # Filter to only get the main feature files (not _gt, _rot, _w)
        self.feature_files = [f for f in self.feature_files if not any(suffix in f.name for suffix in ['_gt', '_rot', '_w'])]
        
        if max_samples:
            self.feature_files = self.feature_files[:max_samples]
        
        print(f"Found {len(self.feature_files)} samples in {data_dir}")
    
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        feature_file = self.feature_files[idx]
        base_name = feature_file.stem
        
        # Load features [seq_len, 768]
        features = torch.from_numpy(np.load(feature_file)).float()
        
        # Load ground truth poses [seq_len, 6] 
        gt_file = self.data_dir / f"{base_name}_gt.npy"
        poses_6d = torch.from_numpy(np.load(gt_file)).float()
        
        # Convert to 7D poses (x,y,z,qx,qy,qz,qw) for compatibility
        seq_len = poses_6d.shape[0]
        poses_7d = torch.zeros(seq_len, 7)
        poses_7d[:, :3] = poses_6d[:, :3]  # Translation
        poses_7d[:, 3:6] = poses_6d[:, 3:6]  # Rotation (Euler or quaternion xyz)
        poses_7d[:, 6] = 1.0  # Set w component to 1 for quaternion (simplified)
        
        # Create IMU data (simulate for now since we don't have it in this format)
        imu_data = torch.randn(seq_len, 6) * 0.1  # Small random IMU data
        
        return {
            'images': features,  # [seq_len, 768] - pre-extracted visual features
            'imus': imu_data,    # [seq_len, 6] - simulated IMU data
            'poses': poses_7d    # [seq_len, 7] - ground truth poses
        }


class SimpleAriaDataModule(L.LightningDataModule):
    """
    Simple data module for AR/VR training with existing latent data.
    """
    
    def __init__(
        self,
        train_data_dir: str = "aria_latent_data/aria_latent_data_80_10_10/train",
        val_data_dir: str = "aria_latent_data/aria_latent_data_80_10_10/val", 
        test_data_dir: str = "aria_latent_data/test",
        batch_size: int = 16,
        num_workers: int = 8,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        
        # Check if val directory exists, otherwise use train for validation
        if not Path(val_data_dir).exists():
            print(f"⚠️ Validation directory {val_data_dir} not found, using train data for validation")
            self.val_data_dir = train_data_dir
            self.max_val_samples = 100  # Use subset for validation
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = SimpleAriaDataset(self.train_data_dir, self.max_train_samples)
            self.val_dataset = SimpleAriaDataset(self.val_data_dir, self.max_val_samples)
        
        if stage == "test" or stage is None:
            self.test_dataset = SimpleAriaDataset(self.test_data_dir)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )