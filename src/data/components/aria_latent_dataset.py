import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class AriaLatentDataset(Dataset):
    """Dataset for loading separate visual and IMU latent features."""
    
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        
        # Get all unique sample indices
        all_files = list(self.root_dir.glob("*_visual.npy"))
        self.indices = sorted([int(f.stem.split('_')[0]) for f in all_files])
        
        # Verify all files exist
        for idx in self.indices:
            assert (self.root_dir / f"{idx}_visual.npy").exists(), f"Missing visual file for {idx}"
            assert (self.root_dir / f"{idx}_imu.npy").exists(), f"Missing IMU file for {idx}"
            assert (self.root_dir / f"{idx}_gt.npy").exists(), f"Missing GT file for {idx}"
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get actual file index
        file_idx = self.indices[idx]
        
        # Load features
        visual_features = np.load(self.root_dir / f"{file_idx}_visual.npy")  # [10, 512]
        imu_features = np.load(self.root_dir / f"{file_idx}_imu.npy")        # [10, 256]
        relative_poses = np.load(self.root_dir / f"{file_idx}_gt.npy")       # [10, 7]
        
        # Convert to tensors
        visual_features = torch.from_numpy(visual_features).float()
        imu_features = torch.from_numpy(imu_features).float()
        relative_poses = torch.from_numpy(relative_poses).float()
        
        # NOTE: GT files already contain poses in centimeters, no scaling needed!
        
        batch = {
            'visual_features': visual_features,
            'imu_features': imu_features,
            'poses': relative_poses  # Changed key to match training dataset
        }
        
        return batch