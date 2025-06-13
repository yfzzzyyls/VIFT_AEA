import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class AriaLatentDataset(Dataset):
    """Dataset for loading separate visual and IMU latent features."""
    
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        
        # Get all unique sample indices - keep the original filenames
        all_files = list(self.root_dir.glob("*_visual.npy"))
        self.file_prefixes = sorted([f.stem.split('_')[0] for f in all_files])
        
        # Verify all files exist
        for prefix in self.file_prefixes:
            assert (self.root_dir / f"{prefix}_visual.npy").exists(), f"Missing visual file for {prefix}"
            assert (self.root_dir / f"{prefix}_imu.npy").exists(), f"Missing IMU file for {prefix}"
            assert (self.root_dir / f"{prefix}_gt.npy").exists(), f"Missing GT file for {prefix}"
        
        # Translation statistics no longer needed - we keep translations in meters
        
        # Compute global mean/std for visual and IMU features
        all_vis = []
        all_imu = []
        for prefix in self.file_prefixes:
            v = np.load(self.root_dir / f"{prefix}_visual.npy")  # [seq_len, 512]
            i = np.load(self.root_dir / f"{prefix}_imu.npy")     # [seq_len, 256]
            all_vis.append(v)
            all_imu.append(i)
        all_vis = np.concatenate(all_vis, axis=0)  # [N*seq_len, 512]
        all_imu = np.concatenate(all_imu, axis=0)  # [N*seq_len, 256]
        self.vis_mean = torch.tensor(all_vis.mean(axis=0), dtype=torch.float32)
        self.vis_std = torch.tensor(all_vis.std(axis=0) + 1e-8, dtype=torch.float32)
        self.imu_mean = torch.tensor(all_imu.mean(axis=0), dtype=torch.float32)
        self.imu_std = torch.tensor(all_imu.std(axis=0) + 1e-8, dtype=torch.float32)
    
    def __len__(self):
        return len(self.file_prefixes)
    
    def __getitem__(self, idx):
        # Get actual file prefix
        file_prefix = self.file_prefixes[idx]
        
        # Determine sequence ID based on file ordering
        # Assuming test files are ordered: 0-98 -> seq 016, 99-197 -> seq 017, etc.
        test_sequences = ['016', '017', '018', '019']
        file_idx = int(file_prefix)
        samples_per_seq = 99  # 396 total files / 4 sequences
        seq_idx = min(file_idx // samples_per_seq, len(test_sequences) - 1)
        sequence_id = test_sequences[seq_idx]
        
        # Load features
        visual_features = np.load(self.root_dir / f"{file_prefix}_visual.npy")  # [10, 512]
        imu_features = np.load(self.root_dir / f"{file_prefix}_imu.npy")        # [10, 256]
        relative_poses = np.load(self.root_dir / f"{file_prefix}_gt.npy")       # [10, 7]
        
        # Convert to tensors and apply global normalization
        visual_features = torch.from_numpy(visual_features).float()
        visual_features = (visual_features - self.vis_mean.to(visual_features.device)) / self.vis_std.to(visual_features.device)
        imu_features = torch.from_numpy(imu_features).float()
        imu_features = (imu_features - self.imu_mean.to(imu_features.device)) / self.imu_std.to(imu_features.device)
        relative_poses = torch.from_numpy(relative_poses).float()
        
        # NOTE: GT files now contain poses in meters (standard VIO unit)
        # Keep translations in meters - no per-sample normalization
        # The model will learn to predict meter-scale translations directly
        
        batch = {
            'visual_features': visual_features,
            'imu_features': imu_features,
            'poses': relative_poses,  # Changed key to match training dataset
            'sequence_id': sequence_id,  # Add sequence ID for proper grouping
            'file_prefix': file_prefix,  # Keep for debugging
            'frame_idx': int(file_prefix)  # Add frame index for temporal ordering
        }
        
        return batch