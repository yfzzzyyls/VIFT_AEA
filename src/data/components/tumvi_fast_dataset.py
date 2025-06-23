#!/usr/bin/env python3
"""
Fast TUM VI Dataset loader using preprocessed numpy arrays.
This is much faster than loading and resizing PNG images on the fly.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy import interpolate

class TUMVIFastDataset(Dataset):
    """Fast TUM VI dataset loader using preprocessed numpy arrays."""
    
    def __init__(self, 
                 root_dir,
                 sequence,
                 sequence_length=11,
                 stride=1,
                 use_preprocessed=True):
        """
        Args:
            root_dir: Root directory containing TUM VI sequences
            sequence: Sequence name (e.g., 'dataset-room1_512_16')
            sequence_length: Number of frames per sample (default: 11)
            stride: Stride between samples (default: 1)
            use_preprocessed: Use .npy files instead of .png (default: True)
        """
        self.root_dir = Path(root_dir)
        self.sequence = sequence
        self.sequence_path = self.root_dir / sequence
        self.sequence_length = sequence_length
        self.stride = stride
        self.use_preprocessed = use_preprocessed
        
        # Load data
        self._load_timestamps()
        self._load_imu_data()
        self._load_ground_truth()
        
        # Pre-compute valid start indices
        self.valid_indices = []
        for i in range(0, len(self.image_timestamps) - self.sequence_length + 1, self.stride):
            self.valid_indices.append(i)
        
        print(f"Loaded {self.sequence} with {len(self.valid_indices)} samples")
    
    def _load_timestamps(self):
        """Load image timestamps."""
        csv_path = self.sequence_path / 'mav0' / 'cam0' / 'data.csv'
        df = pd.read_csv(csv_path)
        
        self.image_timestamps = df['#timestamp [ns]'].values
        self.image_filenames = df['filename'].values
    
    def _load_imu_data(self):
        """Load IMU data."""
        imu_path = self.sequence_path / 'mav0' / 'imu0' / 'data.csv'
        df = pd.read_csv(imu_path, skipinitialspace=True)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        self.imu_timestamps = df['#timestamp [ns]'].values
        self.imu_data = df[['w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]',
                           'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]']].values
    
    def _load_ground_truth(self):
        """Load ground truth poses."""
        gt_path = self.sequence_path / 'mav0' / 'mocap0' / 'data.csv'
        df = pd.read_csv(gt_path, skipinitialspace=True)
        
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        
        self.gt_timestamps = df['#timestamp [ns]'].values
        self.gt_positions = df[['p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]']].values
        self.gt_quaternions = df[['q_RS_w []', 'q_RS_x []', 'q_RS_y []', 'q_RS_z []']].values
        
        # Convert quaternion format
        self.gt_quaternions = self.gt_quaternions[:, [1, 2, 3, 0]]  # [x,y,z,w]
    
    def _sync_imu_to_images(self, start_idx):
        """Get IMU measurements synchronized to image timestamps."""
        imu_synced = []
        
        for i in range(self.sequence_length - 1):
            t_start = self.image_timestamps[start_idx + i]
            t_end = self.image_timestamps[start_idx + i + 1]
            
            mask = (self.imu_timestamps >= t_start) & (self.imu_timestamps < t_end)
            imu_segment = self.imu_data[mask]
            
            if len(imu_segment) < 10:
                imu_segment = np.vstack([imu_segment, 
                                       np.zeros((10 - len(imu_segment), 6))])
            else:
                imu_segment = imu_segment[:10]
            
            imu_synced.append(imu_segment)
        
        return np.vstack(imu_synced)  # [110, 6]
    
    def _get_relative_poses(self, start_idx):
        """Get relative poses between consecutive frames."""
        positions = np.zeros((self.sequence_length, 3))
        quaternions = np.zeros((self.sequence_length, 4))
        
        for i in range(self.sequence_length):
            img_time = self.image_timestamps[start_idx + i]
            
            idx = np.searchsorted(self.gt_timestamps, img_time)
            if idx == 0:
                idx = 1
            elif idx >= len(self.gt_timestamps):
                idx = len(self.gt_timestamps) - 1
            
            t1 = self.gt_timestamps[idx-1]
            t2 = self.gt_timestamps[idx]
            alpha = (img_time - t1) / (t2 - t1)
            
            positions[i] = (1 - alpha) * self.gt_positions[idx-1] + alpha * self.gt_positions[idx]
            
            q0 = R.from_quat(self.gt_quaternions[idx-1])
            q1 = R.from_quat(self.gt_quaternions[idx])
            q_interp = R.from_quat(q0.as_quat()) * R.from_quat((q0.inv() * q1).as_quat()) ** alpha
            quaternions[i] = q_interp.as_quat()
        
        relative_poses = []
        for i in range(self.sequence_length - 1):
            rel_trans = positions[i+1] - positions[i]
            
            q1 = R.from_quat(quaternions[i])
            q2 = R.from_quat(quaternions[i+1])
            rel_rot = q1.inv() * q2
            rel_quat = rel_rot.as_quat()
            
            rel_pose = np.concatenate([rel_trans, rel_quat])
            relative_poses.append(rel_pose)
        
        return torch.tensor(np.array(relative_poses), dtype=torch.float32)
    
    def _load_image_fast(self, idx):
        """Load preprocessed image from numpy file."""
        if self.use_preprocessed:
            # Load from numpy file
            npy_path = self.sequence_path / 'mav0' / 'cam0' / 'data' / \
                       self.image_filenames[idx].replace('.png', '.npy')
            
            if npy_path.exists():
                # Load grayscale image and convert to RGB
                img = np.load(npy_path)  # Already normalized to [0, 1]
                img_rgb = np.stack([img, img, img], axis=2)
                return torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        
        # Fallback to loading PNG (should not happen with preprocessed data)
        raise RuntimeError(f"Preprocessed image not found. Run preprocess_tumvi_images.py first!")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        start_idx = self.valid_indices[idx]
        
        # Load sequence of images (fast from numpy)
        images = []
        for i in range(self.sequence_length):
            img = self._load_image_fast(start_idx + i)
            images.append(img)
        images = torch.stack(images)  # [11, 3, 256, 512]
        
        # Get synchronized IMU data
        imu = self._sync_imu_to_images(start_idx)  # [110, 6]
        imu = torch.from_numpy(imu).float()
        
        # Get relative poses
        gt_poses = self._get_relative_poses(start_idx)  # [10, 7]
        
        return {
            'images': images,
            'imu': imu,
            'gt_poses': gt_poses,
            'sequence_name': f"{self.sequence}_{start_idx:06d}"
        }