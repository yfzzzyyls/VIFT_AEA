#!/usr/bin/env python3
"""
TUM VI Dataset loader for VIFT.
Handles loading and preprocessing of TUM Visual-Inertial dataset.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from pathlib import Path
import cv2
from scipy.spatial.transform import Rotation as R
from scipy import interpolate


class TUMVIDataset(Dataset):
    """TUM VI dataset loader compatible with VIFT architecture."""
    
    def __init__(self, 
                 root_dir,
                 sequence,
                 sequence_length=11,
                 stride=1,
                 transform=None,
                 use_left_cam=True):
        """
        Args:
            root_dir: Root directory containing TUM VI sequences
            sequence: Sequence name (e.g., 'room1', 'room2', etc.)
            sequence_length: Number of frames per sample (default: 11)
            stride: Stride between samples (default: 1)
            transform: Optional transform to apply to images
            use_left_cam: Use left camera (cam0) or right (cam1)
        """
        self.root_dir = Path(root_dir)
        self.sequence = sequence
        
        self.sequence_path = self.root_dir / sequence
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        self.camera = 'cam0' if use_left_cam else 'cam1'
        
        # Load data
        self._load_timestamps()
        self._load_imu_data()
        self._load_ground_truth()
        
        # Create valid indices for sampling
        self._create_valid_indices()
        
    def _load_timestamps(self):
        """Load image timestamps."""
        # TUM VI stores timestamps in data.csv
        cam_data_path = self.sequence_path / 'mav0' / self.camera / 'data.csv'
        
        if not cam_data_path.exists():
            raise FileNotFoundError(f"Camera data file not found: {cam_data_path}")
        
        # Read CSV (timestamp [ns], filename)
        df = pd.read_csv(cam_data_path, header=0, names=['timestamp', 'filename'])
        self.image_timestamps = df['timestamp'].values  # in nanoseconds
        self.image_filenames = df['filename'].values
        
        # Convert to seconds
        self.image_timestamps_sec = self.image_timestamps * 1e-9
        
    def _load_imu_data(self):
        """Load and preprocess IMU data."""
        imu_data_path = self.sequence_path / 'mav0' / 'imu0' / 'data.csv'
        
        if not imu_data_path.exists():
            raise FileNotFoundError(f"IMU data file not found: {imu_data_path}")
        
        # Read IMU data (timestamp [ns], wx, wy, wz, ax, ay, az)
        df = pd.read_csv(imu_data_path, header=0, 
                        names=['timestamp', 'wx', 'wy', 'wz', 'ax', 'ay', 'az'])
        
        self.imu_timestamps = df['timestamp'].values * 1e-9  # Convert to seconds
        
        # Stack as [N, 6] array with order matching VIFT (accel first, then gyro)
        self.imu_data = np.stack([
            df['ax'].values, df['ay'].values, df['az'].values,
            df['wx'].values, df['wy'].values, df['wz'].values
        ], axis=1).astype(np.float32)
        
    def _load_ground_truth(self):
        """Load ground truth poses."""
        gt_path = self.sequence_path / 'mav0' / 'mocap0' / 'data.csv'
        
        if not gt_path.exists():
            # Try alternative GT path
            gt_path = self.sequence_path / 'mav0' / 'state_groundtruth_estimate0' / 'data.csv'
        
        if not gt_path.exists():
            print(f"Warning: Ground truth not found for sequence {self.sequence}")
            self.has_gt = False
            return
        
        # Read ground truth (timestamp [ns], px, py, pz, qw, qx, qy, qz, ...)
        df = pd.read_csv(gt_path, header=0)
        
        self.gt_timestamps = df.iloc[:, 0].values * 1e-9  # Convert to seconds
        self.gt_positions = df.iloc[:, 1:4].values  # px, py, pz
        
        # TUM VI uses Hamilton convention (qw, qx, qy, qz)
        # Convert to our convention (qx, qy, qz, qw)
        self.gt_quaternions = np.stack([
            df.iloc[:, 5].values,  # qx
            df.iloc[:, 6].values,  # qy
            df.iloc[:, 7].values,  # qz
            df.iloc[:, 4].values   # qw
        ], axis=1)
        
        self.has_gt = True
        
    def _create_valid_indices(self):
        """Create valid starting indices for sequences."""
        # Need at least sequence_length frames
        max_start_idx = len(self.image_timestamps) - self.sequence_length
        
        # Also ensure we have enough IMU data
        # For each image sequence, we need corresponding IMU data
        valid_indices = []
        
        for idx in range(0, max_start_idx, self.stride):
            # Check if we have IMU data for this time range
            start_time = self.image_timestamps_sec[idx]
            end_time = self.image_timestamps_sec[idx + self.sequence_length - 1]
            
            # Need IMU data covering this range
            imu_mask = (self.imu_timestamps >= start_time - 0.1) & \
                      (self.imu_timestamps <= end_time + 0.1)
            
            if np.sum(imu_mask) >= 110:  # Need at least 110 IMU samples
                valid_indices.append(idx)
        
        self.valid_indices = valid_indices
        
    def _sync_imu_to_images(self, start_idx):
        """Synchronize IMU data to image timestamps."""
        # Get image timestamps for this sequence
        img_times = self.image_timestamps_sec[start_idx:start_idx + self.sequence_length]
        
        # For each consecutive image pair, get 11 IMU samples
        synced_imu = []
        
        for i in range(self.sequence_length - 1):
            t_start = img_times[i]
            t_end = img_times[i + 1]
            
            # VIFT expects 11 IMU samples per transition
            # With 20Hz camera and 200Hz IMU, we naturally have 10 samples
            # So we interpolate to get 11 samples
            t_samples = np.linspace(t_start, t_end, 11)  # 11 samples including boundaries
            
            interpolated_imu = np.zeros((11, 6))
            for j in range(6):
                f = interpolate.interp1d(self.imu_timestamps, self.imu_data[:, j], 
                                       kind='linear', fill_value='extrapolate')
                interpolated_imu[:, j] = f(t_samples)
            
            synced_imu.append(interpolated_imu)
        
        # Stack to get [110, 6] array (10 transitions × 11 samples)
        return np.vstack(synced_imu).astype(np.float32)
        
    def _get_relative_poses(self, start_idx):
        """Get relative poses between consecutive frames."""
        if not self.has_gt:
            # Return dummy poses
            return torch.zeros(self.sequence_length - 1, 7)
        
        # Get image timestamps
        img_times = self.image_timestamps_sec[start_idx:start_idx + self.sequence_length]
        
        # Interpolate ground truth to image timestamps
        positions = np.zeros((self.sequence_length, 3))
        quaternions = np.zeros((self.sequence_length, 4))
        
        for i in range(self.sequence_length):
            t = img_times[i]
            
            # Find nearest GT timestamps
            idx = np.searchsorted(self.gt_timestamps, t)
            if idx == 0:
                positions[i] = self.gt_positions[0]
                quaternions[i] = self.gt_quaternions[0]
            elif idx >= len(self.gt_timestamps):
                positions[i] = self.gt_positions[-1]
                quaternions[i] = self.gt_quaternions[-1]
            else:
                # Linear interpolation for position
                t0, t1 = self.gt_timestamps[idx-1], self.gt_timestamps[idx]
                alpha = (t - t0) / (t1 - t0)
                positions[i] = (1 - alpha) * self.gt_positions[idx-1] + alpha * self.gt_positions[idx]
                
                # SLERP for quaternion
                q0 = R.from_quat(self.gt_quaternions[idx-1])
                q1 = R.from_quat(self.gt_quaternions[idx])
                q_interp = R.from_quat(q0.as_quat()) * R.from_quat((q0.inv() * q1).as_quat()) ** alpha
                quaternions[i] = q_interp.as_quat()
        
        # Compute relative poses
        relative_poses = []
        for i in range(self.sequence_length - 1):
            # Relative translation
            rel_trans = positions[i+1] - positions[i]
            
            # Relative rotation
            q1 = R.from_quat(quaternions[i])
            q2 = R.from_quat(quaternions[i+1])
            rel_rot = q1.inv() * q2
            rel_quat = rel_rot.as_quat()
            
            # Combine [trans(3), quat(4)]
            rel_pose = np.concatenate([rel_trans, rel_quat])
            relative_poses.append(rel_pose)
        
        return torch.tensor(np.array(relative_poses), dtype=torch.float32)
        
    def _load_and_preprocess_image(self, idx):
        """Load and preprocess a single image."""
        # Image path
        img_path = self.sequence_path / 'mav0' / self.camera / 'data' / self.image_filenames[idx]
        
        # Load image
        img = Image.open(img_path)
        
        # Handle both 512x512 and 1024x1024 images
        if img.size == (1024, 1024):
            # For 1024x1024 images (corridor sequences)
            # Direct resize to 512x256 maintaining more of the image content
            img = img.resize((512, 256), Image.LANCZOS)
        elif img.size == (512, 512):
            # For 512x512 images (room sequences)
            # Resize to target 512x256
            img = img.resize((512, 256), Image.LANCZOS)
        else:
            # Handle any other sizes
            img = img.resize((512, 256), Image.LANCZOS)
        
        # Convert to RGB if grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply transforms if any
        if self.transform:
            img = self.transform(img)
        else:
            # Default: convert to tensor and normalize
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
        
        return img
        
    def __len__(self):
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        """Get a single sample."""
        start_idx = self.valid_indices[idx]
        
        # Load sequence of images
        images = []
        for i in range(self.sequence_length):
            img = self._load_and_preprocess_image(start_idx + i)
            images.append(img)
        images = torch.stack(images)  # [11, 3, 256, 512]
        
        # Get synchronized IMU data
        imu = self._sync_imu_to_images(start_idx)  # [110, 6]
        imu = torch.from_numpy(imu)
        
        # Get relative poses
        gt_poses = self._get_relative_poses(start_idx)  # [10, 7]
        
        return {
            'images': images,
            'imu': imu,
            'gt_poses': gt_poses,
            'sequence_name': f"{self.sequence}_{start_idx:06d}"
        }


class TUMVIDataModule:
    """Data module for TUM VI dataset."""
    
    def __init__(self,
                 root_dir,
                 batch_size=4,
                 num_workers=4,
                 sequence_length=11,
                 stride=1,
                 train_sequences=None,
                 val_sequences=None):
        """
        Args:
            root_dir: Root directory containing TUM VI data
            batch_size: Batch size
            num_workers: Number of data loading workers
            sequence_length: Number of frames per sample
            stride: Stride between samples
            train_sequences: List of training sequences (default: room1-4)
            val_sequences: List of validation sequences (default: room5-6)
        """
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Default sequences if not specified
        if train_sequences is None:
            train_sequences = ['room1', 'room2', 'room3', 'room4']
        if val_sequences is None:
            val_sequences = ['room5', 'room6']
            
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        
    def setup(self):
        """Setup datasets."""
        # Create train dataset from multiple sequences
        train_datasets = []
        for seq in self.train_sequences:
            if (self.root_dir / seq).exists():
                dataset = TUMVIDataset(
                    self.root_dir,
                    seq,
                    sequence_length=self.sequence_length,
                    stride=self.stride
                )
                train_datasets.append(dataset)
            else:
                print(f"Warning: Training sequence {seq} not found")
        
        if train_datasets:
            self.train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        else:
            raise RuntimeError("No training sequences found")
        
        # Create validation dataset
        val_datasets = []
        for seq in self.val_sequences:
            if (self.root_dir / seq).exists():
                dataset = TUMVIDataset(
                    self.root_dir,
                    seq,
                    sequence_length=self.sequence_length,
                    stride=10  # Larger stride for validation
                )
                val_datasets.append(dataset)
            else:
                print(f"Warning: Validation sequence {seq} not found")
        
        if val_datasets:
            self.val_dataset = torch.utils.data.ConcatDataset(val_datasets)
        else:
            print("Warning: No validation sequences found")
            self.val_dataset = None
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )