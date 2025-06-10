"""Fixed AriaDataModule with stride support and augmentation."""

from typing import Any, Dict, Optional, Tuple, List
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import random


class AriaDatasetFixed(Dataset):
    """Enhanced dataset with stride support and augmentation."""
    
    def __init__(
        self, 
        data_dir: str, 
        sequence_ids: list, 
        seq_len: int = 11,
        stride: int = 20,
        pose_scale: float = 100.0,
        augment: bool = False,
        augment_prob: float = 0.5,
        noise_std: float = 0.001,
        use_latent: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.sequence_ids = sequence_ids
        self.seq_len = seq_len
        self.stride = stride
        self.pose_scale = pose_scale
        self.augment = augment
        self.augment_prob = augment_prob
        self.noise_std = noise_std
        self.use_latent = use_latent
        
        print(f"Dataset config: stride={stride}, pose_scale={pose_scale}, augment={augment}")
        
        # Load all sequences
        self.sequences = []
        for seq_id in sequence_ids:
            if isinstance(seq_id, int):
                seq_dir = self.data_dir / f"seq_{seq_id:03d}"
            else:
                seq_dir = self.data_dir / seq_id
                
            if seq_dir.exists():
                seq_data = self._load_sequence(seq_dir)
                if seq_data is not None:
                    self.sequences.append(seq_data)
                else:
                    print(f"⚠️ Failed to load sequence: {seq_dir}")
            else:
                print(f"⚠️ Sequence directory not found: {seq_dir}")
        
        if not self.sequences:
            raise ValueError(f"No valid sequences found in {self.data_dir}")
        
        # Create samples with stride
        self.samples = self._create_strided_windows()
        print(f"Created {len(self.samples)} training samples")
    
    def _load_sequence(self, seq_dir: Path) -> Optional[Dict]:
        """Load a single sequence with proper error handling."""
        try:
            # Load poses - check for quaternion version first
            poses_file = seq_dir / "poses_quaternion.json"
            if not poses_file.exists():
                poses_file = seq_dir / "poses.json"
            
            with open(poses_file, 'r') as f:
                poses_data = json.load(f)
            
            # Convert poses to numpy array with quaternions
            poses = []
            for pose in poses_data:
                t = pose['translation']
                if 'quaternion' in pose:
                    q = pose['quaternion']  # [x,y,z,w]
                else:
                    # Convert from euler
                    euler = pose.get('rotation_euler', [0,0,0])
                    r = Rotation.from_euler('xyz', euler)
                    q = r.as_quat()  # [x,y,z,w]
                poses.append([t[0], t[1], t[2], q[0], q[1], q[2], q[3]])
            
            poses = np.array(poses, dtype=np.float32)
            
            # Note: Translation scaling is already done during feature generation
            # Do NOT scale here to avoid double scaling
            # poses[:, :3] *= self.pose_scale  # COMMENTED OUT to prevent double scaling
            
            # Load IMU data
            imu_data = torch.load(seq_dir / "imu_data.pt").float()
            
            # Load visual features
            if self.use_latent:
                # Try cached features first
                visual_file = seq_dir / "visual_features.pt"
                if not visual_file.exists():
                    visual_file = seq_dir / "visual_latents.pt"
            else:
                visual_file = seq_dir / "visual_data.pt"
            
            if not visual_file.exists():
                print(f"⚠️ Visual features not found in {seq_dir}")
                return None
                
            visual_features = torch.load(visual_file).float()
            
            return {
                'poses': poses,
                'visual_features': visual_features,
                'imu_data': imu_data,
                'seq_name': seq_dir.name
            }
            
        except Exception as e:
            print(f"Error loading sequence {seq_dir}: {e}")
            return None
    
    def _create_strided_windows(self) -> List[Dict]:
        """Create sliding windows with configurable stride."""
        samples = []
        
        for seq in self.sequences:
            poses = seq['poses']
            visual = seq['visual_features']
            imu = seq['imu_data']
            
            # Ensure consistent lengths
            min_len = min(len(poses), len(visual), len(imu))
            
            # Create windows with stride
            for start_idx in range(0, min_len - self.seq_len * self.stride, self.stride):
                # Get indices with stride
                indices = [start_idx + i * self.stride for i in range(self.seq_len)]
                
                if indices[-1] < min_len:
                    # Convert absolute poses to relative
                    window_poses = poses[indices]
                    relative_poses = self._compute_relative_poses(window_poses)
                    
                    samples.append({
                        'visual_indices': indices,
                        'relative_poses': relative_poses,
                        'sequence': seq,
                    })
        
        return samples
    
    def _compute_relative_poses(self, absolute_poses: np.ndarray) -> np.ndarray:
        """Convert absolute poses to relative poses between consecutive frames."""
        relative_poses = []
        
        for i in range(1, len(absolute_poses)):
            # Previous and current poses
            prev_pos = absolute_poses[i-1, :3]
            curr_pos = absolute_poses[i, :3]
            
            prev_rot = Rotation.from_quat(absolute_poses[i-1, 3:])
            curr_rot = Rotation.from_quat(absolute_poses[i, 3:])
            
            # Relative translation in previous frame's coordinates
            rel_trans = prev_rot.inv().apply(curr_pos - prev_pos)
            
            # Relative rotation
            rel_rot = prev_rot.inv() * curr_rot
            rel_quat = rel_rot.as_quat()
            
            relative_poses.append(np.concatenate([rel_trans, rel_quat]))
        
        return np.array(relative_poses, dtype=np.float32)
    
    def _augment_sample(self, visual_features, imu_features, relative_poses):
        """Apply data augmentation."""
        if not self.augment or random.random() > self.augment_prob:
            return visual_features, imu_features, relative_poses
        
        # Add small noise to prevent overfitting
        if self.noise_std > 0:
            pose_noise = torch.randn_like(relative_poses) * self.noise_std
            relative_poses = relative_poses + pose_noise
            
            # Renormalize quaternions
            for i in range(len(relative_poses)):
                q = relative_poses[i, 3:]
                q = q / (torch.norm(q) + 1e-8)
                relative_poses[i, 3:] = q
        
        # Could add more augmentations here:
        # - Temporal jittering
        # - IMU noise
        # - Visual feature dropout
        
        return visual_features, imu_features, relative_poses
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq = sample['sequence']
        indices = sample['visual_indices']
        relative_poses = torch.tensor(sample['relative_poses'])
        
        # Get features at strided indices
        visual_features = seq['visual_features'][indices]
        imu_features = seq['imu_data'][indices]
        
        # Apply augmentation
        visual_features, imu_features, relative_poses = self._augment_sample(
            visual_features, imu_features, relative_poses
        )
        
        return {
            'visual_features': visual_features,
            'imu_features': imu_features,
            'relative_poses': relative_poses,
            'seq_name': seq['seq_name'],
            'indices': indices
        }


class AriaDataModuleFixed(LightningDataModule):
    """Fixed DataModule with stride and augmentation support."""
    
    def __init__(
        self,
        data_dir: str = "aria_latent_data",
        batch_size: int = 32,
        num_workers: int = 8,
        window_size: int = 11,
        stride: int = 20,
        pose_scale: float = 100.0,
        augment: bool = True,
        augment_prob: float = 0.5,
        noise_std: float = 0.001,
        train_sequences: Optional[List] = None,
        val_sequences: Optional[List] = None,
        test_sequences: Optional[List] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        # Default sequence split if not provided
        if train_sequences is None:
            # Use most sequences for training
            self.train_sequences = list(range(0, 100))
        else:
            self.train_sequences = train_sequences
            
        if val_sequences is None:
            # Use some sequences for validation
            self.val_sequences = list(range(100, 115))
        else:
            self.val_sequences = val_sequences
            
        if test_sequences is None:
            # Reserve some for testing
            self.test_sequences = list(range(115, 143))
        else:
            self.test_sequences = test_sequences
    
    def setup(self, stage: Optional[str] = None):
        """Load datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = AriaDatasetFixed(
                data_dir=self.hparams.data_dir,
                sequence_ids=self.train_sequences,
                seq_len=self.hparams.window_size,
                stride=self.hparams.stride,
                pose_scale=self.hparams.pose_scale,
                augment=self.hparams.augment,
                augment_prob=self.hparams.augment_prob,
                noise_std=self.hparams.noise_std,
            )
            
            self.val_dataset = AriaDatasetFixed(
                data_dir=self.hparams.data_dir,
                sequence_ids=self.val_sequences,
                seq_len=self.hparams.window_size,
                stride=self.hparams.stride,
                pose_scale=self.hparams.pose_scale,
                augment=False,  # No augmentation for validation
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = AriaDatasetFixed(
                data_dir=self.hparams.data_dir,
                sequence_ids=self.test_sequences,
                seq_len=self.hparams.window_size,
                stride=self.hparams.stride,
                pose_scale=self.hparams.pose_scale,
                augment=False,
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
        )