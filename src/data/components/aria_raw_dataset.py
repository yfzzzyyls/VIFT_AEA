import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import json
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R


class AriaRawDataset(Dataset):
    """
    Dataset for loading raw Aria data (images + IMU) for end-to-end training.
    Expects data processed with fixed between-frames IMU format.
    """
    
    def __init__(self, data_dir, sequence_length=11, stride=10, transform=None):
        """
        Args:
            data_dir: Directory containing processed Aria sequences (e.g., aria_processed/train)
            sequence_length: Number of frames per sequence (default: 11 for VIFT)
            stride: Stride for sliding window (default: 10 for non-overlapping transitions)
            transform: Optional transforms to apply to images
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        
        # Get all sequences
        self.sequences = []
        for seq_dir in sorted(self.data_dir.iterdir()):
            if seq_dir.is_dir() and seq_dir.name.isdigit():
                # Check if it has the required files
                visual_path = seq_dir / 'visual_data.pt'
                imu_path = seq_dir / 'imu_data.pt'
                poses_path = seq_dir / 'poses_quaternion.json'
                
                if visual_path.exists() and imu_path.exists() and poses_path.exists():
                    self.sequences.append(seq_dir)
        
        print(f"Found {len(self.sequences)} sequences in {data_dir}")
        
        # Create samples with sliding windows
        self.samples = []
        for seq_dir in self.sequences:
            self._create_samples_from_sequence(seq_dir)
        
        print(f"Created {len(self.samples)} samples with window size {sequence_length}")
    
    def _create_samples_from_sequence(self, seq_dir):
        """Create sliding window samples from a sequence."""
        # Load data
        visual_data = torch.load(seq_dir / 'visual_data.pt')  # [N, 3, H, W]
        imu_data = torch.load(seq_dir / 'imu_data.pt')        # [N-1, 11, 6]
        
        with open(seq_dir / 'poses_quaternion.json', 'r') as f:
            poses_data = json.load(f)
        
        # Get number of frames
        num_frames = visual_data.shape[0]
        num_imu_intervals = imu_data.shape[0]
        
        # Ensure consistency
        if num_imu_intervals != num_frames - 1:
            min_frames = min(num_frames, num_imu_intervals + 1)
            visual_data = visual_data[:min_frames]
            imu_data = imu_data[:min_frames - 1]
            poses_data = poses_data[:min_frames]
            num_frames = min_frames
        
        # Create sliding windows with specified stride
        for start_idx in range(0, num_frames - self.sequence_length + 1, self.stride):
            end_idx = start_idx + self.sequence_length
            
            # Extract window data
            window_visual = visual_data[start_idx:end_idx]  # [11, 3, H, W]
            window_imu = imu_data[start_idx:start_idx + self.sequence_length - 1]  # [10, 11, 6]
            window_poses = poses_data[start_idx:end_idx]    # 11 poses
            
            # Store absolute poses for this window
            # Note: These are absolute poses, not relative. The relative pose computation
            # happens in __getitem__ to allow flexibility in representation
            absolute_poses = []
            for i in range(len(window_poses)):
                pose = window_poses[i]
                absolute_poses.append({
                    'translation': pose['translation'],
                    'quaternion': pose['quaternion']
                })
            
            self.samples.append({
                'seq_name': seq_dir.name,
                'start_idx': start_idx,
                'visual': window_visual,
                'imu': window_imu,
                'poses': absolute_poses
            })
    
    def __len__(self):
        return len(self.samples)
    
    def _remove_gravity(self, imu_window):
        """Remove gravity bias from IMU accelerometer data
        
        Args:
            imu_window: [110,6] tensor with (ax,ay,az,gx,gy,gz)
        
        Returns:
            IMU tensor with gravity-bias removed from accelerometer
        """
        # Validate IMU format
        assert imu_window.shape[1] == 6, f"IMU must have 6 channels (ax,ay,az,gx,gy,gz), got {imu_window.shape[1]}"
        
        # Extract accelerometer data
        accel = imu_window[:, :3]  # [110, 3]
        
        # Reshape to compute per-frame bias (11 frames x 10 IMU samples per frame)
        # Average across the 10 samples within each frame
        accel_reshaped = accel.view(11, 10, 3)
        bias = accel_reshaped.mean(dim=1, keepdim=True)  # [11, 1, 3]
        
        # Expand bias for each of the 10 samples in each frame
        bias_expanded = bias.expand(11, 10, 3).reshape(110, 3)
        
        # Remove bias from accelerometer
        accel_corrected = accel - bias_expanded
        
        # Concatenate back with gyroscope data
        return torch.cat([accel_corrected, imu_window[:, 3:]], dim=-1)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get visual data
        visual = sample['visual']  # [11, 3, H, W]
        
        # Resize to expected input size (512x256)
        if visual.shape[-2:] != (256, 512):
            visual = F.interpolate(visual, size=(256, 512), mode='bilinear', align_corners=False)
                
        # Get IMU data - already in correct format (11 samples per interval)
        # VIFT expects 110 samples (11 per transition)
        imu = sample['imu']  # [10, 11, 6]
        
        # Reshape to [110, 6]
        imu_110 = imu.reshape(-1, 6)  # [110, 6]
        
        # Validate IMU data has expected format (ax, ay, az, gx, gy, gz)
        # Gyroscope values should be reasonable (< 10 rad/s typical, < 20 rad/s extreme)
        gyro_data = imu_110[:, 3:]  # Extract gyroscope columns
        gyro_magnitude = torch.norm(gyro_data, dim=-1)
        if gyro_magnitude.max() > 20.0:
            raise ValueError(f"Gyroscope values too large (max: {gyro_magnitude.max():.2f} rad/s). "
                           f"Expected IMU format: (ax, ay, az, gx, gy, gz)")
        
        # Check accelerometer range (typical range is ±40 m/s² for Aria)
        accel_data = imu_110[:, :3]
        accel_magnitude = torch.norm(accel_data, dim=-1)
        if accel_magnitude.max() > 100.0:
            raise ValueError(f"Accelerometer values too large (max: {accel_magnitude.max():.2f} m/s²). "
                           f"Expected IMU format: (ax, ay, az, gx, gy, gz)")
        
        # Remove gravity bias from accelerometer for consistent preprocessing
        imu_110 = self._remove_gravity(imu_110)
        
        # Get ground truth poses
        poses = sample['poses']
        
        # Convert to relative poses in local coordinates
        gt_poses = []
        for i in range(len(poses) - 1):
            # Compute relative transformation
            t1 = np.array(poses[i]['translation'])
            q1 = np.array(poses[i]['quaternion'])
            t2 = np.array(poses[i + 1]['translation'])
            q2 = np.array(poses[i + 1]['quaternion'])
            
            # Compute relative transformation in local frame
            # First get rotations
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            
            # Relative translation in world frame
            dt_world = t2 - t1
            
            # Transform to local frame of pose1
            dt_local = r1.inv().apply(dt_world)
            
            # Relative rotation: q_rel = q1^(-1) * q2
            r_rel = r1.inv() * r2
            q_rel = r_rel.as_quat()  # Returns in xyzw format
            
            gt_poses.append(np.concatenate([dt_local, q_rel]))  # [7]
        
        gt_poses = torch.tensor(np.array(gt_poses), dtype=torch.float32)  # [10, 7]
        
        return {
            'images': visual,        # [11, 3, 256, 512]
            'imu': imu_110,         # [110, 6]
            'gt_poses': gt_poses,   # [10, 7] (3 trans + 4 quat)
            'seq_name': sample['seq_name'],
            'start_idx': sample['start_idx']
        }


class AriaRawDataModule:
    """Data module for raw Aria data."""
    
    def __init__(self, data_dir, batch_size=4, num_workers=4, sequence_length=11, stride=10):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequence_length = sequence_length
        self.stride = stride
    
    def setup(self):
        # Create datasets
        self.train_dataset = AriaRawDataset(
            self.data_dir / 'train',
            sequence_length=self.sequence_length,
            stride=self.stride
        )
        
        self.val_dataset = AriaRawDataset(
            self.data_dir / 'val',
            sequence_length=self.sequence_length,
            stride=self.stride
        )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )