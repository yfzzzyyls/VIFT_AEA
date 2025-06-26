import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import json
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch.nn.utils.rnn import pad_sequence


class AriaRawDataset(Dataset):
    """
    Dataset for loading raw Aria data (images + IMU) for end-to-end training.
    Expects data processed with variable-length IMU format (all samples between frames).
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
        imu_data = torch.load(seq_dir / 'imu_data.pt')        # List of variable-length tensors
        
        with open(seq_dir / 'poses_quaternion.json', 'r') as f:
            poses_data = json.load(f)
        
        # Get number of frames
        num_frames = visual_data.shape[0]
        
        # IMU data must be in new format (list of variable-length tensors)
        if not isinstance(imu_data, list):
            raise ValueError(f"IMU data in {seq_dir} is in old format. Please reprocess with updated process_aria.py")
        
        num_imu_intervals = len(imu_data)
        
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
            
            # Extract list of variable-length tensors
            window_imu = imu_data[start_idx:start_idx + self.sequence_length - 1]  # List of tensors
            
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
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get visual data
        visual = sample['visual']  # [11, 3, H, W]
        
        # Resize to expected input size (704x704)
        if visual.shape[-2:] != (704, 704):
            visual = F.interpolate(visual, size=(704, 704), mode='bilinear', align_corners=False)
                
        # Get IMU data
        imu = sample['imu']  # List of variable-length tensors
        
        # For flexible encoder: Stack into padded tensor for batch processing
        max_len = max(tensor.shape[0] for tensor in imu)
        imu_padded = []
        for tensor in imu:
            if tensor.shape[0] < max_len:
                # Pad with zeros to max length
                padding = torch.zeros(max_len - tensor.shape[0], 6)
                padded = torch.cat([tensor, padding], dim=0)
            else:
                padded = tensor
            imu_padded.append(padded)
        imu_data = torch.stack(imu_padded)  # [10, max_len, 6]
        
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
        
        # Get frame IDs if available
        frame_ids = None
        if 'frame_ids' in sample:
            frame_ids = torch.tensor(sample['frame_ids'], dtype=torch.long)  # [11]
        elif 'timestamps' in sample:
            # Use timestamps as frame IDs if actual IDs not available
            # Convert to unique integers based on start_idx
            start_idx = sample['start_idx']
            frame_ids = torch.arange(start_idx, start_idx + 11, dtype=torch.long)
        
        result = {
            'images': visual,        # [11, 3, H, W]
            'imu': imu_data,  # [10, max_len, 6] with variable max_len per batch
            'gt_poses': gt_poses,   # [10, 7] (3 trans + 4 quat)
            'seq_name': sample['seq_name'],
            'start_idx': sample['start_idx']
        }
        
        if frame_ids is not None:
            result['frame_ids'] = frame_ids  # [11]
            
        return result


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
            pin_memory=True,
            collate_fn=collate_variable_length
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_variable_length
        )


def collate_variable_length(batch):
    """Custom collate function to handle variable-length IMU sequences."""
    # Stack fixed-size tensors normally
    images = torch.stack([item['images'] for item in batch])
    gt_poses = torch.stack([item['gt_poses'] for item in batch])
    
    # Handle IMU data - each item already has padded IMU data but with different max_len
    # Find the maximum length across all items in the batch
    max_len = max(item['imu'].shape[1] for item in batch)
    
    # Pad all IMU sequences to the same max length
    padded_imu_list = []
    for item in batch:
        imu_data = item['imu']  # [10, item_max_len, 6]
        item_max_len = imu_data.shape[1]
        
        if item_max_len < max_len:
            # Pad along the second dimension (samples dimension)
            padding = torch.zeros(10, max_len - item_max_len, 6)
            padded_imu = torch.cat([imu_data, padding], dim=1)
        else:
            padded_imu = imu_data
        
        padded_imu_list.append(padded_imu)
    
    imu = torch.stack(padded_imu_list)  # [B, 10, max_len, 6]
    
    # Metadata
    seq_names = [item['seq_name'] for item in batch]
    start_indices = [item['start_idx'] for item in batch]
    
    result = {
        'images': images,
        'imu': imu,
        'gt_poses': gt_poses,
        'seq_name': seq_names,
        'start_idx': start_indices
    }
    
    # Handle frame IDs if present
    if 'frame_ids' in batch[0]:
        frame_ids = torch.stack([item['frame_ids'] for item in batch])  # [B, 11]
        result['frame_ids'] = frame_ids
    
    return result