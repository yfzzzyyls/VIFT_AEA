"""
Aria Dataset with Variable-Length IMU Sequences - DDP Compatible Version
Follows the same pattern as the working train_aria_from_scratch.py
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F


class AriaVariableIMUDataset(Dataset):
    """
    Dataset for Aria Everyday Activities with variable-length IMU sequences.
    DDP-compatible version that pre-loads data windows like the working script.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        sequence_length: int = 11,
        variable_length: bool = True,
        min_seq_len: int = 5,
        max_seq_len: int = 50,
        stride: int = 1,
        image_size: Tuple[int, int] = (704, 704)  # Original resolution
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.variable_length = variable_length
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.image_size = image_size
        
        # Load split information
        splits_file = self.data_dir / 'splits.json'
        if splits_file.exists():
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            self.sequences = [self.data_dir / seq_name for seq_name in splits['splits'][split]]
        else:
            # Find all sequence directories
            self.sequences = []
            for seq_dir in sorted(self.data_dir.iterdir()):
                if seq_dir.is_dir() and seq_dir.name.isdigit():
                    self.sequences.append(seq_dir)
        
        print(f"Found {len(self.sequences)} sequences for {split}")
        
        # Create samples with sliding windows - STORE WINDOWS NOT FULL SEQUENCES
        self.samples = []
        for seq_dir in self.sequences:
            if self._is_valid_sequence(seq_dir):
                self._create_samples_from_sequence(seq_dir)
        
        print(f"Created {len(self.samples)} samples with stride {stride}")
        
    def _is_valid_sequence(self, seq_dir: Path) -> bool:
        """Check if a sequence directory has all required files."""
        visual_path = seq_dir / 'visual_data.pt'
        poses_path = seq_dir / 'poses_quaternion.json'
        imu_path = seq_dir / 'imu_data.pt'
        
        return visual_path.exists() and imu_path.exists() and poses_path.exists()
        
    def _create_samples_from_sequence(self, seq_dir: Path):
        """Create sliding window samples from a sequence - following working script pattern."""
        # Load data files
        visual_data = torch.load(seq_dir / 'visual_data.pt', map_location='cpu')  # [N, 3, H, W]
        imu_data = torch.load(seq_dir / 'imu_data.pt', map_location='cpu')        # List of variable-length tensors
        
        with open(seq_dir / 'poses_quaternion.json', 'r') as f:
            poses_data = json.load(f)
        
        # Get number of frames
        num_frames = visual_data.shape[0]
        num_imu_intervals = len(imu_data)
        
        # Ensure consistency
        if num_imu_intervals != num_frames - 1:
            min_frames = min(num_frames, num_imu_intervals + 1)
            visual_data = visual_data[:min_frames]
            imu_data = imu_data[:min_frames - 1]
            poses_data = poses_data[:min_frames]
            num_frames = min_frames
        
        # Determine max window size
        if self.variable_length:
            max_window = min(self.max_seq_len, num_frames)
        else:
            max_window = min(self.sequence_length, num_frames)
        
        # Create sliding windows with specified stride
        for start_idx in range(0, num_frames - self.min_seq_len + 1, self.stride):
            # For variable length, randomly determine window size
            if self.variable_length:
                window_size = random.randint(self.min_seq_len, 
                                           min(max_window, num_frames - start_idx))
            else:
                window_size = min(self.sequence_length, num_frames - start_idx)
            
            if start_idx + window_size > num_frames:
                continue
            
            end_idx = start_idx + window_size
            
            # Extract window data - STORE THE ACTUAL DATA LIKE WORKING SCRIPT
            window_visual = visual_data[start_idx:end_idx].clone()  # [window_size, 3, H, W]
            
            # Resize if needed
            if window_visual.shape[-2:] != self.image_size:
                # Resize all frames in the window
                B, C, H, W = window_visual.shape
                window_visual = window_visual.view(-1, C, H, W)
                window_visual = F.interpolate(window_visual, size=self.image_size, 
                                            mode='bilinear', align_corners=False)
                window_visual = window_visual.view(B, C, *self.image_size)
            
            # Extract IMU data for this window
            window_imu = []
            for i in range(window_size - 1):
                imu_idx = start_idx + i
                if imu_idx < len(imu_data):
                    window_imu.append(imu_data[imu_idx])
            
            # Extract poses for this window
            window_poses = poses_data[start_idx:end_idx]
            
            # Store the window data
            self.samples.append({
                'seq_name': seq_dir.name,
                'start_idx': start_idx,
                'visual': window_visual,        # Tensor: [window_size, 3, H, W]
                'imu': window_imu,             # List of tensors (variable length IMU)
                'poses': window_poses,         # List of pose dicts
                'length': window_size
            })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a pre-loaded sample."""
        sample = self.samples[idx]
        
        # Data is already loaded, just need to process poses
        visual = sample['visual']  # Already a tensor
        imu_sequences = sample['imu']  # List of tensors
        poses = self._compute_relative_poses(sample['poses'])
        
        return {
            'images': visual,
            'imu_sequences': imu_sequences,
            'poses': poses,
            'sequence_length': sample['length']
        }
    
    def _compute_relative_poses(self, poses_window: List[Dict]) -> torch.Tensor:
        """Compute relative poses between consecutive frames."""
        relative_poses = []
        
        for i in range(len(poses_window) - 1):
            # Current and next pose
            t1 = np.array(poses_window[i]['translation'])
            q1 = np.array(poses_window[i]['quaternion'])
            t2 = np.array(poses_window[i + 1]['translation'])
            q2 = np.array(poses_window[i + 1]['quaternion'])
            
            # Compute relative transformation
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            
            dt_world = t2 - t1
            dt_local = r1.inv().apply(dt_world)
            
            r_rel = r1.inv() * r2
            q_rel = r_rel.as_quat()
            
            relative_pose = np.concatenate([dt_local, q_rel])
            relative_poses.append(relative_pose)
        
        return torch.tensor(np.array(relative_poses), dtype=torch.float32)


def collate_variable_imu(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching variable-length IMU sequences."""
    # Find max sequence length in batch
    max_seq_len = max(sample['sequence_length'] for sample in batch)
    
    # Prepare batched data
    batched_images = []
    batched_imu_sequences = []
    batched_poses = []
    sequence_lengths = []
    
    for sample in batch:
        seq_len = sample['sequence_length']
        sequence_lengths.append(seq_len)
        
        # Pad images if needed
        images = sample['images']
        if images.shape[0] < max_seq_len:
            pad_len = max_seq_len - images.shape[0]
            images = F.pad(images, (0, 0, 0, 0, 0, 0, 0, pad_len))
        batched_images.append(images)
        
        # Handle variable-length IMU sequences
        imu_sequences = sample['imu_sequences']
        while len(imu_sequences) < max_seq_len - 1:
            # Pad with zeros
            imu_sequences.append(torch.zeros((1, 6), dtype=torch.float32))
        batched_imu_sequences.append(imu_sequences)
        
        # Pad poses if needed
        poses = sample['poses']
        if poses.shape[0] < max_seq_len - 1:
            pad_len = (max_seq_len - 1) - poses.shape[0]
            poses = F.pad(poses, (0, 0, 0, pad_len))
        batched_poses.append(poses)
    
    return {
        'images': torch.stack(batched_images),
        'imu_sequences': batched_imu_sequences,  # List of lists, can't stack due to variable IMU lengths
        'poses': torch.stack(batched_poses),
        'sequence_lengths': torch.tensor(sequence_lengths)
    }