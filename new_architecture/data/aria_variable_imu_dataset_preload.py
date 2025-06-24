"""
Aria Dataset with Variable-Length IMU Sequences - Preloaded Version
Loads all data into memory during initialization to avoid worker issues.
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
    Pre-loads all data into memory during initialization.
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
        image_size: Tuple[int, int] = (704, 704)
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
            self.split_sequences = splits['splits'][split]
        else:
            self.split_sequences = None
        
        # Pre-load all sequences into memory
        print(f"Pre-loading all {split} sequences into memory...")
        self.sequences_data = {}
        self._load_all_sequences()
        
        # Create samples with sliding windows
        self.samples = []
        self._create_samples()
        
        print(f"Created {len(self.samples)} samples from {len(self.sequences_data)} sequences")
        print(f"Variable length: {self.variable_length}")
        
    def _load_all_sequences(self):
        """Pre-load all sequences into memory."""
        sequence_names = self.split_sequences if self.split_sequences else []
        
        if not sequence_names:
            # Find all sequence directories
            for seq_dir in sorted(self.data_dir.iterdir()):
                if seq_dir.is_dir() and seq_dir.name.isdigit():
                    sequence_names.append(seq_dir.name)
        
        # Calculate total memory requirement
        total_sequences = len(sequence_names)
        print(f"  Will load {total_sequences} sequences for {self.split} split")
        
        for idx, seq_name in enumerate(sequence_names):
            seq_dir = self.data_dir / seq_name
            if self._is_valid_sequence(seq_dir):
                print(f"  Loading sequence {seq_name} ({idx+1}/{total_sequences})...")
                
                # Load all data for this sequence
                visual_data = torch.load(seq_dir / 'visual_data.pt', map_location='cpu')
                imu_data = torch.load(seq_dir / 'imu_data.pt', map_location='cpu')
                
                with open(seq_dir / 'poses_quaternion.json', 'r') as f:
                    poses_data = json.load(f)
                
                # Resize images if needed
                if visual_data.shape[-2:] != self.image_size:
                    visual_data = F.interpolate(visual_data, size=self.image_size, 
                                              mode='bilinear', align_corners=False)
                
                self.sequences_data[seq_name] = {
                    'visual': visual_data,
                    'imu': imu_data,
                    'poses': poses_data,
                    'length': visual_data.shape[0]
                }
        
        # Calculate memory usage
        mem_gb = sum(
            seq['visual'].element_size() * seq['visual'].nelement() / (1024**3)
            for seq in self.sequences_data.values()
        )
        # IMU data is stored as a list, estimate its size
        imu_gb = len(self.sequences_data) * 0.01  # ~10MB per sequence
        total_gb = mem_gb + imu_gb
        print(f"Loaded {len(self.sequences_data)} sequences into memory (~{total_gb:.1f} GB)")
        
    def _is_valid_sequence(self, seq_dir: Path) -> bool:
        """Check if a sequence directory has all required files."""
        visual_path = seq_dir / 'visual_data.pt'
        poses_path = seq_dir / 'poses_quaternion.json'
        imu_path = seq_dir / 'imu_data.pt'
        
        return visual_path.exists() and imu_path.exists() and poses_path.exists()
        
    def _create_samples(self):
        """Create training samples using sliding windows."""
        for seq_name, seq_data in self.sequences_data.items():
            seq_len = seq_data['length']
            
            if self.variable_length:
                max_window = min(self.max_seq_len, seq_len)
            else:
                max_window = min(self.sequence_length, seq_len)
            
            for start_idx in range(0, seq_len - self.min_seq_len + 1, self.stride):
                if self.variable_length:
                    window_size = random.randint(self.min_seq_len, 
                                               min(max_window, seq_len - start_idx))
                else:
                    window_size = min(self.sequence_length, seq_len - start_idx)
                
                if start_idx + window_size <= seq_len:
                    self.samples.append({
                        'sequence_name': seq_name,
                        'start_idx': start_idx,
                        'length': window_size
                    })
                    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from pre-loaded data."""
        sample_info = self.samples[idx]
        seq_name = sample_info['sequence_name']
        start_idx = sample_info['start_idx']
        length = sample_info['length']
        
        # Get data from pre-loaded sequences
        seq_data = self.sequences_data[seq_name]
        
        # Extract window
        images = seq_data['visual'][start_idx:start_idx + length]
        
        # Get IMU sequences
        imu_sequences = []
        for i in range(length - 1):
            frame_idx = start_idx + i
            if frame_idx < len(seq_data['imu']):
                imu_sequences.append(seq_data['imu'][frame_idx])
            else:
                imu_sequences.append(torch.zeros((1, 6), dtype=torch.float32))
        
        # Get poses and compute relative transformations
        poses_window = seq_data['poses'][start_idx:start_idx + length]
        poses = self._compute_relative_poses(poses_window)
        
        return {
            'images': images,
            'imu_sequences': imu_sequences,
            'poses': poses,
            'sequence_length': length
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