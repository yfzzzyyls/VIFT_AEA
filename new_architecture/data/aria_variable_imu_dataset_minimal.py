"""
Minimal Aria Dataset - Loads data on demand, no pre-loading
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
    Minimal dataset that loads data on-demand without pre-loading.
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
            self.sequence_names = splits['splits'][split]
        else:
            self.sequence_names = [d.name for d in self.data_dir.iterdir() 
                                  if d.is_dir() and d.name.isdigit()]
        
        # Create sample indices without loading any data
        self.samples = []
        for seq_name in self.sequence_names:
            seq_dir = self.data_dir / seq_name
            if self._is_valid_sequence(seq_dir):
                # Just get the number of frames from metadata
                metadata_path = seq_dir / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    num_frames = metadata['num_frames']
                else:
                    # Have to load just to get shape
                    visual_data = torch.load(seq_dir / 'visual_data.pt', map_location='cpu')
                    num_frames = visual_data.shape[0]
                    del visual_data  # Free memory immediately
                
                # Create sample indices
                for start_idx in range(0, num_frames - self.min_seq_len + 1, self.stride):
                    self.samples.append({
                        'seq_name': seq_name,
                        'start_idx': start_idx,
                        'max_len': num_frames - start_idx
                    })
        
        print(f"Created {len(self.samples)} sample indices for {split} split")
        
    def _is_valid_sequence(self, seq_dir: Path) -> bool:
        """Check if a sequence directory has all required files."""
        return all((seq_dir / f).exists() for f in 
                  ['visual_data.pt', 'imu_data.pt', 'poses_quaternion.json'])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load data on-demand."""
        sample_info = self.samples[idx]
        seq_dir = self.data_dir / sample_info['seq_name']
        start_idx = sample_info['start_idx']
        
        # Determine sequence length
        if self.variable_length:
            max_len = min(self.max_seq_len, sample_info['max_len'])
            length = random.randint(self.min_seq_len, max_len)
        else:
            length = min(self.sequence_length, sample_info['max_len'])
        
        # Load only the needed portion of data
        visual_data = torch.load(seq_dir / 'visual_data.pt', map_location='cpu')
        images = visual_data[start_idx:start_idx + length].clone()
        del visual_data  # Free memory immediately
        
        # Resize if needed
        if images.shape[-2:] != self.image_size:
            images = F.interpolate(images, size=self.image_size, 
                                 mode='bilinear', align_corners=False)
        
        # Load IMU data
        all_imu = torch.load(seq_dir / 'imu_data.pt', map_location='cpu')
        imu_sequences = []
        for i in range(length - 1):
            frame_idx = start_idx + i
            if frame_idx < len(all_imu):
                imu_sequences.append(all_imu[frame_idx])
            else:
                imu_sequences.append(torch.zeros((1, 6), dtype=torch.float32))
        del all_imu  # Free memory immediately
        
        # Load poses
        with open(seq_dir / 'poses_quaternion.json', 'r') as f:
            all_poses = json.load(f)
        poses_window = all_poses[start_idx:start_idx + length]
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
        'imu_sequences': batched_imu_sequences,
        'poses': torch.stack(batched_poses),
        'sequence_lengths': torch.tensor(sequence_lengths)
    }