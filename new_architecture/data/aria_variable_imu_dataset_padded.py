"""
Aria Dataset with Padded Variable-Length IMU Sequences
Maintains all functionality while being DDP-compatible
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
    Uses padded tensors for efficient GPU transfer while maintaining variable-length functionality.
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
        image_size: Tuple[int, int] = (704, 704),
        max_imu_length: int = 400  # Max IMU samples per interval (351 + buffer)
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.variable_length = variable_length
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.image_size = image_size
        self.max_imu_length = max_imu_length
        
        # Load split information
        splits_file = self.data_dir / 'splits.json'
        if splits_file.exists():
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            self.sequence_names = splits['splits'][split]
        else:
            self.sequence_names = [d.name for d in self.data_dir.iterdir() 
                                  if d.is_dir() and d.name.isdigit()]
        
        # Create sample indices without loading data
        self.samples = []
        self._create_sample_indices()
        
        print(f"Created {len(self.samples)} sample indices for {split} split")
        
        # Pre-generate random lengths for variable sequences (for consistency)
        if self.variable_length:
            random.seed(42)  # Fixed seed for reproducibility
            self.sample_lengths = []
            for sample in self.samples:
                max_len = min(self.max_seq_len, sample['max_len'])
                length = random.randint(self.min_seq_len, max_len)
                self.sample_lengths.append(length)
        
    def _create_sample_indices(self):
        """Create sample indices without loading data."""
        for seq_name in self.sequence_names:
            seq_dir = self.data_dir / seq_name
            if not self._is_valid_sequence(seq_dir):
                continue
                
            # Get number of frames from metadata
            metadata_path = seq_dir / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                num_frames = metadata['num_frames']
            else:
                # Fallback: load to get shape
                visual_data = torch.load(seq_dir / 'visual_data.pt', map_location='cpu')
                num_frames = visual_data.shape[0]
                del visual_data
            
            # Create sample indices
            for start_idx in range(0, num_frames - self.min_seq_len + 1, self.stride):
                self.samples.append({
                    'seq_name': seq_name,
                    'start_idx': start_idx,
                    'max_len': num_frames - start_idx
                })
    
    def _is_valid_sequence(self, seq_dir: Path) -> bool:
        """Check if a sequence directory has all required files."""
        return all((seq_dir / f).exists() for f in 
                  ['visual_data.pt', 'imu_data.pt', 'poses_quaternion.json'])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load data and convert to padded tensors."""
        sample_info = self.samples[idx]
        seq_dir = self.data_dir / sample_info['seq_name']
        start_idx = sample_info['start_idx']
        
        # Determine sequence length
        if self.variable_length:
            length = self.sample_lengths[idx]
        else:
            length = min(self.sequence_length, sample_info['max_len'])
        
        # Load visual data
        visual_data = torch.load(seq_dir / 'visual_data.pt', map_location='cpu')
        images = visual_data[start_idx:start_idx + length].clone()
        del visual_data
        
        # Resize if needed
        if images.shape[-2:] != self.image_size:
            images = F.interpolate(images, size=self.image_size, 
                                 mode='bilinear', align_corners=False)
        
        # Load IMU data and convert to padded tensor
        all_imu = torch.load(seq_dir / 'imu_data.pt', map_location='cpu')
        
        # Create padded IMU tensor and lengths
        imu_padded = torch.zeros(length - 1, self.max_imu_length, 6)
        imu_lengths = torch.zeros(length - 1, dtype=torch.long)
        
        for i in range(length - 1):
            frame_idx = start_idx + i
            if frame_idx < len(all_imu):
                imu_seq = all_imu[frame_idx]
                actual_len = min(imu_seq.shape[0], self.max_imu_length)
                imu_padded[i, :actual_len] = imu_seq[:actual_len]
                imu_lengths[i] = actual_len
            else:
                # No IMU data, use length 1 with zeros
                imu_lengths[i] = 1
        
        del all_imu
        
        # Load poses
        with open(seq_dir / 'poses_quaternion.json', 'r') as f:
            all_poses = json.load(f)
        poses_window = all_poses[start_idx:start_idx + length]
        poses = self._compute_relative_poses(poses_window)
        
        return {
            'images': images,                    # [seq_len, 3, H, W]
            'imu_padded': imu_padded,           # [seq_len-1, max_imu_len, 6]
            'imu_lengths': imu_lengths,         # [seq_len-1]
            'poses': poses,                     # [seq_len-1, 7]
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


def collate_variable_imu_padded(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that maintains compatibility with existing training code.
    Returns both padded tensors (for efficient GPU transfer) and list format (for backward compatibility).
    """
    # Find max sequence length in batch
    max_seq_len = max(sample['sequence_length'] for sample in batch)
    
    # Prepare batched data
    batched_images = []
    batched_imu_padded = []
    batched_imu_lengths = []
    batched_poses = []
    sequence_lengths = []
    
    # Also prepare list format for backward compatibility
    batched_imu_sequences = []
    
    for sample in batch:
        seq_len = sample['sequence_length']
        sequence_lengths.append(seq_len)
        
        # Pad images if needed
        images = sample['images']
        if images.shape[0] < max_seq_len:
            pad_len = max_seq_len - images.shape[0]
            images = F.pad(images, (0, 0, 0, 0, 0, 0, 0, pad_len))
        batched_images.append(images)
        
        # Handle padded IMU data
        imu_padded = sample['imu_padded']
        imu_lengths = sample['imu_lengths']
        
        # Pad to max sequence length
        if imu_padded.shape[0] < max_seq_len - 1:
            pad_len = (max_seq_len - 1) - imu_padded.shape[0]
            imu_padded = F.pad(imu_padded, (0, 0, 0, 0, 0, pad_len))
            imu_lengths = F.pad(imu_lengths, (0, pad_len), value=1)  # Pad with length 1
        
        batched_imu_padded.append(imu_padded)
        batched_imu_lengths.append(imu_lengths)
        
        # Create list format for backward compatibility
        imu_list = []
        for t in range(sample['imu_padded'].shape[0]):
            actual_len = sample['imu_lengths'][t].item()
            imu_list.append(sample['imu_padded'][t, :actual_len])
        # Pad list to max length
        while len(imu_list) < max_seq_len - 1:
            imu_list.append(torch.zeros((1, 6), dtype=torch.float32))
        batched_imu_sequences.append(imu_list)
        
        # Pad poses if needed
        poses = sample['poses']
        if poses.shape[0] < max_seq_len - 1:
            pad_len = (max_seq_len - 1) - poses.shape[0]
            poses = F.pad(poses, (0, 0, 0, pad_len))
        batched_poses.append(poses)
    
    return {
        'images': torch.stack(batched_images),           # [B, T, 3, H, W]
        'imu_padded': torch.stack(batched_imu_padded),  # [B, T-1, max_imu_len, 6]
        'imu_lengths': torch.stack(batched_imu_lengths), # [B, T-1]
        'imu_sequences': batched_imu_sequences,          # List format for compatibility
        'poses': torch.stack(batched_poses),             # [B, T-1, 7]
        'sequence_lengths': torch.tensor(sequence_lengths)
    }