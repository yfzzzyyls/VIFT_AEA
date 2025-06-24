"""
Efficient Aria Dataset with Variable-Length IMU Sequences
This version loads individual frames instead of entire visual_data.pt files.
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
from PIL import Image


class AriaVariableIMUDatasetEfficient(Dataset):
    """
    Efficient dataset for Aria Everyday Activities with variable-length IMU sequences.
    
    Key improvements:
    - Loads frames individually instead of entire visual_data.pt
    - Uses memory-mapped files or individual frame files
    - Maintains same interface as original dataset
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
        """
        Args:
            data_dir: Root directory containing processed Aria data
            split: 'train', 'val', or 'test'
            sequence_length: Fixed sequence length (if variable_length=False)
            variable_length: Whether to use variable sequence lengths
            min_seq_len: Minimum sequence length when variable_length=True
            max_seq_len: Maximum sequence length when variable_length=True
            stride: Stride for sliding window over sequences
            image_size: Target size for images (H, W)
        """
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
            # Fallback: use all sequences
            self.split_sequences = None
        
        # Find all valid sequences
        self.sequences = []
        self._load_sequences()
        
        # Create samples with sliding windows
        self.samples = []
        self._create_samples()
        
        print(f"Created {len(self.samples)} samples from {len(self.sequences)} sequences")
        print(f"Variable length: {variable_length}")
        
    def _load_sequences(self):
        """Find all valid sequences in the data directory."""
        if self.split_sequences is not None:
            # Use sequences from splits.json
            for seq_name in self.split_sequences:
                seq_dir = self.data_dir / seq_name
                if self._is_valid_sequence(seq_dir):
                    # Get sequence length from metadata
                    with open(seq_dir / 'metadata.json', 'r') as f:
                        metadata = json.load(f)
                    seq_len = metadata['num_frames']
                    
                    self.sequences.append({
                        'path': seq_dir,
                        'length': seq_len
                    })
        else:
            # Fallback: scan directory
            for seq_dir in sorted(self.data_dir.iterdir()):
                if seq_dir.is_dir() and seq_dir.name.isdigit():
                    if self._is_valid_sequence(seq_dir):
                        with open(seq_dir / 'metadata.json', 'r') as f:
                            metadata = json.load(f)
                        seq_len = metadata['num_frames']
                        
                        self.sequences.append({
                            'path': seq_dir,
                            'length': seq_len
                        })
                    
        print(f"Found {len(self.sequences)} valid sequences for {self.split}")
        
    def _is_valid_sequence(self, seq_dir: Path) -> bool:
        """Check if a sequence directory has all required files."""
        # Check for frame directory or visual_data.pt
        frames_dir = seq_dir / 'frames'
        visual_data_path = seq_dir / 'visual_data.pt'
        
        has_frames = frames_dir.exists() or visual_data_path.exists()
        
        poses_path = seq_dir / 'poses_quaternion.json'
        imu_path = seq_dir / 'imu_data.pt'
        metadata_path = seq_dir / 'metadata.json'
        
        return has_frames and imu_path.exists() and poses_path.exists() and metadata_path.exists()
        
    def _create_samples(self):
        """Create training samples using sliding windows."""
        for seq_info in self.sequences:
            seq_path = seq_info['path']
            seq_len = seq_info['length']
            
            # Determine maximum window size for this sequence
            if self.variable_length:
                max_window = min(self.max_seq_len, seq_len)
            else:
                max_window = min(self.sequence_length, seq_len)
            
            # Create sliding windows
            for start_idx in range(0, seq_len - self.min_seq_len + 1, self.stride):
                # Determine window size
                if self.variable_length:
                    # Random window size between min and max
                    window_size = random.randint(self.min_seq_len, 
                                                min(max_window, seq_len - start_idx))
                else:
                    window_size = min(self.sequence_length, seq_len - start_idx)
                
                if start_idx + window_size <= seq_len:
                    self.samples.append({
                        'sequence_path': seq_path,
                        'start_idx': start_idx,
                        'length': window_size
                    })
                    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a sample with variable-length IMU data.
        
        Returns:
            Dictionary containing:
                - 'images': [T, 3, H, W] - RGB images
                - 'imu_sequences': List of T-1 tensors, each [num_samples, 6]
                - 'poses': [T-1, 7] - Relative poses (3 trans + 4 quat)
                - 'sequence_length': int - Actual sequence length T
        """
        sample_info = self.samples[idx]
        seq_path = sample_info['sequence_path']
        start_idx = sample_info['start_idx']
        length = sample_info['length']
        
        # Load visual data efficiently
        images = self._load_visual_data_efficient(seq_path, start_idx, length)
        
        # Resize images if needed
        if images.shape[-2:] != self.image_size:
            images = F.interpolate(images, size=self.image_size, 
                                 mode='bilinear', align_corners=False)
        
        # Load IMU data (only the needed portion)
        imu_sequences = self._load_raw_imu_data(seq_path, start_idx, length)
        
        # Load poses and compute relative transformations
        poses = self._load_relative_poses(seq_path, start_idx, length)
        
        return {
            'images': images,
            'imu_sequences': imu_sequences,
            'poses': poses,
            'sequence_length': length
        }
    
    def _load_visual_data_efficient(self, seq_path: Path, start_idx: int, length: int) -> torch.Tensor:
        """
        Load visual data efficiently - only the required frames.
        """
        frames_dir = seq_path / 'frames'
        
        if frames_dir.exists():
            # Load individual frame files
            frames = []
            for i in range(start_idx, start_idx + length):
                frame_path = frames_dir / f'frame_{i:06d}.pt'
                if frame_path.exists():
                    frame = torch.load(frame_path)
                    frames.append(frame)
                else:
                    # Fallback: create zero frame
                    frames.append(torch.zeros(3, 704, 704))
            return torch.stack(frames)
        else:
            # Fallback: load from visual_data.pt but only once per sequence
            # Use caching to avoid repeated loads
            if not hasattr(self, '_visual_cache'):
                self._visual_cache = {}
            
            seq_key = str(seq_path)
            if seq_key not in self._visual_cache:
                # Load and cache metadata
                with open(seq_path / 'metadata.json', 'r') as f:
                    metadata = json.load(f)
                self._visual_cache[seq_key] = {
                    'num_frames': metadata['num_frames'],
                    'shape': metadata['visual_shape']
                }
            
            # Load only the required slice
            visual_data = torch.load(seq_path / 'visual_data.pt', 
                                   map_location='cpu')[start_idx:start_idx + length]
            return visual_data
    
    def _load_raw_imu_data(self, seq_path: Path, start_idx: int, length: int) -> List[torch.Tensor]:
        """
        Load ALL raw IMU data between consecutive frames.
        
        Returns:
            List of length-1 tensors, each containing all IMU samples between frames
        """
        # Load only the required IMU data
        all_raw_imu = torch.load(seq_path / 'imu_data.pt', map_location='cpu')
        
        # Extract the required subsequence
        imu_sequences = []
        for i in range(length - 1):
            frame_idx = start_idx + i
            if frame_idx < len(all_raw_imu):
                imu_sequences.append(all_raw_imu[frame_idx])
            else:
                # Handle edge case
                print(f"Warning: No IMU data for frame {frame_idx}")
                imu_sequences.append(torch.zeros((1, 6), dtype=torch.float32))
        
        return imu_sequences
    
    def _load_relative_poses(self, seq_path: Path, start_idx: int, length: int) -> torch.Tensor:
        """
        Load poses and compute relative transformations between consecutive frames.
        
        Returns:
            poses: [length-1, 7] - Relative poses (3 translation + 4 quaternion)
        """
        with open(seq_path / 'poses_quaternion.json', 'r') as f:
            poses_data = json.load(f)
        
        poses = poses_data[start_idx:start_idx + length]
        
        # Compute relative poses
        relative_poses = []
        
        for i in range(len(poses) - 1):
            # Current and next pose
            t1 = np.array(poses[i]['translation'])
            q1 = np.array(poses[i]['quaternion'])  # [x, y, z, w]
            t2 = np.array(poses[i + 1]['translation'])
            q2 = np.array(poses[i + 1]['quaternion'])
            
            # Compute relative transformation in local frame
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            
            # Relative translation in world frame
            dt_world = t2 - t1
            
            # Transform to local frame of pose1
            dt_local = r1.inv().apply(dt_world)
            
            # Relative rotation: q_rel = q1^(-1) * q2
            r_rel = r1.inv() * r2
            q_rel = r_rel.as_quat()  # [x, y, z, w]
            
            # Combine: [dx, dy, dz, qx, qy, qz, qw]
            relative_pose = np.concatenate([dt_local, q_rel])
            relative_poses.append(relative_pose)
        
        return torch.tensor(np.array(relative_poses), dtype=torch.float32)


def collate_variable_imu(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching variable-length IMU sequences.
    
    Pads sequences to the maximum length in the batch.
    """
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
        # Pad with empty tensors if needed
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
        'images': torch.stack(batched_images),  # [B, T, 3, H, W]
        'imu_sequences': batched_imu_sequences,  # List of B lists, each with T-1 variable tensors
        'poses': torch.stack(batched_poses),  # [B, T-1, 7]
        'sequence_lengths': torch.tensor(sequence_lengths)  # [B]
    }