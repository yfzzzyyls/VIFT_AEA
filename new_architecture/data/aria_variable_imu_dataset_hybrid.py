"""
Hybrid Aria Dataset - Pre-loads data efficiently like working script
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
import gc


class AriaVariableIMUDataset(Dataset):
    """
    Hybrid dataset that pre-loads data windows efficiently.
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
        
        # Pre-compute all samples with actual data
        self.samples = []
        self._precompute_samples()
        
        print(f"Created {len(self.samples)} pre-computed samples for {split} split")
        
    def _precompute_samples(self):
        """Pre-compute and store sample windows like the working script."""
        for seq_idx, seq_name in enumerate(self.sequence_names):
            seq_dir = self.data_dir / seq_name
            if not self._is_valid_sequence(seq_dir):
                continue
                
            print(f"Processing sequence {seq_idx+1}/{len(self.sequence_names)}: {seq_name}")
            
            # Load the full sequence data
            visual_data = torch.load(seq_dir / 'visual_data.pt', map_location='cpu')
            imu_data = torch.load(seq_dir / 'imu_data.pt', map_location='cpu')
            with open(seq_dir / 'poses_quaternion.json', 'r') as f:
                poses_data = json.load(f)
            
            num_frames = visual_data.shape[0]
            
            # Pre-compute all windows for this sequence
            for start_idx in range(0, num_frames - self.min_seq_len + 1, self.stride):
                # Determine window size
                if self.variable_length:
                    max_len = min(self.max_seq_len, num_frames - start_idx)
                    # Pre-determine the length for consistency
                    window_size = random.randint(self.min_seq_len, max_len)
                else:
                    window_size = min(self.sequence_length, num_frames - start_idx)
                
                if start_idx + window_size > num_frames:
                    continue
                
                # Extract and store the actual window data
                window_visual = visual_data[start_idx:start_idx + window_size].clone()
                
                # Resize if needed
                if window_visual.shape[-2:] != self.image_size:
                    window_visual = F.interpolate(
                        window_visual, 
                        size=self.image_size,
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Extract IMU window
                window_imu = []
                for i in range(window_size - 1):
                    imu_idx = start_idx + i
                    if imu_idx < len(imu_data):
                        window_imu.append(imu_data[imu_idx].clone())
                
                # Extract poses window
                window_poses = poses_data[start_idx:start_idx + window_size]
                
                # Pre-compute relative poses
                relative_poses = self._compute_relative_poses(window_poses)
                
                # Store everything
                self.samples.append({
                    'images': window_visual,
                    'imu_sequences': window_imu,
                    'poses': relative_poses,
                    'sequence_length': window_size
                })
            
            # Free memory after processing each sequence
            del visual_data
            del imu_data
            gc.collect()
    
    def _is_valid_sequence(self, seq_dir: Path) -> bool:
        """Check if a sequence directory has all required files."""
        return all((seq_dir / f).exists() for f in 
                  ['visual_data.pt', 'imu_data.pt', 'poses_quaternion.json'])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Simply return the pre-computed sample."""
        return self.samples[idx]
    
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