"""Ultra-fast lazy loading dataset for Aria data."""

import os
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R


class AriaLazyDataset(Dataset):
    """
    Lazy loading dataset that's extremely fast to initialize.
    Only loads data when actually needed.
    """
    
    def __init__(self, data_dir, sequence_length=11, stride=10, transform=None):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        
        # Quick initialization - just get sequence info
        self.sequences = []
        self.sequence_frames = {}
        self.cumulative_samples = [0]  # Cumulative sum for indexing
        total_samples = 0
        
        print(f"Quick initialization of dataset from {data_dir}")
        
        # Fast scan - only read metadata files
        for seq_dir in sorted(self.data_dir.iterdir()):
            if seq_dir.is_dir() and seq_dir.name.isdigit():
                metadata_path = seq_dir / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    num_frames = metadata['num_frames']
                    num_samples = (num_frames - sequence_length) // stride + 1
                    
                    if num_samples > 0:
                        self.sequences.append(seq_dir)
                        self.sequence_frames[seq_dir] = num_frames
                        total_samples += num_samples
                        self.cumulative_samples.append(total_samples)
        
        self.total_samples = total_samples
        print(f"Found {len(self.sequences)} sequences, {total_samples} samples total")
        
        # Cache for recently loaded sequences
        self._cache = {}
        self._cache_size = 5  # Keep 5 sequences in memory
    
    def __len__(self):
        return self.total_samples
    
    def _get_sample_info(self, idx):
        """Convert global index to (sequence, local_index)."""
        # Binary search to find which sequence this index belongs to
        seq_idx = np.searchsorted(self.cumulative_samples, idx, side='right') - 1
        local_idx = idx - self.cumulative_samples[seq_idx]
        
        seq_dir = self.sequences[seq_idx]
        num_frames = self.sequence_frames[seq_dir]
        
        # Convert local index to start frame
        start_idx = local_idx * self.stride
        end_idx = start_idx + self.sequence_length
        
        return seq_dir, start_idx, end_idx
    
    def _load_sequence_data(self, seq_dir):
        """Load data for a sequence with caching."""
        if seq_dir in self._cache:
            return self._cache[seq_dir]
        
        # Load data
        visual_data = torch.load(seq_dir / 'visual_data.pt', map_location='cpu')
        imu_data = torch.load(seq_dir / 'imu_data.pt', map_location='cpu')
        with open(seq_dir / 'poses_quaternion.json', 'r') as f:
            poses_data = json.load(f)
        
        # Add to cache
        self._cache[seq_dir] = (visual_data, imu_data, poses_data)
        
        # Evict oldest if cache is full
        if len(self._cache) > self._cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        
        return visual_data, imu_data, poses_data
    
    def __getitem__(self, idx):
        # Get sample info
        seq_dir, start_idx, end_idx = self._get_sample_info(idx)
        
        # Load sequence data (with caching)
        visual_data, imu_data, poses_data = self._load_sequence_data(seq_dir)
        
        # Extract windows
        window_visual = visual_data[start_idx:end_idx].clone()
        window_imu = imu_data[start_idx:start_idx + self.sequence_length - 1]
        window_poses = poses_data[start_idx:end_idx]
        
        # Resize visual data if needed
        if window_visual.shape[-2:] != (704, 704):
            window_visual = F.interpolate(window_visual, size=(704, 704), mode='bilinear', align_corners=False)
        
        # Process IMU data
        max_len = max(tensor.shape[0] for tensor in window_imu)
        imu_padded = []
        for tensor in window_imu:
            if tensor.shape[0] < max_len:
                padding = torch.zeros(max_len - tensor.shape[0], 6)
                padded = torch.cat([tensor, padding], dim=0)
            else:
                padded = tensor
            imu_padded.append(padded)
        imu_tensor = torch.stack(imu_padded)
        
        # Convert to relative poses
        gt_poses = []
        for i in range(len(window_poses) - 1):
            t1 = np.array(window_poses[i]['translation'])
            q1 = np.array(window_poses[i]['quaternion'])
            t2 = np.array(window_poses[i + 1]['translation'])
            q2 = np.array(window_poses[i + 1]['quaternion'])
            
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            
            dt_world = t2 - t1
            dt_local = r1.inv().apply(dt_world)
            
            r_rel = r1.inv() * r2
            q_rel = r_rel.as_quat()
            
            gt_poses.append(np.concatenate([dt_local, q_rel]))
        
        gt_poses = torch.tensor(np.array(gt_poses), dtype=torch.float32)
        
        # Generate frame IDs
        frame_ids = torch.arange(start_idx, start_idx + self.sequence_length, dtype=torch.long)
        
        return {
            'images': window_visual,
            'imu': imu_tensor,
            'gt_poses': gt_poses,
            'seq_name': seq_dir.name,
            'start_idx': start_idx,
            'frame_ids': frame_ids
        }