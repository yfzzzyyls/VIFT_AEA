"""Efficient memory-mapped dataset using numpy memmap for Aria data."""

import os
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R


class AriaRawDatasetEfficient(Dataset):
    """
    Truly efficient memory-mapped dataset for Aria data.
    Uses numpy memmap to read only the required frames from disk.
    """
    
    def __init__(self, data_dir, sequence_length=11, stride=10, transform=None):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        
        # Get all sequences
        self.sequences = []
        self.visual_memmaps = {}  # Cache memmap objects
        self.samples = []
        
        print(f"Initializing efficient dataset from {data_dir}")
        
        for seq_dir in sorted(self.data_dir.iterdir()):
            if seq_dir.is_dir() and seq_dir.name.isdigit():
                # Check required files
                visual_path = seq_dir / 'visual_data.npy'
                visual_pt_path = seq_dir / 'visual_data.pt'
                imu_path = seq_dir / 'imu_data.pt'
                poses_path = seq_dir / 'poses_quaternion.json'
                metadata_path = seq_dir / 'metadata.json'
                
                # Check if we need to convert .pt to .npy for efficient access
                if not visual_path.exists() and visual_pt_path.exists():
                    print(f"Converting {seq_dir.name} to numpy format for efficient access...")
                    self._convert_pt_to_npy(visual_pt_path, visual_path)
                
                if visual_path.exists() and imu_path.exists() and poses_path.exists():
                    # Load metadata
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    num_frames = metadata['num_frames']
                    visual_shape = metadata['visual_shape']  # [N, C, H, W]
                    
                    # Create memmap for this sequence
                    self.visual_memmaps[seq_dir] = np.memmap(
                        visual_path,
                        dtype='float32',
                        mode='r',
                        shape=tuple(visual_shape)
                    )
                    
                    # Create samples
                    for start_idx in range(0, num_frames - self.sequence_length + 1, self.stride):
                        self.samples.append({
                            'seq_dir': seq_dir,
                            'seq_name': seq_dir.name,
                            'start_idx': start_idx,
                            'end_idx': start_idx + self.sequence_length,
                        })
                    
                    self.sequences.append(seq_dir)
        
        print(f"Found {len(self.sequences)} sequences")
        print(f"Created {len(self.samples)} samples with window size {sequence_length}")
    
    def _convert_pt_to_npy(self, pt_path, npy_path):
        """Convert PyTorch tensor to numpy array for memmap access."""
        data = torch.load(pt_path, map_location='cpu')
        np.save(npy_path, data.numpy())
        del data
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq_dir = sample['seq_dir']
        start_idx = sample['start_idx']
        end_idx = sample['end_idx']
        
        # Get visual data from memmap (only reads required frames!)
        visual_memmap = self.visual_memmaps[seq_dir]
        window_visual_np = visual_memmap[start_idx:end_idx].copy()  # [11, 3, H, W]
        window_visual = torch.from_numpy(window_visual_np)
        
        # Resize if needed
        if window_visual.shape[-2:] != (704, 704):
            window_visual = F.interpolate(window_visual, size=(704, 704), mode='bilinear', align_corners=False)
        
        # Load IMU data (small file, okay to load fully)
        imu_path = seq_dir / 'imu_data.pt'
        imu_data = torch.load(imu_path, map_location='cpu')
        window_imu = imu_data[start_idx:start_idx + self.sequence_length - 1]
        
        # Stack IMU data
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
        
        # Load poses (small file)
        with open(seq_dir / 'poses_quaternion.json', 'r') as f:
            poses_data = json.load(f)
        window_poses = poses_data[start_idx:end_idx]
        
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
            'seq_name': sample['seq_name'],
            'start_idx': sample['start_idx'],
            'frame_ids': frame_ids
        }