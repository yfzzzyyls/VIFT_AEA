"""
Debug version of Aria Dataset with Variable-Length IMU Sequences
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
import psutil
import traceback


class AriaVariableIMUDataset(Dataset):
    """
    Dataset for Aria Everyday Activities with variable-length IMU sequences.
    Debug version with memory tracking.
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
        
        # Debug: Print initialization
        print(f"[Dataset Init] PID: {os.getpid()}, Split: {split}, Memory: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
        
        # Load split information
        splits_file = self.data_dir / 'splits.json'
        if splits_file.exists():
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            self.split_sequences = splits['splits'][split]
        else:
            self.split_sequences = None
        
        # Find all valid sequences
        self.sequences = []
        self._load_sequences()
        
        # Create samples with sliding windows
        self.samples = []
        self._create_samples()
        
        print(f"[Dataset Init Complete] PID: {os.getpid()}, Created {len(self.samples)} samples from {len(self.sequences)} sequences")
        print(f"[Dataset Init Complete] Memory: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
        
    def _load_sequences(self):
        """Find all valid sequences in the data directory."""
        if self.split_sequences is not None:
            for seq_name in self.split_sequences:
                seq_dir = self.data_dir / seq_name
                if self._is_valid_sequence(seq_dir):
                    visual_data = torch.load(seq_dir / 'visual_data.pt')
                    seq_len = visual_data.shape[0]
                    
                    self.sequences.append({
                        'path': seq_dir,
                        'length': seq_len
                    })
        else:
            for seq_dir in sorted(self.data_dir.iterdir()):
                if seq_dir.is_dir() and seq_dir.name.isdigit():
                    if self._is_valid_sequence(seq_dir):
                        visual_data = torch.load(seq_dir / 'visual_data.pt')
                        seq_len = visual_data.shape[0]
                        
                        self.sequences.append({
                            'path': seq_dir,
                            'length': seq_len
                        })
                    
        print(f"Found {len(self.sequences)} valid sequences for {self.split}")
        
    def _is_valid_sequence(self, seq_dir: Path) -> bool:
        """Check if a sequence directory has all required files."""
        visual_path = seq_dir / 'visual_data.pt'
        poses_path = seq_dir / 'poses_quaternion.json'
        imu_path = seq_dir / 'imu_data.pt'
        
        return visual_path.exists() and imu_path.exists() and poses_path.exists()
        
    def _create_samples(self):
        """Create training samples using sliding windows."""
        for seq_info in self.sequences:
            seq_path = seq_info['path']
            seq_len = seq_info['length']
            
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
                        'sequence_path': seq_path,
                        'start_idx': start_idx,
                        'length': window_size
                    })
                    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a sample with variable-length IMU data."""
        try:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info else -1
            
            print(f"\n[Dataset.__getitem__] PID: {os.getpid()}, Worker: {worker_id}, Idx: {idx}")
            print(f"[Memory Before] {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
            
            sample_info = self.samples[idx]
            seq_path = sample_info['sequence_path']
            start_idx = sample_info['start_idx']
            length = sample_info['length']
            
            print(f"[Loading] Seq: {seq_path.name}, Start: {start_idx}, Length: {length}")
            
            # Load visual data
            print(f"[Loading Visual] File: {seq_path / 'visual_data.pt'}")
            visual_data = torch.load(seq_path / 'visual_data.pt')
            print(f"[Visual Loaded] Shape: {visual_data.shape}, Memory: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
            
            images = visual_data[start_idx:start_idx + length]
            
            # Clean up visual data immediately
            del visual_data
            torch.cuda.empty_cache()
            
            # Resize images if needed
            if images.shape[-2:] != self.image_size:
                images = F.interpolate(images, size=self.image_size, 
                                     mode='bilinear', align_corners=False)
            
            # Load IMU data
            imu_sequences = self._load_raw_imu_data(seq_path, start_idx, length)
            
            # Load poses
            poses = self._load_relative_poses(seq_path, start_idx, length)
            
            print(f"[Memory After] {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
            
            return {
                'images': images,
                'imu_sequences': imu_sequences,
                'poses': poses,
                'sequence_length': length
            }
            
        except Exception as e:
            print(f"\n[ERROR in __getitem__] PID: {os.getpid()}, Worker: {worker_id}, Idx: {idx}")
            print(f"[ERROR] {str(e)}")
            traceback.print_exc()
            raise
    
    def _load_raw_imu_data(self, seq_path: Path, start_idx: int, length: int) -> List[torch.Tensor]:
        """Load ALL raw IMU data between consecutive frames."""
        all_raw_imu = torch.load(seq_path / 'imu_data.pt')
        
        imu_sequences = []
        for i in range(length - 1):
            frame_idx = start_idx + i
            if frame_idx < len(all_raw_imu):
                imu_sequences.append(all_raw_imu[frame_idx])
            else:
                print(f"Warning: No IMU data for frame {frame_idx}")
                imu_sequences.append(torch.zeros((1, 6), dtype=torch.float32))
        
        return imu_sequences
    
    def _load_relative_poses(self, seq_path: Path, start_idx: int, length: int) -> torch.Tensor:
        """Load poses and compute relative transformations between consecutive frames."""
        with open(seq_path / 'poses_quaternion.json', 'r') as f:
            poses_data = json.load(f)
        
        poses = poses_data[start_idx:start_idx + length]
        
        # Compute relative poses
        relative_poses = []
        
        for i in range(len(poses) - 1):
            # Current and next pose
            t1 = np.array(poses[i]['translation'])
            q1 = np.array(poses[i]['quaternion'])
            t2 = np.array(poses[i + 1]['translation'])
            q2 = np.array(poses[i + 1]['quaternion'])
            
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
    print(f"\n[Collate] PID: {os.getpid()}, Batch size: {len(batch)}")
    print(f"[Collate Memory] {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
    
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