"""
Memory-Mapped Shared Dataset for True Multi-Process Sharing
Uses numpy memory-mapped files that can be shared across processes
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import tempfile
import shutil
import pickle
import time
import torch.distributed as dist
from filelock import FileLock


class AriaDatasetMMapShared(Dataset):
    """
    Memory-mapped dataset that truly shares memory across processes.
    Uses numpy memory-mapped arrays stored on disk (or tmpfs for speed).
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        sequence_length: int = 21,
        stride: int = 5,
        image_size: Tuple[int, int] = (704, 704),
        max_imu_length: int = 400,
        cache_dir: Optional[str] = None,
        rank: int = 0,
        world_size: int = 1,
        **kwargs
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride
        self.image_size = image_size
        self.max_imu_length = max_imu_length
        self.rank = rank
        self.world_size = world_size
        
        # Use tmpfs for fast memory-mapped files (RAM-backed filesystem)
        if cache_dir is None:
            cache_dir = "/dev/shm/aria_mmap_cache"  # Shared memory filesystem
        self.cache_dir = Path(cache_dir) / split
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for memory-mapped files and metadata
        self.metadata_path = self.cache_dir / "metadata.pkl"
        self.lock_path = self.cache_dir / "init.lock"
        
        # Load split information
        splits_file = self.data_dir / 'splits.json'
        if splits_file.exists():
            with open(splits_file, 'r') as f:
                splits_data = json.load(f)
                self.sequence_names = splits_data.get('splits', {}).get(split, [])
        else:
            self.sequence_names = [d.name for d in self.data_dir.iterdir() 
                                 if d.is_dir() and not d.name.startswith('.')]
        
        # Initialize or attach to memory-mapped data
        self._initialize_mmap_data()
        
        # Create sample indices
        self._create_samples()
        
        print(f"[Rank {self.rank}] Ready with {len(self.samples)} samples")
        
    def _initialize_mmap_data(self):
        """Initialize or attach to memory-mapped data with proper synchronization."""
        lock = FileLock(str(self.lock_path))
        
        with lock:
            if self.rank == 0 or not self.metadata_path.exists():
                # First process to arrive creates the data
                if not self.metadata_path.exists():
                    print(f"[Rank {self.rank}] Creating memory-mapped data...")
                    self._create_mmap_data()
                else:
                    print(f"[Rank {self.rank}] Memory-mapped data already exists, attaching...")
                    self._attach_to_mmap_data()
            else:
                # Other ranks wait and attach
                print(f"[Rank {self.rank}] Waiting for memory-mapped data...")
                while not self.metadata_path.exists():
                    time.sleep(0.1)
                time.sleep(0.5)  # Extra delay to ensure files are ready
                self._attach_to_mmap_data()
    
    def _create_mmap_data(self):
        """Create memory-mapped arrays from original data."""
        self.mmap_arrays = {}
        self.metadata = {
            'sequences': {},
            'total_memory': 0
        }
        
        for seq_idx, seq_name in enumerate(self.sequence_names):
            seq_dir = self.data_dir / seq_name
            if not self._is_valid_sequence(seq_dir):
                continue
                
            print(f"Processing sequence {seq_idx+1}/{len(self.sequence_names)}: {seq_name}...", end='', flush=True)
            
            # Load original data
            visual_data = torch.load(seq_dir / 'visual_data.pt', map_location='cpu')
            if visual_data.shape[-2:] != self.image_size:
                visual_data = F.interpolate(visual_data, size=self.image_size, 
                                          mode='bilinear', align_corners=False)
            
            # Create memory-mapped array for visual data
            visual_shape = visual_data.shape
            visual_dtype = visual_data.numpy().dtype
            visual_mmap_path = self.cache_dir / f"{seq_name}_visual.mmap"
            
            # Create and fill memory-mapped array
            visual_mmap = np.memmap(visual_mmap_path, dtype=visual_dtype, 
                                   mode='w+', shape=visual_shape)
            visual_mmap[:] = visual_data.numpy()
            visual_mmap.flush()
            
            # Load IMU data
            imu_data = torch.load(seq_dir / 'imu_data.pt', map_location='cpu')
            
            # Save IMU data as separate arrays (since they're variable length)
            imu_dir = self.cache_dir / f"{seq_name}_imu"
            imu_dir.mkdir(exist_ok=True)
            
            imu_metadata = []
            for i, imu_tensor in enumerate(imu_data):
                if isinstance(imu_tensor, torch.Tensor) and imu_tensor.numel() > 0:
                    imu_path = imu_dir / f"frame_{i:06d}.npy"
                    np.save(imu_path, imu_tensor.numpy())
                    imu_metadata.append({
                        'path': str(imu_path),
                        'shape': list(imu_tensor.shape),
                        'dtype': str(imu_tensor.numpy().dtype)
                    })
                else:
                    imu_metadata.append(None)
            
            # Load poses (keep as JSON since they're small)
            with open(seq_dir / 'poses_quaternion.json', 'r') as f:
                poses_data = json.load(f)
            
            # Save poses
            poses_path = self.cache_dir / f"{seq_name}_poses.json"
            with open(poses_path, 'w') as f:
                json.dump(poses_data, f)
            
            # Store metadata
            seq_memory = visual_data.element_size() * visual_data.nelement()
            self.metadata['sequences'][seq_name] = {
                'visual': {
                    'path': str(visual_mmap_path),
                    'shape': list(visual_shape),
                    'dtype': str(visual_dtype)
                },
                'imu': imu_metadata,
                'poses_path': str(poses_path),
                'memory_gb': seq_memory / 1024**3
            }
            self.metadata['total_memory'] += seq_memory
            
            print(f" done ({seq_memory / 1024**3:.2f} GB)")
            
            # Clean up original tensors
            del visual_data
            del imu_data
        
        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"\nTotal memory-mapped: {self.metadata['total_memory'] / 1024**3:.2f} GB")
        
        # Now attach to the created data
        self._attach_to_mmap_data()
    
    def _attach_to_mmap_data(self):
        """Attach to existing memory-mapped data."""
        # Load metadata
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Open memory-mapped arrays in read mode
        self.mmap_arrays = {}
        self.imu_data = {}
        self.poses_data = {}
        
        for seq_name, seq_meta in self.metadata['sequences'].items():
            # Attach to visual data
            visual_meta = seq_meta['visual']
            self.mmap_arrays[seq_name] = np.memmap(
                visual_meta['path'],
                dtype=visual_meta['dtype'],
                mode='r',  # Read-only
                shape=tuple(visual_meta['shape'])
            )
            
            # Load poses
            with open(seq_meta['poses_path'], 'r') as f:
                self.poses_data[seq_name] = json.load(f)
            
            # Store IMU metadata (load on demand)
            self.imu_data[seq_name] = seq_meta['imu']
        
        print(f"[Rank {self.rank}] Attached to {len(self.mmap_arrays)} sequences")
    
    def _create_samples(self):
        """Create sample indices."""
        self.samples = []
        
        for seq_name in self.mmap_arrays.keys():
            num_frames = self.mmap_arrays[seq_name].shape[0]
            
            for start_idx in range(0, num_frames - self.sequence_length + 1, self.stride):
                self.samples.append({
                    'seq_name': seq_name,
                    'start_idx': start_idx
                })
    
    def _is_valid_sequence(self, seq_dir: Path) -> bool:
        """Check if a sequence directory has all required files."""
        return all((seq_dir / f).exists() for f in 
                  ['visual_data.pt', 'imu_data.pt', 'poses_quaternion.json'])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from memory-mapped data."""
        sample_info = self.samples[idx]
        seq_name = sample_info['seq_name']
        start_idx = sample_info['start_idx']
        end_idx = start_idx + self.sequence_length
        
        # Get visual data from memory-mapped array
        visual_np = self.mmap_arrays[seq_name][start_idx:end_idx]
        # Use as_tensor for potential zero-copy when possible
        images = torch.as_tensor(visual_np, dtype=torch.float32)
        
        # Process IMU data
        imu_padded = torch.zeros(self.sequence_length - 1, self.max_imu_length, 6)
        imu_lengths = torch.zeros(self.sequence_length - 1, dtype=torch.long)
        
        imu_metadata = self.imu_data[seq_name]
        for i in range(self.sequence_length - 1):
            frame_idx = start_idx + i
            if frame_idx < len(imu_metadata) and imu_metadata[frame_idx] is not None:
                # Load IMU data on demand
                imu_meta = imu_metadata[frame_idx]
                imu_np = np.load(imu_meta['path'])
                imu_tensor = torch.from_numpy(imu_np)
                
                actual_len = min(imu_tensor.shape[0], self.max_imu_length)
                imu_padded[i, :actual_len] = imu_tensor[:actual_len]
                imu_lengths[i] = actual_len
            else:
                imu_lengths[i] = 1
        
        # Get poses
        poses_window = self.poses_data[seq_name][start_idx:end_idx]
        poses = self._compute_relative_poses(poses_window)
        
        return {
            'images': images,
            'imu_padded': imu_padded,
            'imu_lengths': imu_lengths,
            'poses': poses,
            'sequence_length': self.sequence_length
        }
    
    def _compute_relative_poses(self, poses_window: List[Dict]) -> torch.Tensor:
        """Compute relative poses between consecutive frames."""
        relative_poses = []
        
        for i in range(len(poses_window) - 1):
            t1 = np.array(poses_window[i]['translation'])
            q1 = np.array(poses_window[i]['quaternion'])
            t2 = np.array(poses_window[i + 1]['translation'])
            q2 = np.array(poses_window[i + 1]['quaternion'])
            
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            
            dt_world = t2 - t1
            dt_local = r1.inv().apply(dt_world)
            
            r_rel = r1.inv() * r2
            q_rel = r_rel.as_quat()
            
            relative_pose = np.concatenate([dt_local, q_rel])
            relative_poses.append(relative_pose)
        
        while len(relative_poses) < self.sequence_length - 1:
            relative_poses.append(relative_poses[-1] if relative_poses else np.zeros(7))
        
        return torch.tensor(np.array(relative_poses), dtype=torch.float32)
    
    def cleanup(self):
        """Clean up memory-mapped files (call this when done training)."""
        if self.rank == 0 and hasattr(self, 'cache_dir') and self.cache_dir.exists():
            print(f"Cleaning up memory-mapped files at {self.cache_dir}")
            shutil.rmtree(self.cache_dir)


def collate_mmap_shared(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for memory-mapped dataset."""
    return {
        'images': torch.stack([item['images'] for item in batch]),
        'imu_padded': torch.stack([item['imu_padded'] for item in batch]),
        'imu_lengths': torch.stack([item['imu_lengths'] for item in batch]),
        'poses': torch.stack([item['poses'] for item in batch]),
        'sequence_lengths': torch.tensor([item['sequence_length'] for item in batch])
    }