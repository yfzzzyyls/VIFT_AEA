#!/usr/bin/env python3
"""
Shared Memory Dataset for Distributed Training
Only rank 0 loads data, other ranks attach to shared memory.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import torch.distributed as dist
import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple
import pickle
import time


class AriaSharedMemoryDataset(Dataset):
    """
    Dataset that loads data only once in rank 0 and shares via shared memory.
    Other ranks attach to the same shared memory without duplicating data.
    """
    
    def __init__(
        self, 
        data_dir: str, 
        sequence_length: int = 11, 
        stride: int = 10,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        verbose: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.verbose = verbose
        
        # Get rank info
        if rank is None:
            self.rank = dist.get_rank() if dist.is_initialized() else 0
        else:
            self.rank = rank
            
        if world_size is None:
            self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        else:
            self.world_size = world_size
        
        # Shared memory handles
        self.shm_visual_list = []
        self.shm_imu_list = []
        self.shm_metadata = None
        
        # Load or attach to data
        if self.rank == 0:
            self._load_and_share_data()
        else:
            self._attach_to_shared_data()
            
        # All ranks should have the same samples info
        self._setup_samples()
        
    def _print(self, msg: str):
        """Print with rank prefix."""
        if self.verbose:
            print(f"[Rank {self.rank}] {msg}")
            
    def _load_and_share_data(self):
        """Rank 0: Load all data and put in shared memory."""
        self._print("Loading dataset and creating shared memory...")
        start_time = time.time()
        
        # Find all sequences
        sequences = []
        for seq_dir in sorted(self.data_dir.iterdir()):
            if seq_dir.is_dir() and seq_dir.name.isdigit():
                visual_path = seq_dir / 'visual_data.pt'
                imu_path = seq_dir / 'imu_data.pt'
                poses_path = seq_dir / 'poses_quaternion.json'
                
                if visual_path.exists() and imu_path.exists() and poses_path.exists():
                    sequences.append(seq_dir)
        
        self._print(f"Found {len(sequences)} sequences")
        
        # Storage for all data
        self.visual_data_list = []
        self.imu_data_list = []
        self.poses_data_list = []
        self.sequence_info = []
        
        # Load each sequence
        for seq_idx, seq_dir in enumerate(sequences):
            if seq_idx % 10 == 0:
                self._print(f"Loading sequence {seq_idx+1}/{len(sequences)}...")
                
            # Load visual data
            visual_data = torch.load(seq_dir / 'visual_data.pt', map_location='cpu')
            
            # Load IMU data 
            imu_data = torch.load(seq_dir / 'imu_data.pt', map_location='cpu')
            
            # Load poses
            with open(seq_dir / 'poses_quaternion.json', 'r') as f:
                poses_data = json.load(f)
            
            # Store sequence info
            self.sequence_info.append({
                'name': seq_dir.name,
                'num_frames': visual_data.shape[0],
                'visual_shape': visual_data.shape,
                'num_imu_intervals': len(imu_data) if isinstance(imu_data, list) else imu_data.shape[0]
            })
            
            # Create shared memory for visual data
            shm_visual = shared_memory.SharedMemory(
                create=True, 
                size=visual_data.nbytes,
                name=f"aria_visual_{self.rank}_{seq_idx}"
            )
            # Copy data to shared memory
            shm_visual_tensor = torch.from_numpy(
                np.ndarray(visual_data.shape, dtype=np.float32, buffer=shm_visual.buf)
            )
            shm_visual_tensor.copy_(visual_data)
            
            self.shm_visual_list.append(shm_visual)
            self.visual_data_list.append(visual_data)
            
            # For IMU data (variable length list), we need to handle it differently
            # Convert to padded tensor for shared memory
            if isinstance(imu_data, list):
                max_len = max(tensor.shape[0] for tensor in imu_data)
                imu_padded = torch.zeros((len(imu_data), max_len, 6), dtype=torch.float32)
                imu_lengths = torch.zeros(len(imu_data), dtype=torch.long)
                
                for i, tensor in enumerate(imu_data):
                    imu_padded[i, :tensor.shape[0]] = tensor
                    imu_lengths[i] = tensor.shape[0]
                    
                # Create shared memory for padded IMU data
                shm_imu = shared_memory.SharedMemory(
                    create=True,
                    size=imu_padded.nbytes + imu_lengths.nbytes,
                    name=f"aria_imu_{self.rank}_{seq_idx}"
                )
                
                # Store both padded data and lengths
                imu_buffer = np.ndarray(
                    (imu_padded.numel() + imu_lengths.numel(),), 
                    dtype=np.float32, 
                    buffer=shm_imu.buf
                )
                imu_buffer[:imu_padded.numel()] = imu_padded.numpy().flatten()
                imu_buffer[imu_padded.numel():] = imu_lengths.numpy().astype(np.float32)
                
                self.shm_imu_list.append(shm_imu)
                self.imu_data_list.append((imu_padded, imu_lengths))
            else:
                # Old format - just store as tensor
                shm_imu = shared_memory.SharedMemory(
                    create=True,
                    size=imu_data.nbytes,
                    name=f"aria_imu_{self.rank}_{seq_idx}"
                )
                shm_imu_tensor = torch.from_numpy(
                    np.ndarray(imu_data.shape, dtype=np.float32, buffer=shm_imu.buf)
                )
                shm_imu_tensor.copy_(imu_data)
                
                self.shm_imu_list.append(shm_imu)
                self.imu_data_list.append(imu_data)
            
            self.poses_data_list.append(poses_data)
        
        # Create metadata shared memory
        metadata = {
            'sequence_info': self.sequence_info,
            'poses_data_list': self.poses_data_list,
            'num_sequences': len(sequences)
        }
        metadata_bytes = pickle.dumps(metadata)
        
        self.shm_metadata = shared_memory.SharedMemory(
            create=True,
            size=len(metadata_bytes),
            name=f"aria_metadata_{self.rank}"
        )
        self.shm_metadata.buf[:] = metadata_bytes
        
        load_time = time.time() - start_time
        self._print(f"Data loading complete in {load_time:.1f}s")
        
        # Synchronize with other ranks
        if dist.is_initialized():
            dist.barrier()
            
    def _attach_to_shared_data(self):
        """Other ranks: Attach to shared memory created by rank 0."""
        self._print("Waiting for rank 0 to load data...")
        
        # Wait for rank 0 to finish loading
        if dist.is_initialized():
            dist.barrier()
            
        self._print("Attaching to shared memory...")
        
        # First get metadata
        self.shm_metadata = shared_memory.SharedMemory(name=f"aria_metadata_0")
        metadata = pickle.loads(bytes(self.shm_metadata.buf))
        
        self.sequence_info = metadata['sequence_info']
        self.poses_data_list = metadata['poses_data_list']
        num_sequences = metadata['num_sequences']
        
        # Attach to visual and IMU data
        self.visual_data_list = []
        self.imu_data_list = []
        
        for seq_idx in range(num_sequences):
            seq_info = self.sequence_info[seq_idx]
            
            # Attach to visual shared memory
            shm_visual = shared_memory.SharedMemory(name=f"aria_visual_0_{seq_idx}")
            visual_data = torch.from_numpy(
                np.ndarray(
                    seq_info['visual_shape'], 
                    dtype=np.float32, 
                    buffer=shm_visual.buf
                )
            )
            self.shm_visual_list.append(shm_visual)
            self.visual_data_list.append(visual_data)
            
            # Attach to IMU shared memory
            shm_imu = shared_memory.SharedMemory(name=f"aria_imu_0_{seq_idx}")
            
            if seq_info['num_imu_intervals'] > 0:
                # Reconstruct padded tensor and lengths
                num_intervals = seq_info['num_imu_intervals']
                # Estimate max length from buffer size
                total_elements = shm_imu.size // 4  # float32 = 4 bytes
                max_len = (total_elements - num_intervals) // (num_intervals * 6)
                
                imu_buffer = np.ndarray(
                    (total_elements,), 
                    dtype=np.float32, 
                    buffer=shm_imu.buf
                )
                
                imu_padded = torch.from_numpy(
                    imu_buffer[:num_intervals * max_len * 6].reshape(num_intervals, max_len, 6)
                )
                imu_lengths = torch.from_numpy(
                    imu_buffer[num_intervals * max_len * 6:].astype(np.int64)[:num_intervals]
                )
                
                self.shm_imu_list.append(shm_imu)
                self.imu_data_list.append((imu_padded, imu_lengths))
            
        self._print("Successfully attached to shared memory")
        
    def _setup_samples(self):
        """Create sample indices for sliding windows."""
        self.samples = []
        
        for seq_idx, seq_info in enumerate(self.sequence_info):
            num_frames = seq_info['num_frames']
            
            # Create sliding windows
            for start_idx in range(0, num_frames - self.sequence_length + 1, self.stride):
                self.samples.append({
                    'seq_idx': seq_idx,
                    'start_idx': start_idx,
                    'seq_name': seq_info['name']
                })
                
        self._print(f"Created {len(self.samples)} samples")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        seq_idx = sample_info['seq_idx']
        start_idx = sample_info['start_idx']
        end_idx = start_idx + self.sequence_length
        
        # Get visual data window
        visual_data = self.visual_data_list[seq_idx]
        window_visual = visual_data[start_idx:end_idx]  # [11, 3, H, W]
        
        # Resize if needed
        if window_visual.shape[-2:] != (704, 704):
            window_visual = F.interpolate(
                window_visual, 
                size=(704, 704), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Get IMU data window
        imu_data_info = self.imu_data_list[seq_idx]
        if isinstance(imu_data_info, tuple):
            # New format with padding
            imu_padded, imu_lengths = imu_data_info
            window_imu_padded = imu_padded[start_idx:start_idx + self.sequence_length - 1]
            window_imu_lengths = imu_lengths[start_idx:start_idx + self.sequence_length - 1]
            
            # Reconstruct list of variable length tensors
            window_imu = []
            for i in range(len(window_imu_lengths)):
                length = int(window_imu_lengths[i].item())
                window_imu.append(window_imu_padded[i, :length])
        else:
            # Old format
            window_imu = imu_data_info[start_idx:start_idx + self.sequence_length - 1]
        
        # Get poses
        poses_data = self.poses_data_list[seq_idx]
        window_poses = poses_data[start_idx:end_idx]
        
        # Convert to relative poses
        gt_poses = []
        for i in range(len(window_poses) - 1):
            t1 = np.array(window_poses[i]['translation'])
            q1 = np.array(window_poses[i]['quaternion'])
            t2 = np.array(window_poses[i + 1]['translation'])
            q2 = np.array(window_poses[i + 1]['quaternion'])
            
            # Compute relative transformation
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            
            dt_world = t2 - t1
            dt_local = r1.inv().apply(dt_world)
            
            r_rel = r1.inv() * r2
            q_rel = r_rel.as_quat()
            
            gt_poses.append(np.concatenate([dt_local, q_rel]))
        
        gt_poses = torch.tensor(np.array(gt_poses), dtype=torch.float32)
        
        # Process IMU for batching
        if isinstance(window_imu, list):
            max_len = max(tensor.shape[0] for tensor in window_imu)
            imu_padded = []
            for tensor in window_imu:
                if tensor.shape[0] < max_len:
                    padding = torch.zeros(max_len - tensor.shape[0], 6)
                    padded = torch.cat([tensor, padding], dim=0)
                else:
                    padded = tensor
                imu_padded.append(padded)
            imu_data = torch.stack(imu_padded)
        else:
            imu_data = window_imu
        
        return {
            'images': window_visual.clone(),  # Clone to ensure contiguous
            'imu': imu_data,
            'gt_poses': gt_poses,
            'seq_name': sample_info['seq_name'],
            'start_idx': start_idx
        }
    
    def cleanup(self):
        """Clean up shared memory (only rank 0 should unlink)."""
        if self.rank == 0:
            self._print("Cleaning up shared memory...")
            for shm in self.shm_visual_list:
                shm.close()
                shm.unlink()
            for shm in self.shm_imu_list:
                shm.close()
                shm.unlink()
            if self.shm_metadata:
                self.shm_metadata.close()
                self.shm_metadata.unlink()
        else:
            # Other ranks just close without unlinking
            for shm in self.shm_visual_list:
                shm.close()
            for shm in self.shm_imu_list:
                shm.close()
            if self.shm_metadata:
                self.shm_metadata.close()
                
    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass