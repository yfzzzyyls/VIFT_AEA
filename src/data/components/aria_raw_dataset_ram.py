"""RAM-based dataset for Aria data - preloads most sequences for fastest training."""

import os
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import OrderedDict


class AriaRawDatasetRAM(Dataset):
    """
    High-performance dataset that preloads most sequences into RAM.
    Designed for systems with sufficient memory (>400GB).
    """
    
    def __init__(self, data_dir, sequence_length=11, stride=10, transform=None, 
                 preload_ratio=0.8, max_workers=8):
        """
        Args:
            data_dir: Directory containing processed Aria sequences
            sequence_length: Number of frames per sequence (default: 11)
            stride: Stride for sliding window (default: 10)
            transform: Optional transforms to apply to images
            preload_ratio: Fraction of sequences to preload (default: 0.8)
            max_workers: Number of parallel workers for loading (default: 8)
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        self.preload_ratio = preload_ratio
        
        print(f"Initializing RAM-based dataset from {data_dir}")
        print(f"Will preload {preload_ratio*100:.0f}% of sequences into RAM")
        
        # Phase 1: Discover and index all sequences
        self.sequences = []
        self.sequence_metadata = {}
        self.samples = []
        
        for seq_dir in sorted(self.data_dir.iterdir()):
            if seq_dir.is_dir() and seq_dir.name.isdigit():
                # Check required files
                visual_path = seq_dir / 'visual_data.pt'
                imu_path = seq_dir / 'imu_data.pt'
                poses_path = seq_dir / 'poses_quaternion.json'
                metadata_path = seq_dir / 'metadata.json'
                
                if all(p.exists() for p in [visual_path, imu_path, poses_path, metadata_path]):
                    # Load metadata
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    self.sequences.append(seq_dir)
                    self.sequence_metadata[seq_dir] = metadata
                    
                    # Create sliding window samples
                    num_frames = metadata['num_frames']
                    for start_idx in range(0, num_frames - self.sequence_length + 1, self.stride):
                        self.samples.append({
                            'seq_dir': seq_dir,
                            'seq_name': seq_dir.name,
                            'start_idx': start_idx,
                            'end_idx': start_idx + self.sequence_length,
                        })
        
        print(f"Found {len(self.sequences)} sequences")
        print(f"Created {len(self.samples)} training samples")
        
        # Phase 2: Determine which sequences to preload
        num_to_preload = int(len(self.sequences) * self.preload_ratio)
        self.preload_sequences = self.sequences[:num_to_preload]
        self.ondemand_sequences = self.sequences[num_to_preload:]
        
        print(f"\nPreloading {len(self.preload_sequences)} sequences into RAM...")
        print(f"Keeping {len(self.ondemand_sequences)} sequences for on-demand loading")
        
        # Phase 3: Preload sequences in parallel
        self.preloaded_visual = {}
        self.preloaded_imu = {}
        self.preloaded_poses = {}
        self._lock = threading.Lock()
        
        self._preload_sequences(max_workers)
        
        # Phase 4: Setup LRU cache for on-demand sequences
        self.lru_cache = OrderedDict()
        self.cache_size = min(10, len(self.ondemand_sequences))  # Cache up to 10 sequences
        
        print(f"\n✓ Dataset ready! Preloaded {len(self.preloaded_visual)} sequences")
        self._print_memory_usage()
    
    def _load_sequence_data(self, seq_dir):
        """Load all data for a sequence."""
        visual_data = torch.load(seq_dir / 'visual_data.pt', map_location='cpu')
        imu_data = torch.load(seq_dir / 'imu_data.pt', map_location='cpu')
        
        with open(seq_dir / 'poses_quaternion.json', 'r') as f:
            poses_data = json.load(f)
        
        return visual_data, imu_data, poses_data
    
    def _preload_sequences(self, max_workers):
        """Preload sequences in parallel with progress bar."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            # Submit all loading tasks
            for seq_dir in self.preload_sequences:
                future = executor.submit(self._load_sequence_data, seq_dir)
                futures[future] = seq_dir
            
            # Process completed tasks with progress bar
            with tqdm(total=len(self.preload_sequences), desc="Loading sequences") as pbar:
                for future in as_completed(futures):
                    seq_dir = futures[future]
                    try:
                        visual_data, imu_data, poses_data = future.result()
                        
                        # Thread-safe storage
                        with self._lock:
                            self.preloaded_visual[seq_dir] = visual_data
                            self.preloaded_imu[seq_dir] = imu_data
                            self.preloaded_poses[seq_dir] = poses_data
                        
                        # Update progress
                        pbar.update(1)
                        pbar.set_postfix({'seq': seq_dir.name})
                        
                    except Exception as e:
                        print(f"\n✗ Error loading {seq_dir}: {e}")
    
    def _get_from_cache(self, seq_dir):
        """Get sequence from LRU cache or load it."""
        if seq_dir in self.lru_cache:
            # Move to end (most recently used)
            self.lru_cache.move_to_end(seq_dir)
            return self.lru_cache[seq_dir]
        
        # Load sequence
        visual_data, imu_data, poses_data = self._load_sequence_data(seq_dir)
        
        # Add to cache
        self.lru_cache[seq_dir] = (visual_data, imu_data, poses_data)
        
        # Remove oldest if cache is full
        if len(self.lru_cache) > self.cache_size:
            self.lru_cache.popitem(last=False)
        
        return visual_data, imu_data, poses_data
    
    def _print_memory_usage(self):
        """Print estimated memory usage."""
        import psutil
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / 1024**3
        print(f"Current process memory: {memory_gb:.1f} GB")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq_dir = sample['seq_dir']
        start_idx = sample['start_idx']
        end_idx = sample['end_idx']
        
        # Get data (from preloaded or cache)
        if seq_dir in self.preloaded_visual:
            visual_data = self.preloaded_visual[seq_dir]
            imu_data = self.preloaded_imu[seq_dir]
            poses_data = self.preloaded_poses[seq_dir]
        else:
            visual_data, imu_data, poses_data = self._get_from_cache(seq_dir)
        
        # Extract window
        window_visual = visual_data[start_idx:end_idx].clone()
        window_imu = imu_data[start_idx:start_idx + self.sequence_length - 1]
        window_poses = poses_data[start_idx:end_idx]
        
        # Resize visual data if needed
        if window_visual.shape[-2:] != (704, 704):
            window_visual = F.interpolate(window_visual, size=(704, 704), mode='bilinear', align_corners=False)
        
        # Process IMU data (variable length)
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
            
            # Compute relative transformation
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