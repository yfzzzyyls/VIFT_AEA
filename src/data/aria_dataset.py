"""
AriaEveryday Dataset for VIFT
Custom dataset class that loads processed AriaEveryday data in VIFT-compatible format
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from torch.utils.data import Dataset

from src.data.kitti_dataset import KittiDataset

class AriaEverydayDataset(KittiDataset):
    """
    AriaEveryday dataset for VIFT training
    Inherits from KittiDataset and adapts for AriaEveryday data format
    """
    
    def __init__(
        self,
        data_root: str,
        train_list: str,
        val_list: str,
        seq_len: int = 11,
        imu_dim: int = 6,
        img_dim: int = 512,
        mode: str = "train",
        **kwargs
    ):
        # Don't call parent __init__ directly, implement our own initialization
        self.data_root = Path(data_root)
        self.seq_len = seq_len
        self.imu_dim = imu_dim
        self.img_dim = img_dim
        self.mode = mode
        
        # Load sequence lists
        if mode == "train":
            self.sequences = self._load_sequence_list(train_list)
        else:
            self.sequences = self._load_sequence_list(val_list)
        
        # Build sample list
        self.samples = self._build_sample_list()
        
        print(f"📊 AriaEveryday Dataset ({mode}): {len(self.samples)} samples from {len(self.sequences)} sequences")
    
    def _load_sequence_list(self, list_file: str) -> List[str]:
        """Load sequence IDs from text file"""
        sequences = []
        with open(list_file, 'r') as f:
            for line in f:
                seq_id = line.strip()
                if seq_id:
                    sequences.append(seq_id)
        return sequences
    
    def _build_sample_list(self) -> List[Dict]:
        """Build list of valid samples"""
        samples = []
        
        for seq_id in self.sequences:
            seq_dir = self.data_root / seq_id
            
            if not seq_dir.exists():
                print(f"⚠️ Sequence directory not found: {seq_dir}")
                continue
            
            # Load metadata to get frame count
            metadata_file = seq_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                num_frames = metadata.get('num_frames', 0)
            else:
                # Fallback: count files
                num_frames = len(list(seq_dir.glob("*.pt"))) // 2  # visual + imu files
            
            # Create samples with sliding window
            for start_frame in range(0, max(1, num_frames - self.seq_len + 1)):
                end_frame = start_frame + self.seq_len
                if end_frame <= num_frames:
                    samples.append({
                        'sequence_id': seq_id,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'seq_dir': seq_dir
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get a sample (sequence of frames)"""
        sample_info = self.samples[index]
        
        seq_dir = sample_info['seq_dir']
        start_frame = sample_info['start_frame']
        end_frame = sample_info['end_frame']
        
        # Load visual and IMU data
        visual_data = torch.load(seq_dir / "visual_data.pt")
        imu_data = torch.load(seq_dir / "imu_data.pt")
        
        # Load poses
        with open(seq_dir / "poses.json", 'r') as f:
            poses_data = json.load(f)
        
        # Extract sequence
        visual_seq = visual_data[start_frame:end_frame]  # [seq_len, C, H, W]
        imu_seq = imu_data[start_frame:end_frame]        # [seq_len, samples_per_frame, 6]
        
        # Extract poses for the sequence
        pose_seq = []
        for i in range(start_frame, end_frame):
            if i < len(poses_data):
                pose = poses_data[i]
                # Convert to 6-DOF pose [tx, ty, tz, rx, ry, rz] (euler angles)
                translation = pose['translation']
                quaternion = pose['rotation']  # [qx, qy, qz, qw]
                
                # Convert quaternion to euler angles (simplified)
                euler = self._quaternion_to_euler(quaternion)
                pose_6dof = translation + euler
                pose_seq.append(pose_6dof)
            else:
                # Pad with last pose if needed
                pose_seq.append(pose_seq[-1] if pose_seq else [0, 0, 0, 0, 0, 0])
        
        poses = torch.tensor(pose_seq, dtype=torch.float32)  # [seq_len, 6]
        
        # Reshape IMU data: [seq_len, samples_per_frame, 6] -> [seq_len, imu_features]
        # For now, take mean across samples_per_frame
        imu_features = imu_seq.mean(dim=1)  # [seq_len, 6]
        
        # Ensure visual data is in correct format
        if visual_seq.dim() == 4:  # [seq_len, C, H, W]
            # Convert to latent features if needed (placeholder for now)
            # In practice, this would go through the pretrained visual encoder
            visual_features = visual_seq.mean(dim=[2, 3])  # [seq_len, C] - simplified
        else:
            visual_features = visual_seq
        
        return {
            'visual_features': visual_features,  # [seq_len, visual_dim]
            'imu_features': imu_features,        # [seq_len, imu_dim]
            'poses': poses,                      # [seq_len, 6]
            'sequence_id': sample_info['sequence_id']
        }
    
    def _quaternion_to_euler(self, quaternion: List[float]) -> List[float]:
        """Convert quaternion [qx, qy, qz, qw] to euler angles [rx, ry, rz]"""
        qx, qy, qz, qw = quaternion
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return [roll, pitch, yaw]