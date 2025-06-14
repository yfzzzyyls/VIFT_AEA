"""
Aria Latent Dataset in KITTI format - loads pre-extracted features
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class AriaLatentKITTIFormat(Dataset):
    """
    Dataset for loading pre-extracted Aria latent features in KITTI format.
    This dataset is compatible with KITTI-trained VIFT models.
    """
    
    def __init__(self, root, sequences=['016', '017', '018', '019']):
        """
        Args:
            root: Path to the directory containing latent features
            sequences: List of sequence names to load
        """
        self.root = Path(root)
        self.sequences = sequences
        
        # Load all samples
        self.samples = []
        self.weights = []
        
        for seq in sequences:
            seq_dir = self.root / seq
            if not seq_dir.exists():
                print(f"Warning: Sequence directory {seq_dir} not found")
                continue
                
            # Find all feature files in this sequence
            feature_files = sorted(seq_dir.glob("*[0-9].npy"))
            
            for feat_file in feature_files:
                # Extract sample index from filename
                idx = int(feat_file.stem)
                
                # Construct paths for associated files
                gt_file = seq_dir / f"{idx}_gt.npy"
                rot_file = seq_dir / f"{idx}_rot.npy"
                w_file = seq_dir / f"{idx}_w.npy"
                
                # Check all files exist
                if not all(f.exists() for f in [gt_file, rot_file, w_file]):
                    print(f"Warning: Missing files for sample {idx} in sequence {seq}")
                    continue
                
                sample = {
                    'feature_path': feat_file,
                    'gt_path': gt_file,
                    'rot_path': rot_file,
                    'weight_path': w_file,
                    'seq': seq,
                    'idx': idx
                }
                
                self.samples.append(sample)
                
                # Load weight
                weight = np.load(w_file).item()
                self.weights.append(weight)
        
        print(f"Loaded {len(self.samples)} samples from {len(sequences)} sequences")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Returns:
            features: Concatenated visual and IMU features [seq_len-1, 768]
            gt: Ground truth relative poses [seq_len-1, 6]
            rot: Rotation magnitude for the segment
            weight: Sample weight
        """
        sample = self.samples[index]
        
        # Load data
        features = np.load(sample['feature_path'])  # [seq_len-1, 768]
        gt = np.load(sample['gt_path'])  # [seq_len-1, 6] or similar shape
        rot = np.load(sample['rot_path'])  # scalar
        weight = self.weights[index]
        
        # Convert to tensors
        features = torch.from_numpy(features).float()
        gt = torch.from_numpy(gt).float()
        rot = torch.tensor(rot).float()
        weight = torch.tensor(weight).float()
        
        # Ensure gt is the right shape
        if len(gt.shape) == 3:
            # If gt has batch dimension, squeeze it
            gt = gt.squeeze(0)
        
        return (features, gt, rot, weight), gt
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += f'    Root: {self.root}\n'
        fmt_str += f'    Sequences: {self.sequences}\n'
        fmt_str += f'    Number of samples: {len(self.samples)}\n'
        return fmt_str