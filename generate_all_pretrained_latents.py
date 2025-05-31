#!/usr/bin/env python3
"""
Complete pipeline to generate latent features using Visual-Selective-VIO pretrained model
from processed Aria data.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models.components.vsvio import Encoder


class WrapperModel(nn.Module):
    """Wrapper for pretrained encoder."""
    def __init__(self):
        super().__init__()
        class Params:
            v_f_len = 512
            i_f_len = 256
            img_w = 512
            img_h = 256
            imu_dropout = 0.2
            
        self.Feature_net = Encoder(Params())
        
    def forward(self, imgs, imus):
        # The model outputs 10 frames for 11 input frames
        v_feat, i_feat = self.Feature_net(imgs, imus)
        # Pad to 11 frames by repeating the last frame
        v_feat_padded = torch.cat([v_feat, v_feat[:, -1:, :]], dim=1)
        i_feat_padded = torch.cat([i_feat, i_feat[:, -1:, :]], dim=1)
        return torch.cat([v_feat_padded, i_feat_padded], dim=-1)


def load_pretrained_model(model_path):
    """Load pretrained Visual-Selective-VIO model."""
    model = WrapperModel()
    checkpoint = torch.load(model_path, map_location='cpu')
    
    encoder_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('Feature_net.'):
            new_k = k.replace('Feature_net.', '')
            encoder_dict[new_k] = v
    
    model.Feature_net.load_state_dict(encoder_dict, strict=False)
    print(f"✅ Loaded {len(encoder_dict)} encoder parameters")
    return model


def process_sequence(seq_dir, model, device, window_size=11, stride=1):
    """Process a single sequence and extract windowed features."""
    
    # Load data
    visual_data = torch.load(os.path.join(seq_dir, 'visual_data.pt'))  # [N, 3, H, W]
    imu_data = torch.load(os.path.join(seq_dir, 'imu_data.pt'))        # [N, 33, 6]
    
    # Load poses
    with open(os.path.join(seq_dir, 'poses.json'), 'r') as f:
        poses_data = json.load(f)
    
    # Convert poses to tensor
    poses = []
    for pose in poses_data:
        # Get translation and rotation
        t = pose['translation']
        euler = pose['rotation_euler']  # [roll, pitch, yaw]
        
        # Convert Euler angles to quaternion
        # Using scipy's conversion
        from scipy.spatial.transform import Rotation
        r = Rotation.from_euler('xyz', euler)
        q = r.as_quat()  # [x, y, z, w]
        
        # Convert to [x, y, z, qw, qx, qy, qz] format (w first for quaternion)
        pose_vec = [t[0], t[1], t[2], q[3], q[0], q[1], q[2]]
        poses.append(pose_vec)
    poses = torch.tensor(poses, dtype=torch.float32)
    
    num_frames = visual_data.shape[0]
    features_list = []
    poses_list = []
    imus_list = []
    
    # Process with sliding window
    for start_idx in range(0, num_frames - window_size + 1, stride):
        end_idx = start_idx + window_size
        
        # Extract window
        window_visual = visual_data[start_idx:end_idx]  # [11, 3, H, W]
        window_imu = imu_data[start_idx:end_idx]        # [11, 33, 6]
        window_poses = poses[start_idx:end_idx]         # [11, 7]
        
        # Resize images to 256x512 (model expects this size)
        window_visual_resized = F.interpolate(
            window_visual, 
            size=(256, 512), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Normalize images to [-0.5, 0.5] (currently in [0, 1])
        window_visual_normalized = window_visual_resized - 0.5
        
        # Prepare IMU data - we need 110 samples (10 per frame)
        # Simple approach: take first 10 IMU samples from each frame's 33 samples
        window_imu_110 = []
        for i in range(window_size):
            window_imu_110.append(window_imu[i, :10, :])  # Take first 10 samples
        window_imu_110 = torch.cat(window_imu_110, dim=0)  # [110, 6]
        
        # Add batch dimension
        batch_visual = window_visual_normalized.unsqueeze(0).to(device)  # [1, 11, 3, 256, 512]
        batch_imu = window_imu_110.unsqueeze(0).to(device)              # [1, 110, 6]
        
        # Generate features
        with torch.no_grad():
            features = model(batch_visual, batch_imu)  # [1, 11, 768]
            features = features.squeeze(0).cpu()        # [11, 768]
        
        # Average IMU samples for each frame (33 -> 1)
        window_imu_avg = window_imu.mean(dim=1)  # [11, 6]
        
        features_list.append(features)
        poses_list.append(window_poses)
        imus_list.append(window_imu_avg)
    
    return features_list, poses_list, imus_list


def generate_split_data(processed_dir, output_dir, model, device, split_ratios=(0.8, 0.1, 0.1)):
    """Generate train/val/test splits from processed sequences."""
    
    # Get all sequence directories
    seq_dirs = sorted([d for d in Path(processed_dir).iterdir() if d.is_dir()])
    num_sequences = len(seq_dirs)
    
    print(f"Found {num_sequences} sequences")
    
    # Calculate split sizes
    train_size = int(num_sequences * split_ratios[0])
    val_size = int(num_sequences * split_ratios[1])
    test_size = num_sequences - train_size - val_size
    
    # Split sequences
    train_seqs = seq_dirs[:train_size]
    val_seqs = seq_dirs[train_size:train_size + val_size]
    test_seqs = seq_dirs[train_size + val_size:]
    
    print(f"Split: {len(train_seqs)} train, {len(val_seqs)} val, {len(test_seqs)} test sequences")
    
    # Process each split
    splits = {
        'train': train_seqs,
        'val': val_seqs,
        'test': test_seqs
    }
    
    sample_counter = {
        'train': 0,
        'val': 0,
        'test': 0
    }
    
    for split_name, sequences in splits.items():
        print(f"\nProcessing {split_name} split...")
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for seq_dir in tqdm(sequences, desc=f"{split_name} sequences"):
            try:
                # Process sequence
                features_list, poses_list, imus_list = process_sequence(seq_dir, model, device)
                
                # Save each window as a sample
                for features, poses, imus in zip(features_list, poses_list, imus_list):
                    sample_id = sample_counter[split_name]
                    
                    # Save features (768-dim latent features)
                    np.save(os.path.join(split_dir, f"{sample_id}.npy"), features.numpy())
                    
                    # Save ground truth poses
                    np.save(os.path.join(split_dir, f"{sample_id}_gt.npy"), poses.numpy())
                    
                    # Save IMU data
                    np.save(os.path.join(split_dir, f"{sample_id}_w.npy"), imus.numpy())
                    
                    # Save rotation component separately (for compatibility)
                    rotations = poses[:, 3:7].numpy()  # quaternions
                    np.save(os.path.join(split_dir, f"{sample_id}_rot.npy"), rotations)
                    
                    sample_counter[split_name] += 1
                    
            except Exception as e:
                print(f"Error processing {seq_dir}: {e}")
                continue
        
        print(f"Generated {sample_counter[split_name]} samples for {split_name}")
    
    return sample_counter


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate all latent features with pretrained model')
    parser.add_argument('--processed-dir', type=str, default='data/aria_processed',
                        help='Directory with processed Aria sequences')
    parser.add_argument('--output-dir', type=str, default='aria_latent_data_pretrained',
                        help='Output directory for latent features')
    parser.add_argument('--model-path', type=str, 
                        default='pretrained_models/vf_512_if_256_3e-05.model',
                        help='Path to pretrained model')
    parser.add_argument('--window-size', type=int, default=11,
                        help='Window size for sequences')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for sliding window')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pretrained model
    print(f"\nLoading pretrained model from {args.model_path}...")
    model = load_pretrained_model(args.model_path)
    model = model.to(device)
    model.eval()
    
    # Generate features for all splits
    print(f"\nGenerating latent features...")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Output directory: {args.output_dir}")
    
    sample_counts = generate_split_data(
        args.processed_dir, 
        args.output_dir, 
        model, 
        device
    )
    
    # Save metadata
    metadata = {
        'feature_dim': 768,
        'visual_dim': 512,
        'inertial_dim': 256,
        'window_size': args.window_size,
        'stride': args.stride,
        'model_path': args.model_path,
        'normalization': '[-0.5, 0.5]',
        'sample_counts': sample_counts,
        'note': 'Features generated using Visual-Selective-VIO pretrained encoders'
    }
    
    import pickle
    with open(os.path.join(args.output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n✅ Successfully generated all latent features!")
    print(f"Total samples:")
    for split, count in sample_counts.items():
        print(f"  {split}: {count}")
    print(f"\nFeatures saved to: {args.output_dir}")
    print(f"\nYou can now train your model using:")
    print(f"python train_multihead_only.py")


if __name__ == '__main__':
    main()