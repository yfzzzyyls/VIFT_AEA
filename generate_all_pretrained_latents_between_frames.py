#!/usr/bin/env python3
"""
Generate latent features using Visual-Selective-VIO pretrained model
with FIXED between-frames IMU data format.
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
        
    def forward(self, imgs, imu):
        # The model outputs 10 transition features for 11 input frames
        v_feat, i_feat = self.Feature_net(imgs, imu)
        return v_feat, i_feat  # Both are [B, 10, feature_dim]


def load_pretrained_model(model_path):
    """Load pretrained Visual-Selective-VIO model."""
    model = WrapperModel()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    encoder_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('Feature_net.'):
            new_k = k.replace('Feature_net.', '')
            encoder_dict[new_k] = v
    
    model.Feature_net.load_state_dict(encoder_dict, strict=False)
    print(f"✅ Loaded {len(encoder_dict)} encoder parameters")
    return model


def process_sequence_between_frames(seq_dir, model, device, window_size=11, stride=1):
    """Process a single sequence with between-frames IMU data."""
    
    # Load data
    visual_data = torch.load(os.path.join(seq_dir, 'visual_data.pt'))  # [N, 3, H, W]
    imu_data = torch.load(os.path.join(seq_dir, 'imu_data.pt'))        # [N-1, 50, 6]
    
    # Ensure float32 data type
    visual_data = visual_data.float()
    imu_data = imu_data.float()
    
    # Load poses
    poses_file = os.path.join(seq_dir, 'poses_quaternion.json')
    with open(poses_file, 'r') as f:
        poses_data = json.load(f)
    
    # Verify data consistency
    num_frames = visual_data.shape[0]
    num_imu_intervals = imu_data.shape[0]
    
    if num_imu_intervals != num_frames - 1:
        print(f"⚠️ Warning: IMU intervals ({num_imu_intervals}) != frames-1 ({num_frames-1})")
        # Adjust to consistent length
        min_intervals = min(num_imu_intervals, num_frames - 1)
        visual_data = visual_data[:min_intervals + 1]
        imu_data = imu_data[:min_intervals]
        num_frames = min_intervals + 1
    
    features_list = []
    poses_list = []
    
    # Process with sliding window
    for start_idx in range(0, num_frames - window_size + 1, stride):
        end_idx = start_idx + window_size
        
        # Extract window of 11 frames
        window_visual = visual_data[start_idx:end_idx]  # [11, 3, H, W]
        
        # Extract corresponding 10 IMU intervals
        window_imu = imu_data[start_idx:start_idx + window_size - 1]  # [10, 50, 6]
        
        # Resize images to 256x512
        window_visual_resized = F.interpolate(
            window_visual, 
            size=(256, 512), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Normalize images to [-0.5, 0.5]
        window_visual_normalized = window_visual_resized - 0.5
        
        # Prepare IMU data - extract 11 samples per interval
        # VIFT expects overlapping windows of 11 IMU samples for each transition
        # We have 50 samples per interval, so we can extract 11 samples with proper spacing
        window_imu_110 = []
        
        # Process each of the 10 intervals
        for i in range(window_size - 1):  # 10 intervals
            interval_imu = window_imu[i]  # [50, 6]
            
            # Extract 11 evenly spaced samples from the 50 available
            # This maintains temporal coverage of the full interval
            indices = np.linspace(0, 49, 11, dtype=int)
            sampled = interval_imu[indices]  # [11, 6]
            window_imu_110.append(sampled)
        
        window_imu_110 = torch.cat(window_imu_110, dim=0)  # [110, 6]
        
        # Add batch dimension
        batch_visual = window_visual_normalized.unsqueeze(0).to(device)  # [1, 11, 3, 256, 512]
        batch_imu = window_imu_110.unsqueeze(0).to(device)              # [1, 110, 6]
        
        # Generate features
        with torch.no_grad():
            v_feat, i_feat = model(batch_visual, batch_imu)  # Both are [1, 10, feature_dim]
            v_feat = v_feat.squeeze(0).cpu()  # [10, 512]
            i_feat = i_feat.squeeze(0).cpu()  # [10, 256]
        
        # Store features
        features_list.append((v_feat, i_feat))
        
        # Extract the 10 relative poses for this window
        # These are already in the correct format (between consecutive frames)
        window_poses = []
        for i in range(start_idx, start_idx + window_size - 1):
            pose = poses_data[i]
            next_pose = poses_data[i + 1]
            
            # Compute relative pose (already done in the fixed script)
            # Just extract the pose data
            t = pose['translation']
            q = pose['quaternion']
            pose_vec = np.array([t[0], t[1], t[2], q[0], q[1], q[2], q[3]], dtype=np.float32)
            window_poses.append(pose_vec)
        
        window_poses = np.array(window_poses)
        poses_list.append(torch.from_numpy(window_poses))  # [10, 7]
    
    return features_list, poses_list


def generate_split_data_between_frames(processed_dir, output_dir, model, device, stride=1, split_ratios=(0.6, 0.2, 0.2), skip_test=True):
    """Generate train/val/test splits from sequences with between-frames IMU."""
    
    # Get all sequence directories
    seq_dirs = sorted([d for d in Path(processed_dir).iterdir() 
                      if d.is_dir() and d.name.isdigit()])
    
    num_sequences = len(seq_dirs)
    print(f"Found {num_sequences} sequences")
    
    # Calculate split sizes
    train_size = int(num_sequences * split_ratios[0])
    val_size = int(num_sequences * split_ratios[1])
    
    # Split sequences
    train_seqs = seq_dirs[:train_size]
    val_seqs = seq_dirs[train_size:train_size + val_size]
    test_seqs = seq_dirs[train_size + val_size:]
    
    print(f"Split: {len(train_seqs)} train, {len(val_seqs)} val, {len(test_seqs)} test sequences")
    
    # Process each split
    splits = {
        'train': train_seqs,
        'val': val_seqs
    }
    
    if not skip_test:
        splits['test'] = test_seqs
    
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
                # Check if this is a between-frames format
                metadata_file = seq_dir / 'metadata.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    if metadata.get('imu_format') != 'between_frames':
                        print(f"⚠️ Skipping {seq_dir.name}: Not between-frames format")
                        continue
                
                # Process sequence
                features_list, poses_list = process_sequence_between_frames(
                    seq_dir, model, device, 
                    window_size=11, stride=stride
                )
                
                # Save each window as a sample
                for (v_feat, i_feat), poses in zip(features_list, poses_list):
                    sample_id = sample_counter[split_name]
                    
                    # Save visual and IMU features separately
                    np.save(os.path.join(split_dir, f"{sample_id}_visual.npy"), v_feat.numpy())
                    np.save(os.path.join(split_dir, f"{sample_id}_imu.npy"), i_feat.numpy())
                    
                    # Save ground truth poses
                    np.save(os.path.join(split_dir, f"{sample_id}_gt.npy"), poses.numpy())
                    
                    sample_counter[split_name] += 1
                    
            except Exception as e:
                print(f"Error processing {seq_dir}: {e}")
                continue
        
        print(f"Generated {sample_counter[split_name]} samples for {split_name}")
    
    # Save metadata
    metadata = {
        'feature_dim': 768,
        'visual_dim': 512,
        'inertial_dim': 256,
        'sequence_length': 10,
        'window_size': 11,
        'stride': stride,
        'imu_format': 'between_frames',
        'imu_downsampling': '50->10 samples per interval',
        'sample_counts': sample_counter,
        'note': 'Features extracted from between-frames IMU data with proper temporal alignment'
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        import pickle
        pickle.dump(metadata, f)
    
    print(f"\n✅ Successfully generated latent features with between-frames IMU!")
    return sample_counter


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate latent features with between-frames IMU')
    parser.add_argument('--processed-dir', type=str, default='aria_processed_fixed',
                        help='Directory with processed Aria sequences (between-frames format)')
    parser.add_argument('--output-dir', type=str, default='aria_latent_between_frames',
                        help='Output directory for latent features')
    parser.add_argument('--model-path', type=str, 
                        default='pretrained_models/vf_512_if_256_3e-05.model',
                        help='Path to pretrained model')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for sliding window')
    parser.add_argument('--skip-test', action='store_true', default=True,
                        help='Skip test set processing')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pretrained model
    print(f"\nLoading pretrained model from {args.model_path}...")
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found at {args.model_path}")
        return
    
    model = load_pretrained_model(args.model_path)
    model = model.to(device)
    model.eval()
    
    # Generate features
    print(f"\nGenerating latent features with between-frames IMU...")
    print(f"Input directory: {args.processed_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Stride: {args.stride}")
    
    sample_counts = generate_split_data_between_frames(
        args.processed_dir, 
        args.output_dir, 
        model, 
        device,
        stride=args.stride,
        skip_test=not args.skip_test
    )
    
    print(f"\nTotal samples:")
    for split, count in sample_counts.items():
        if count > 0:
            print(f"  {split}: {count}")
    
    print(f"\nYou can now train your model using:")
    print(f"python train_efficient.py \\")
    print(f"    --data-dir {args.output_dir} \\")
    print(f"    --epochs 50 --batch-size 32 --lr 5e-5")


if __name__ == '__main__':
    main()