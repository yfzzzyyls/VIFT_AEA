#!/usr/bin/env python3
"""
Evaluate KITTI-trained VIFT model on Aria dataset.
This script properly loads the Aria data and evaluates the model on it.
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader

# Import necessary modules
from src.data.components.aria_kitti_format_dataset import AriaKITTIFormat
from src.models.components.pose_transformer import PoseTransformer
# We'll implement our own error computation functions
from src.utils.custom_transform import Compose, ToTensor, Resize


def evaluate_on_aria(checkpoint_path, aria_data_path, test_sequences):
    """Evaluate KITTI-trained model on Aria sequences."""
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    
    # Extract model state dict - handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'net.' prefix if present
        state_dict = {k.replace('net.', ''): v for k, v in state_dict.items() if k.startswith('net.')}
    else:
        state_dict = checkpoint
    
    # Initialize model
    model = PoseTransformer(
        input_dim=768,
        embedding_dim=768,
        num_layers=4,
        nhead=6,
        dim_feedforward=128,
        dropout=0.0
    ).cuda()
    
    # Load model weights
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create transforms
    transform = Compose([
        ToTensor(),
        Resize(args=(256, 512))  # Height, Width for KITTI format
    ])
    
    # Evaluate on each test sequence
    all_results = {}
    
    for seq in test_sequences:
        print(f"\nEvaluating on sequence {seq}...")
        
        # Create dataset for single sequence
        dataset = AriaKITTIFormat(
            root=aria_data_path,
            sequence_length=11,
            train_seqs=[seq],
            transform=transform
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        # Evaluate
        trans_errors = []
        rot_errors = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=f"Sequence {seq}")):
                imgs, imus, rots, weights = inputs
                
                # Move to GPU
                imgs = imgs.cuda()
                imus = imus.cuda()
                targets = targets.cuda()
                
                # Forward pass
                predictions = model(imgs, imus)
                
                # Compute errors
                for i in range(predictions.shape[0]):
                    for j in range(predictions.shape[1]):
                        pred = predictions[i, j].cpu().numpy()
                        gt = targets[i, j].cpu().numpy()
                        
                        # Translation error (percentage)
                        trans_err = np.linalg.norm(pred[:3] - gt[:3]) / np.linalg.norm(gt[:3]) * 100
                        trans_errors.append(trans_err)
                        
                        # Rotation error (degrees)
                        from scipy.spatial.transform import Rotation as R
                        pred_rot = R.from_euler('xyz', pred[3:])
                        gt_rot = R.from_euler('xyz', gt[3:])
                        # Compute relative rotation
                        rel_rot = gt_rot.inv() * pred_rot
                        # Get angle in degrees
                        rot_err = np.abs(rel_rot.magnitude() * 180 / np.pi)
                        rot_errors.append(rot_err)
        
        # Compute statistics
        trans_errors = np.array(trans_errors)
        rot_errors = np.array(rot_errors)
        
        results = {
            'trans_mean': np.mean(trans_errors),
            'trans_median': np.median(trans_errors),
            'trans_std': np.std(trans_errors),
            'rot_mean': np.mean(rot_errors),
            'rot_median': np.median(rot_errors),
            'rot_std': np.std(rot_errors),
            'num_samples': len(trans_errors)
        }
        
        all_results[seq] = results
        
        print(f"Sequence {seq} Results:")
        print(f"  Translation Error: {results['trans_mean']:.2f}% (mean), {results['trans_median']:.2f}% (median)")
        print(f"  Rotation Error: {results['rot_mean']:.2f}° (mean), {results['rot_median']:.2f}° (median)")
    
    # Overall statistics
    print("\n=== Overall Results ===")
    all_trans = []
    all_rot = []
    for seq, res in all_results.items():
        print(f"\nSequence {seq}:")
        print(f"  Translation: {res['trans_mean']:.2f}% ± {res['trans_std']:.2f}%")
        print(f"  Rotation: {res['rot_mean']:.2f}° ± {res['rot_std']:.2f}°")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                        default='/home/external/VIFT_AEA/logs/train/runs/2025-06-14_12-53-26/checkpoints/epoch_197.ckpt',
                        help='Path to checkpoint')
    parser.add_argument('--data-path', type=str,
                        default='/home/external/VIFT_AEA/aria_processed',
                        help='Path to Aria processed data')
    parser.add_argument('--test-sequences', nargs='+', 
                        default=['016', '017', '018', '019'],
                        help='Test sequences to evaluate')
    
    args = parser.parse_args()
    
    evaluate_on_aria(args.checkpoint, args.data_path, args.test_sequences)