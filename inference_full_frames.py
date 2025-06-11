#!/usr/bin/env python3
"""
Run inference with full frame model and generate 3D trajectory plots
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import argparse

# Import model and dataset
from src.models.multihead_vio_separate_fixed import MultiHeadVIOModelSeparate
from generate_all_pretrained_latents_fixed import process_sequence, load_pretrained_model


def load_checkpoint(checkpoint_path):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model configuration
    config = checkpoint.get('config', {})
    
    # Initialize model
    model = MultiHeadVIOModelSeparate(
        visual_dim=512,
        imu_dim=256,
        hidden_dim=config.get('hidden_dim', 128),
        num_heads=config.get('num_heads', 4),
        num_shared_layers=4,
        num_specialized_layers=3,
        dropout=0.0,  # No dropout for inference
        sequence_length=10
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def run_inference_on_sequence(model, vift_model, sequence_path, window_size=11, stride=10, device='cuda'):
    """Run inference on a full sequence"""
    
    # Generate latent features for the sequence
    print(f"Generating latent features for {sequence_path}")
    features_list, poses_list = process_sequence(
        sequence_path, 
        vift_model,
        device,
        window_size=window_size,
        stride=stride,
        pose_scale=100.0
    )
    
    if not features_list:
        print(f"Failed to generate latent features for {sequence_path}")
        return None
    
    # Prepare for inference
    all_predictions = []
    all_ground_truth = []
    
    num_samples = len(features_list)
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Running inference"):
            # Extract features for this window
            v_feat, i_feat = features_list[i]
            gt_poses = poses_list[i]
            
            # Prepare batch
            batch = {
                'visual_features': v_feat.unsqueeze(0).float().to(device),
                'imu_features': i_feat.unsqueeze(0).float().to(device),
                'poses': gt_poses.unsqueeze(0).float().to(device)
            }
            
            # Run model
            output = model(batch)
            
            # Get predictions (already in cm)
            pred_trans = output['translation'].squeeze(0).cpu().numpy()
            pred_rot = output['rotation'].squeeze(0).cpu().numpy()
            
            # Get ground truth (already in cm from dataset)
            gt_trans = gt_poses[:, :3].numpy()
            gt_rot = gt_poses[:, 3:].numpy()
            
            # Store predictions and ground truth
            pred_poses = np.concatenate([pred_trans, pred_rot], axis=-1)
            all_predictions.append(pred_poses)
            all_ground_truth.append(gt_poses)
    
    return {
        'predictions': all_predictions,
        'ground_truth': all_ground_truth,
        'metadata': {
            'sequence_name': Path(sequence_path).name,
            'num_windows': len(all_predictions),
            'window_size': window_size,
            'stride': stride
        }
    }


def reconstruct_trajectory(relative_poses, start_pose=None):
    """Reconstruct absolute trajectory from relative poses"""
    if start_pose is None:
        # Start at origin with identity rotation
        start_pose = np.array([0, 0, 0, 0, 0, 0, 1])  # [x, y, z, qx, qy, qz, qw]
    
    trajectory = [start_pose[:3].copy()]
    current_pos = start_pose[:3].copy()
    current_rot = start_pose[3:].copy()
    
    for rel_pose in relative_poses:
        # Update position (relative translation is already in world frame for this dataset)
        current_pos = current_pos + rel_pose[:3]
        trajectory.append(current_pos.copy())
        
        # Update rotation (simplified - not using quaternion multiplication for visualization)
        current_rot = rel_pose[3:]
    
    return np.array(trajectory)


def plot_3d_trajectory(pred_trajectory, gt_trajectory, output_path, title="Trajectory Comparison"):
    """Create 3D trajectory plot"""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], 
            'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], 
            'r--', linewidth=2, label='Prediction', alpha=0.8)
    
    # Mark start and end
    ax.scatter(*gt_trajectory[0], color='green', s=100, marker='o', label='Start')
    ax.scatter(*gt_trajectory[-1], color='red', s=100, marker='x', label='End')
    
    # Calculate trajectory length
    gt_length = np.sum(np.linalg.norm(np.diff(gt_trajectory, axis=0), axis=1))
    pred_length = np.sum(np.linalg.norm(np.diff(pred_trajectory, axis=0), axis=0))
    
    # Set labels and title
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title(f'{title}\nGT Length: {gt_length:.1f}cm, Pred Length: {pred_length:.1f}cm')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([
        gt_trajectory[:, 0].max() - gt_trajectory[:, 0].min(),
        gt_trajectory[:, 1].max() - gt_trajectory[:, 1].min(),
        gt_trajectory[:, 2].max() - gt_trajectory[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (gt_trajectory[:, 0].max() + gt_trajectory[:, 0].min()) * 0.5
    mid_y = (gt_trajectory[:, 1].max() + gt_trajectory[:, 1].min()) * 0.5
    mid_z = (gt_trajectory[:, 2].max() + gt_trajectory[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                       default='full_frames_checkpoints/20250611_161413/best_model_epoch_3.pt')
    parser.add_argument('--test-sequences', type=str,
                       default='/home/external/VIFT_AEA/aria_latent_full_frames/test_sequences.txt')
    parser.add_argument('--processed-dir', type=str,
                       default='/home/external/VIFT_AEA/aria_processed_full_frames')
    parser.add_argument('--output-dir', type=str,
                       default='full_frames_results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--window-size', type=int, default=11)
    parser.add_argument('--stride', type=int, default=10)
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_checkpoint(args.checkpoint)
    model = model.to(device)
    
    # Load pretrained VIFT encoder
    print("Loading pretrained VIFT encoder...")
    vift_model = load_pretrained_model('pretrained_models/vf_512_if_256_3e-05.model')
    vift_model = vift_model.to(device)
    
    # Load test sequences
    with open(args.test_sequences, 'r') as f:
        test_sequences = [line.strip() for line in f if line.strip()]
    
    print(f"Test sequences: {test_sequences}")
    
    # Run inference on each test sequence
    results = {}
    
    for seq_id in test_sequences:
        print(f"\nProcessing sequence {seq_id}")
        sequence_path = Path(args.processed_dir) / seq_id
        
        if not sequence_path.exists():
            print(f"Sequence path {sequence_path} not found")
            continue
        
        # Run inference
        result = run_inference_on_sequence(
            model, vift_model, sequence_path, 
            window_size=args.window_size,
            stride=args.stride,
            device=device
        )
        
        if result is None:
            continue
        
        results[seq_id] = result
        
        # Reconstruct full trajectory from predictions
        print("Reconstructing trajectories...")
        
        # Combine all window predictions (with overlap handling)
        all_pred_poses = []
        all_gt_poses = []
        
        for i, (pred, gt) in enumerate(zip(result['predictions'], result['ground_truth'])):
            if i == 0:
                # First window - use all frames
                all_pred_poses.extend(pred)
                all_gt_poses.extend(gt)
            else:
                # Subsequent windows - skip overlapping frames
                # Since stride=10 and window=10, no overlap
                all_pred_poses.extend(pred)
                all_gt_poses.extend(gt)
        
        # Convert to numpy arrays
        all_pred_poses = np.array(all_pred_poses)
        all_gt_poses = np.array(all_gt_poses)
        
        # Reconstruct absolute trajectories
        pred_trajectory = reconstruct_trajectory(all_pred_poses)
        gt_trajectory = reconstruct_trajectory(all_gt_poses)
        
        # Generate plots
        plot_path = output_dir / f'trajectory_3d_{seq_id}.png'
        plot_3d_trajectory(
            pred_trajectory, 
            gt_trajectory, 
            plot_path,
            title=f"Sequence {seq_id} - Full Frame Model"
        )
        
        # Calculate metrics
        frame_errors = np.linalg.norm(all_pred_poses[:, :3] - all_gt_poses[:, :3], axis=1)
        mean_error = np.mean(frame_errors)
        std_error = np.std(frame_errors)
        
        print(f"Sequence {seq_id} - Mean error: {mean_error:.2f}cm (±{std_error:.2f}cm)")
        
        # Save detailed results
        np.savez(
            output_dir / f'results_{seq_id}.npz',
            pred_poses=all_pred_poses,
            gt_poses=all_gt_poses,
            pred_trajectory=pred_trajectory,
            gt_trajectory=gt_trajectory,
            frame_errors=frame_errors,
            metadata=result['metadata']
        )
    
    # Generate summary
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)
    print(f"Processed {len(results)} sequences")
    print(f"Results saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review the 3D trajectory plots")
    print("2. Analyze alignment between predictions and ground truth")
    print("3. Compare with previous results to see improvements")


if __name__ == "__main__":
    main()