#!/usr/bin/env python3
"""
Inference script for unified VIFT model with direct prediction.
Works with models trained using the updated train_vift_aria.py.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the unified model architecture
from train_vift_aria import VIFTDirect


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model with same config
    config = checkpoint.get('config', {
        'input_dim': 768,
        'embedding_dim': 128,
        'num_layers': 2,
        'nhead': 8,
        'dim_feedforward': 512
    })
    
    model = VIFTDirect(
        input_dim=config.get('input_dim', 768),
        embedding_dim=config.get('embedding_dim', 128),
        num_layers=config.get('num_layers', 2),
        nhead=config.get('nhead', 8),
        dim_feedforward=config.get('dim_feedforward', 512)
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Architecture: {config.get('architecture', 'unknown')}")
    print(f"Rotation loss type: {config.get('rotation_loss', 'unknown')}")
    
    return model


def integrate_trajectory(relative_poses):
    """
    Integrate relative poses to get absolute trajectory
    
    Args:
        relative_poses: Array of shape [N, 7] with relative poses (tx, ty, tz, qx, qy, qz, qw)
    
    Returns:
        absolute_positions: Array of shape [N+1, 3] with absolute positions
        absolute_rotations: Array of shape [N+1, 4] with absolute quaternions
    """
    N = relative_poses.shape[0]
    
    # Initialize arrays
    absolute_positions = np.zeros((N + 1, 3))
    absolute_rotations = np.zeros((N + 1, 4))
    absolute_rotations[0] = [0, 0, 0, 1]  # Identity quaternion
    
    # Current pose
    current_position = np.zeros(3)
    current_rotation = R.from_quat([0, 0, 0, 1])  # Identity
    
    for i in range(N):
        # Extract relative pose
        rel_trans = relative_poses[i, :3]
        rel_quat = relative_poses[i, 3:]
        
        # Normalize quaternion
        rel_quat = rel_quat / (np.linalg.norm(rel_quat) + 1e-8)
        rel_rotation = R.from_quat(rel_quat)
        
        # Transform translation to world frame and update position
        world_trans = current_rotation.apply(rel_trans)
        current_position += world_trans
        
        # Update rotation
        current_rotation = current_rotation * rel_rotation
        
        # Store absolute pose
        absolute_positions[i + 1] = current_position
        absolute_rotations[i + 1] = current_rotation.as_quat()
    
    return absolute_positions, absolute_rotations


def run_inference(model, processed_dir, test_sequences, device, output_dir):
    """Run inference on test sequences"""
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for seq_id in test_sequences:
        print(f"\nProcessing sequence {seq_id}...")
        
        # Load cached features
        seq_path = os.path.join(processed_dir, f'seq_{seq_id}_features.npz')
        if not os.path.exists(seq_path):
            print(f"Sequence {seq_id} not found, skipping...")
            continue
        
        data = np.load(seq_path)
        visual_features = torch.from_numpy(data['visual_features']).float().unsqueeze(0).to(device)
        imu_features = torch.from_numpy(data['imu_features']).float().unsqueeze(0).to(device)
        gt_poses = data['relative_poses']  # Ground truth relative poses
        
        # Prepare batch
        batch = {
            'visual_features': visual_features,
            'imu_features': imu_features
        }
        
        # Run inference
        with torch.no_grad():
            predictions = model(batch)
            pred_poses = predictions['poses'].squeeze(0).cpu().numpy()  # [seq_len-1, 7]
        
        # Integrate trajectories
        pred_positions, pred_rotations = integrate_trajectory(pred_poses)
        gt_positions, gt_rotations = integrate_trajectory(gt_poses)
        
        # Store results
        results = {
            'pred_poses': pred_poses,
            'gt_poses': gt_poses,
            'pred_positions': pred_positions,
            'gt_positions': gt_positions,
            'pred_rotations': pred_rotations,
            'gt_rotations': gt_rotations
        }
        all_results[seq_id] = results
        
        # Save results
        np.savez(
            os.path.join(output_dir, f'seq_{seq_id}_results.npz'),
            **results
        )
        
        # Print statistics
        print(f"Sequence {seq_id} statistics:")
        print(f"  Prediction shape: {pred_poses.shape}")
        print(f"  Trajectory length: {gt_positions[-1, 0]:.1f}, {gt_positions[-1, 1]:.1f}, {gt_positions[-1, 2]:.1f} cm")
        print(f"  Predicted length: {pred_positions[-1, 0]:.1f}, {pred_positions[-1, 1]:.1f}, {pred_positions[-1, 2]:.1f} cm")
        
        # Sample predictions
        print("  Sample predictions (first 5 frames):")
        for i in range(min(5, pred_poses.shape[0])):
            print(f"    Frame {i}: Pred=[{pred_poses[i,0]:.3f}, {pred_poses[i,1]:.3f}, {pred_poses[i,2]:.3f}] cm")
    
    return all_results


def plot_results(all_results, output_dir):
    """Generate comprehensive plots"""
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Combined 2D trajectory plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, (seq_id, results) in enumerate(all_results.items()):
        if idx >= 4:
            break
        
        ax = axes[idx]
        pred_pos = results['pred_positions']
        gt_pos = results['gt_positions']
        
        # Plot full trajectory
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], 'b-', label='Ground Truth', linewidth=2)
        ax.plot(pred_pos[:, 0], pred_pos[:, 1], 'r--', label='Prediction', linewidth=2)
        
        # Mark start and end
        ax.scatter(gt_pos[0, 0], gt_pos[0, 1], c='green', s=100, marker='o', label='Start')
        ax.scatter(gt_pos[-1, 0], gt_pos[-1, 1], c='black', s=100, marker='x', label='GT End')
        ax.scatter(pred_pos[-1, 0], pred_pos[-1, 1], c='red', s=100, marker='x', label='Pred End')
        
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_title(f'Sequence {seq_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'trajectories_2d.png'), dpi=150)
    plt.close()
    
    # 2. 3D trajectory plots (1s and 5s)
    for duration in [1, 5]:  # 1 second and 5 seconds
        frames = duration * 20  # 20 FPS
        
        fig = plt.figure(figsize=(16, 12))
        
        for idx, (seq_id, results) in enumerate(all_results.items()):
            if idx >= 4:
                break
            
            # Translation subplot
            ax1 = fig.add_subplot(2, 4, idx+1, projection='3d')
            pred_pos = results['pred_positions'][:frames+1]
            gt_pos = results['gt_positions'][:frames+1]
            
            ax1.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 'b-', label='GT', linewidth=2)
            ax1.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 'r--', label='Pred', linewidth=2)
            ax1.scatter(gt_pos[0, 0], gt_pos[0, 1], gt_pos[0, 2], c='green', s=100, marker='o')
            ax1.set_xlabel('X (cm)')
            ax1.set_ylabel('Y (cm)')
            ax1.set_zlabel('Z (cm)')
            ax1.set_title(f'Seq {seq_id} - Translation ({duration}s)')
            ax1.legend()
            
            # Rotation subplot (axis-angle representation)
            ax2 = fig.add_subplot(2, 4, idx+5, projection='3d')
            
            # Convert quaternions to axis-angle for visualization
            pred_rot = results['pred_rotations'][:frames+1]
            gt_rot = results['gt_rotations'][:frames+1]
            
            pred_aa = []
            gt_aa = []
            
            for i in range(len(pred_rot)):
                # Predicted rotation axis-angle
                r_pred = R.from_quat(pred_rot[i])
                aa_pred = r_pred.as_rotvec()
                pred_aa.append(aa_pred)
                
                # Ground truth rotation axis-angle  
                r_gt = R.from_quat(gt_rot[i])
                aa_gt = r_gt.as_rotvec()
                gt_aa.append(aa_gt)
            
            pred_aa = np.array(pred_aa)
            gt_aa = np.array(gt_aa)
            
            ax2.plot(gt_aa[:, 0], gt_aa[:, 1], gt_aa[:, 2], 'b-', label='GT', linewidth=2)
            ax2.plot(pred_aa[:, 0], pred_aa[:, 1], pred_aa[:, 2], 'r--', label='Pred', linewidth=2)
            ax2.scatter(gt_aa[0, 0], gt_aa[0, 1], gt_aa[0, 2], c='green', s=100, marker='o')
            ax2.set_xlabel('X (rad)')
            ax2.set_ylabel('Y (rad)')
            ax2.set_zlabel('Z (rad)')
            ax2.set_title(f'Seq {seq_id} - Rotation ({duration}s)')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'trajectories_3d_{duration}s.png'), dpi=150)
        plt.close()
    
    # 3. Error analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for idx, (seq_id, results) in enumerate(all_results.items()):
        if idx >= 4:
            break
        
        # Translation error over time
        ax1 = axes[idx//2, idx%2*2]
        pred_pos = results['pred_positions']
        gt_pos = results['gt_positions']
        trans_error = np.linalg.norm(pred_pos - gt_pos, axis=1)
        frames = np.arange(len(trans_error))
        time_sec = frames / 20.0  # 20 FPS
        
        ax1.plot(time_sec, trans_error, 'b-', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Translation Error (cm)')
        ax1.set_title(f'Sequence {seq_id} - Translation Error')
        ax1.grid(True, alpha=0.3)
        
        # Rotation error over time
        ax2 = axes[idx//2, idx%2*2+1]
        pred_rot = results['pred_rotations']
        gt_rot = results['gt_rotations']
        
        rot_errors = []
        for i in range(len(pred_rot)):
            r_pred = R.from_quat(pred_rot[i])
            r_gt = R.from_quat(gt_rot[i])
            r_error = r_gt.inv() * r_pred
            angle_error = np.abs(r_error.as_rotvec())
            rot_errors.append(np.linalg.norm(angle_error))
        
        rot_errors = np.array(rot_errors)
        rot_errors_deg = np.rad2deg(rot_errors)
        
        ax2.plot(time_sec[:len(rot_errors_deg)], rot_errors_deg, 'r-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Rotation Error (degrees)')
        ax2.set_title(f'Sequence {seq_id} - Rotation Error')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'error_analysis.png'), dpi=150)
    plt.close()
    
    print(f"\nPlots saved to {plot_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run inference with unified VIFT model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--processed-dir', type=str, default='aria_processed_full_frames',
                        help='Directory with processed features')
    parser.add_argument('--output-dir', type=str, default='full_frames_results_unified',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--test-sequences', type=str, nargs='+', default=['016', '017', '018', '019'],
                        help='Test sequence IDs')
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_checkpoint(args.checkpoint, device)
    
    # Run inference
    results = run_inference(model, args.processed_dir, args.test_sequences, device, args.output_dir)
    
    # Generate plots
    if results:
        plot_results(results, args.output_dir)
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()