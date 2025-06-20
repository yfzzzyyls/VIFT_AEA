#!/usr/bin/env python3
"""
Evaluate IMU-only model on Aria Everyday Activities dataset.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import argparse
import json


# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from train_aria_from_scratch_imu_only import IMUOnlyVIO
from src.data.components.aria_raw_dataset import AriaRawDataset
from umeyama_alignment import umeyama_alignment


def plot_trajectory_3d(positions_pred, positions_gt, sequence_name, output_path, time_window=None):
    """Create 3D trajectory plot with better visualization."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert to centimeters for display
    pred_positions_cm = positions_pred * 100
    gt_positions_cm = positions_gt * 100
    
    # Plot trajectories
    ax.plot(gt_positions_cm[:, 0], gt_positions_cm[:, 1], gt_positions_cm[:, 2], 
            'b-', linewidth=3, label='Ground Truth', alpha=0.8)
    ax.plot(pred_positions_cm[:, 0], pred_positions_cm[:, 1], pred_positions_cm[:, 2], 
            'r--', linewidth=2, label='IMU-Only Prediction', alpha=0.8)
    
    # Mark start and end
    ax.scatter(*gt_positions_cm[0], color='green', s=100, marker='o', label='Start')
    ax.scatter(*gt_positions_cm[-1], color='red', s=100, marker='x', label='End')
    
    # Calculate path lengths and error
    gt_length = np.sum(np.linalg.norm(np.diff(positions_gt, axis=0), axis=1)) * 100
    pred_length = np.sum(np.linalg.norm(np.diff(positions_pred, axis=0), axis=1)) * 100
    
    # Calculate APE
    min_len = min(len(positions_pred), len(positions_gt))
    ape_mean = np.mean(np.linalg.norm(positions_pred[:min_len] - positions_gt[:min_len], axis=1)) * 100
    
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)') 
    ax.set_zlabel('Z (cm)')
    
    # Set equal aspect ratio for all axes
    try:
        ax.set_box_aspect([1, 1, 1])  # Available in matplotlib >= 3.4
    except AttributeError:
        # Fallback for older matplotlib versions
        pass
    
    # Create title with metrics
    if time_window:
        title = f'Sequence {sequence_name} - First {time_window}\n'
    else:
        title = f'Sequence {sequence_name}\n'
    title += f'GT Length: {gt_length:.1f}cm, Pred Length: {pred_length:.1f}cm, APE: {ape_mean:.2f}cm'
    
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set view angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_trajectory_csv(pred_positions, gt_positions, pred_rotations, gt_rotations, seq_id, output_dir):
    """Save trajectory data to separate CSV files for ground truth and predictions"""
    
    # Save ground truth data
    gt_data = []
    for i in range(len(gt_positions)):
        row = {
            'frame': i,
            'x': gt_positions[i, 0],
            'y': gt_positions[i, 1],
            'z': gt_positions[i, 2]
        }
        
        if i < len(gt_rotations):
            # Store quaternions directly
            row.update({
                'qx': gt_rotations[i, 0],
                'qy': gt_rotations[i, 1],
                'qz': gt_rotations[i, 2],
                'qw': gt_rotations[i, 3]
            })
        
        gt_data.append(row)
    
    gt_df = pd.DataFrame(gt_data)
    gt_csv_path = output_dir / f'trajectory_{seq_id}_gt.csv'
    gt_df.to_csv(gt_csv_path, index=False)
    print(f"  Saved ground truth to {gt_csv_path}")
    
    # Save prediction data
    pred_data = []
    for i in range(len(pred_positions)):
        row = {
            'frame': i,
            'x': pred_positions[i, 0],
            'y': pred_positions[i, 1],
            'z': pred_positions[i, 2]
        }
        
        if i < len(pred_rotations):
            # Store quaternions directly
            row.update({
                'qx': pred_rotations[i, 0],
                'qy': pred_rotations[i, 1],
                'qz': pred_rotations[i, 2],
                'qw': pred_rotations[i, 3]
            })
        
        # Add errors compared to ground truth
        if i < len(gt_positions):
            row['trans_error'] = np.linalg.norm(pred_positions[i] - gt_positions[i])
            
            if i < len(gt_rotations) and i < len(pred_rotations):
                # Compute rotation error
                gt_rot = R.from_quat(gt_rotations[i])
                pred_rot = R.from_quat(pred_rotations[i])
                rel_rot = gt_rot.inv() * pred_rot
                rot_error = np.abs(rel_rot.magnitude() * 180 / np.pi)
                row['rot_error_deg'] = rot_error
        
        pred_data.append(row)
    
    pred_df = pd.DataFrame(pred_data)
    pred_csv_path = output_dir / f'trajectory_{seq_id}_pred.csv'
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"  Saved predictions to {pred_csv_path}")


def integrate_trajectory(relative_poses, initial_pose=None):
    """Integrate relative poses to get absolute trajectory.
    
    Args:
        relative_poses: [N, 7] array of relative poses (translation + quaternion)
        initial_pose: [7] array of initial pose, or None to start at origin
    
    Returns:
        positions: [N+1, 3] array of absolute positions
        orientations: [N+1, 4] array of absolute orientations (quaternions)
    """
    num_poses = len(relative_poses)
    absolute_positions = np.zeros((num_poses + 1, 3))
    absolute_orientations = np.zeros((num_poses + 1, 4))
    
    # Initial pose
    if initial_pose is not None:
        current_position = initial_pose[:3].copy()
        current_rotation = R.from_quat(initial_pose[3:])
    else:
        current_position = np.zeros(3)
        current_rotation = R.from_quat([0, 0, 0, 1])  # Identity quaternion
    
    absolute_positions[0] = current_position
    absolute_orientations[0] = current_rotation.as_quat()
    
    # Integrate relative poses
    for i, rel_pose in enumerate(relative_poses):
        rel_trans = rel_pose[:3]
        rel_quat = rel_pose[3:]
        rel_rotation = R.from_quat(rel_quat)
        
        # Apply transformation
        world_trans = current_rotation.apply(rel_trans)
        current_position += world_trans
        current_rotation = current_rotation * rel_rotation
        
        absolute_positions[i + 1] = current_position
        absolute_orientations[i + 1] = current_rotation.as_quat()
    
    return absolute_positions, absolute_orientations


def compute_ate(pred_positions, gt_positions):
    """Compute Absolute Trajectory Error (ATE)."""
    assert pred_positions.shape == gt_positions.shape
    errors = np.linalg.norm(pred_positions - gt_positions, axis=1)
    return {
        'mean': np.mean(errors),
        'std': np.std(errors),
        'median': np.median(errors),
        'min': np.min(errors),
        'max': np.max(errors),
        'rmse': np.sqrt(np.mean(errors**2))
    }


def evaluate_model(model, dataloader, device, output_dir):
    """Evaluate the model on the test set."""
    model.eval()
    
    all_metrics = []
    sequence_results = {}
    
    # Create output directories
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move to device (only IMU and ground truth needed)
            batch_data = {
                'imu': batch['imu'].to(device).float(),
                'gt_poses': batch['gt_poses'].to(device)
            }
            
            # Get metadata
            seq_names = batch['seq_name']
            start_indices = batch['start_idx'].cpu().numpy()
            
            # Forward pass
            predictions = model(batch_data)
            pred_poses = predictions['poses'].cpu().numpy()  # [B, 20, 7]
            gt_poses = batch['gt_poses'].cpu().numpy()      # [B, 20, 7]
            
            # Process each sequence in the batch
            for i in range(pred_poses.shape[0]):
                seq_name = seq_names[i]
                start_idx = start_indices[i]
                
                # Get poses for this sequence
                pred_poses_seq = pred_poses[i]  # [20, 7]
                gt_poses_seq = gt_poses[i]      # [20, 7]
                
                # Initialize sequence results if needed
                if seq_name not in sequence_results:
                    sequence_results[seq_name] = {
                        'pred_poses': [],
                        'gt_poses': [],
                        'start_indices': []
                    }
                
                # Store results
                sequence_results[seq_name]['pred_poses'].append(pred_poses_seq)
                sequence_results[seq_name]['gt_poses'].append(gt_poses_seq)
                sequence_results[seq_name]['start_indices'].append(start_idx)
                
                # Compute per-step errors
                trans_errors = np.linalg.norm(pred_poses_seq[:, :3] - gt_poses_seq[:, :3], axis=1)
                
                # Rotation errors
                rot_errors = []
                for j in range(len(pred_poses_seq)):
                    pred_quat = pred_poses_seq[j, 3:]
                    gt_quat = gt_poses_seq[j, 3:]
                    
                    # Normalize quaternions
                    pred_quat = pred_quat / np.linalg.norm(pred_quat)
                    gt_quat = gt_quat / np.linalg.norm(gt_quat)
                    
                    # Compute angle between quaternions
                    dot_product = np.abs(np.dot(pred_quat, gt_quat))
                    dot_product = np.clip(dot_product, -1.0, 1.0)
                    angle_error = 2 * np.arccos(dot_product)
                    rot_errors.append(np.rad2deg(angle_error))
                
                rot_errors = np.array(rot_errors)
                
                # Store metrics
                all_metrics.append({
                    'sequence': seq_name,
                    'start_idx': start_idx,
                    'trans_error_mean': np.mean(trans_errors),
                    'trans_error_std': np.std(trans_errors),
                    'rot_error_mean': np.mean(rot_errors),
                    'rot_error_std': np.std(rot_errors)
                })
    
    # Process and visualize complete sequences
    print("\nProcessing complete sequences...")
    sequence_ate_metrics = {}
    
    for seq_name, results in sequence_results.items():
        # Sort by start index
        sorted_indices = np.argsort(results['start_indices'])
        
        print(f"\n{seq_name}:")
        print(f"  Total windows: {len(sorted_indices)}")
        
        # Use all overlapping windows and average predictions
        # This matches the training procedure
        
        # Load absolute ground truth poses from JSON file
        seq_dir = dataloader.dataset.data_dir / seq_name
        with open(seq_dir / 'poses_quaternion.json', 'r') as f:
            raw_poses = json.load(f)
        
        # Get all frame indices covered by our predictions
        all_frame_indices = []
        for idx in sorted_indices:
            start_idx = results['start_indices'][idx]
            # Each window covers 21 frames (indices start_idx to start_idx+20)
            frame_indices = list(range(start_idx, start_idx + 21))
            all_frame_indices.extend(frame_indices)
        
        # Remove duplicates and sort
        all_frame_indices = sorted(list(set(all_frame_indices)))
        print(f"  Frame indices range: {all_frame_indices[0]} to {all_frame_indices[-1]}")
        print(f"  Total unique frames: {len(all_frame_indices)}")
        
        # Extract ground truth absolute poses
        gt_positions = np.array([raw_poses[i]['translation'] for i in all_frame_indices])
        gt_quaternions = np.array([raw_poses[i]['quaternion'] for i in all_frame_indices])
        
        # Integrate predictions starting from the first frame's absolute pose
        initial_frame_idx = all_frame_indices[0]
        initial_pose = np.concatenate([
            raw_poses[initial_frame_idx]['translation'],
            raw_poses[initial_frame_idx]['quaternion']
        ])
        
        # Average overlapping predictions - this matches training procedure
        # Create a dictionary to accumulate predictions for each transition
        transition_predictions = {}
        transition_counts = {}
        
        for idx in sorted_indices:
            start_idx = results['start_indices'][idx]
            pred_poses = results['pred_poses'][idx]  # [20, 7]
            
            # Each window predicts 20 transitions
            for i in range(20):
                # Transition from frame start_idx+i to start_idx+i+1
                trans_key = (start_idx + i, start_idx + i + 1)
                
                if trans_key not in transition_predictions:
                    transition_predictions[trans_key] = pred_poses[i].copy()
                    transition_counts[trans_key] = 1
                else:
                    transition_predictions[trans_key] += pred_poses[i]
                    transition_counts[trans_key] += 1
        
        # Average the accumulated predictions
        transition_keys = sorted(transition_predictions.keys())
        all_pred = []
        for key in transition_keys:
            avg_pred = transition_predictions[key] / transition_counts[key]
            all_pred.append(avg_pred)
        all_pred = np.array(all_pred)
        
        print(f"  Unique transitions covered: {len(all_pred)}")
        print(f"  Average predictions per transition: {np.mean(list(transition_counts.values())):.1f}")
        
        # Debug the initial pose
        print(f"  Initial frame index: {initial_frame_idx}")
        print(f"  Initial pose from JSON: {initial_pose}")
        print(f"  Number of averaged predictions to integrate: {len(all_pred)}")
        
        # Integrate predictions from initial pose
        pred_positions, pred_orientations = integrate_trajectory(all_pred, initial_pose)
        gt_orientations = gt_quaternions
        
        # Debug info
        print(f"  Pred positions shape: {pred_positions.shape}")
        print(f"  GT positions shape: {gt_positions.shape}")
        print(f"  Number of averaged transitions: {len(all_pred)}")
        print(f"  Expected GT frames: {len(all_pred) + 1}")
        print(f"  First prediction step: {all_pred[0] if len(all_pred) > 0 else 'None'}")
        print(f"  Pred range: X[{pred_positions[:,0].min():.3f}, {pred_positions[:,0].max():.3f}], "
              f"Y[{pred_positions[:,1].min():.3f}, {pred_positions[:,1].max():.3f}], "
              f"Z[{pred_positions[:,2].min():.3f}, {pred_positions[:,2].max():.3f}]")
        print(f"  GT range: X[{gt_positions[:,0].min():.3f}, {gt_positions[:,0].max():.3f}], "
              f"Y[{gt_positions[:,1].min():.3f}, {gt_positions[:,1].max():.3f}], "
              f"Z[{gt_positions[:,2].min():.3f}, {gt_positions[:,2].max():.3f}]")
        
        # Ensure same length - predictions have N+1 positions, GT has N
        # We need to match them properly
        if len(pred_positions) > len(gt_positions):
            # Trim predictions to match GT length
            pred_positions_trimmed = pred_positions[:len(gt_positions)]
            pred_orientations_trimmed = pred_orientations[:len(gt_positions)]
        else:
            pred_positions_trimmed = pred_positions
            pred_orientations_trimmed = pred_orientations
            gt_positions = gt_positions[:len(pred_positions)]
            gt_orientations = gt_orientations[:len(pred_positions)]
        
        print(f"  After trimming - Pred: {pred_positions_trimmed.shape}, GT: {gt_positions.shape}")
        print(f"  Initial pred position: {pred_positions_trimmed[0]}")
        print(f"  Initial GT position: {gt_positions[0]}")
        print(f"  Pred should start at GT initial: {initial_pose[:3]}")
        
        # Skip Umeyama alignment - we want to see raw predictions from the same starting point
        # Apply Umeyama alignment only for metrics, not for visualization
        from umeyama_alignment import align_trajectory
        aligned_pred_positions, _, _, _ = align_trajectory(pred_positions_trimmed, gt_positions, with_scale=False)
        
        # Save CSV files (using non-aligned predictions)
        save_trajectory_csv(
            pred_positions_trimmed,
            gt_positions,
            pred_orientations_trimmed,
            gt_orientations,
            seq_name,
            plots_dir
        )
        
        # Plot full trajectory (using non-aligned predictions)
        plot_trajectory_3d(
            pred_positions_trimmed,
            gt_positions,
            seq_name,
            plots_dir / f'trajectory_3d_{seq_name}.png'
        )
        
        # Generate 1-second plot (20 frames at 20 FPS)
        fps = 20
        frames_1s = min(fps + 1, len(pred_positions_trimmed))  # +1 for initial position
        if frames_1s > 1:
            plot_trajectory_3d(
                pred_positions_trimmed[:frames_1s],
                gt_positions[:frames_1s],
                seq_name,
                plots_dir / f'trajectory_3d_{seq_name}_1s.png',
                time_window='1s'
            )
        
        # Generate 5-second plot (100 frames at 20 FPS)
        frames_5s = min(5 * fps + 1, len(pred_positions_trimmed))  # +1 for initial position
        if frames_5s > 1:
            plot_trajectory_3d(
                pred_positions_trimmed[:frames_5s],
                gt_positions[:frames_5s],
                seq_name,
                plots_dir / f'trajectory_3d_{seq_name}_5s.png',
                time_window='5s'
            )
        
        # Compute ATE metrics (using aligned positions for fair comparison)
        ate_metrics = compute_ate(aligned_pred_positions, gt_positions)
        sequence_ate_metrics[seq_name] = ate_metrics
        
        print(f"\n{seq_name}:")
        print(f"  ATE Mean: {ate_metrics['mean']*100:.2f} cm")
        print(f"  ATE RMSE: {ate_metrics['rmse']*100:.2f} cm")
    
    # Compute overall metrics
    all_trans_errors = [m['trans_error_mean'] for m in all_metrics]
    all_rot_errors = [m['rot_error_mean'] for m in all_metrics]
    all_ate_means = [m['mean'] for m in sequence_ate_metrics.values()]
    
    print("\n" + "="*60)
    print("Overall Evaluation Metrics (IMU-Only Model)")
    print("="*60)
    print(f"Per-step Translation Error: {np.mean(all_trans_errors)*100:.2f} ± {np.std(all_trans_errors)*100:.2f} cm")
    print(f"Per-step Rotation Error: {np.mean(all_rot_errors):.2f} ± {np.std(all_rot_errors):.2f}°")
    print(f"ATE (after alignment): {np.mean(all_ate_means)*100:.2f} ± {np.std(all_ate_means)*100:.2f} cm")
    print("="*60)
    
    # Save metrics
    metrics_file = output_dir / 'evaluation_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("IMU-Only Model Evaluation Results\n")
        f.write("="*60 + "\n\n")
        
        f.write("Per-sequence ATE metrics (after Umeyama alignment):\n")
        for seq_name, ate_metrics in sequence_ate_metrics.items():
            f.write(f"\n{seq_name}:\n")
            f.write(f"  Mean: {ate_metrics['mean']*100:.2f} cm\n")
            f.write(f"  Std: {ate_metrics['std']*100:.2f} cm\n")
            f.write(f"  RMSE: {ate_metrics['rmse']*100:.2f} cm\n")
            f.write(f"  Min: {ate_metrics['min']*100:.2f} cm\n")
            f.write(f"  Max: {ate_metrics['max']*100:.2f} cm\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Overall Metrics:\n")
        f.write(f"Per-step Translation Error: {np.mean(all_trans_errors)*100:.2f} ± {np.std(all_trans_errors)*100:.2f} cm\n")
        f.write(f"Per-step Rotation Error: {np.mean(all_rot_errors):.2f} ± {np.std(all_rot_errors):.2f}°\n")
        f.write(f"ATE (after alignment): {np.mean(all_ate_means)*100:.2f} ± {np.std(all_ate_means)*100:.2f} cm\n")
    
    print(f"\nResults saved to: {output_dir}")
    return sequence_ate_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate IMU-only model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='aria_processed',
                        help='Directory with processed Aria data')
    parser.add_argument('--output-dir', type=str, default='evaluation_imu_only',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    model = IMUOnlyVIO()
    
    # Load state dict with compatibility for missing keys
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle missing s_p parameter (added in later version)
    if 's_p' not in state_dict and hasattr(model, 's_p'):
        print("Note: Checkpoint missing 's_p' parameter, initializing to 0")
        state_dict['s_p'] = torch.zeros(())
    
    # Load with strict=False to handle any other mismatches
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
    
    model = model.to(device)
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    if 'epoch' in checkpoint:
        print(f"Checkpoint from epoch: {checkpoint['epoch']}")
    if 'val_metrics' in checkpoint:
        val_metrics = checkpoint['val_metrics']
        print(f"Validation metrics at checkpoint:")
        print(f"  Loss: {val_metrics['loss']:.6f}")
        print(f"  Trans Error: {val_metrics['trans_error']*100:.2f} cm")
        print(f"  Rot Error: {val_metrics['rot_error']:.2f}°")
    
    # Create test dataset directly
    test_dir = Path(args.data_dir) / 'test'
    if not test_dir.exists():
        print(f"Error: Test directory not found at {test_dir}")
        return
        
    test_dataset = AriaRawDataset(
        test_dir,
        sequence_length=21,  # 21 frames -> 20 transitions
        stride=1  # Use stride=1 to match training - we'll average overlapping predictions
    )
    
    # Create test dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"\nTest samples: {len(test_dataset)}")
    
    # Evaluate
    print("\nEvaluating IMU-only model...")
    evaluate_model(model, test_loader, device, output_dir)


if __name__ == "__main__":
    main()