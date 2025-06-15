#!/usr/bin/env python3
"""
Evaluate VIFT model trained from scratch on test data.
Generates 3D plots and detailed metrics for sequences 016, 017, 018, 019.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from train_aria_from_scratch import VIFTFromScratch
from src.data.components.aria_raw_dataset import AriaRawDataset


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = VIFTFromScratch().to(device)
    
    # Handle DataParallel state dict
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    
    print(f"\n📊 Model Information:")
    print(f"   Loaded checkpoint from epoch {checkpoint['epoch']}")
    if 'val_metrics' in checkpoint:
        metrics = checkpoint['val_metrics']
        print(f"   Validation loss: {metrics['loss']:.4f}")
        if 'trans_error' in metrics:
            print(f"   Val translation error: {metrics['trans_error']*100:.2f} cm")
        if 'rot_error' in metrics:
            print(f"   Val rotation error: {metrics['rot_error']:.2f}°")
    
    return model


def integrate_trajectory(relative_poses, initial_pose=None):
    """Integrate relative poses to get absolute trajectory
    
    Args:
        relative_poses: Array of relative poses [N, 7] where each pose is [x,y,z,qx,qy,qz,qw]
        initial_pose: Optional initial pose [7]. If None, starts at origin.
    """
    N = relative_poses.shape[0]
    
    absolute_positions = np.zeros((N + 1, 3))
    absolute_rotations = np.zeros((N + 1, 4))
    
    # Set initial pose
    if initial_pose is not None:
        absolute_positions[0] = initial_pose[:3]
        absolute_rotations[0] = initial_pose[3:]
        current_position = initial_pose[:3].copy()
        current_rotation = R.from_quat(initial_pose[3:])
    else:
        absolute_positions[0] = [0, 0, 0]
        absolute_rotations[0] = [0, 0, 0, 1]
        current_position = np.zeros(3)
        current_rotation = R.from_quat([0, 0, 0, 1])
    
    for i in range(N):
        rel_trans = relative_poses[i, :3]
        
        # Quaternion format
        rel_quat = relative_poses[i, 3:]
        rel_quat = rel_quat / (np.linalg.norm(rel_quat) + 1e-8)
        
        # Check for valid quaternion
        if np.any(np.isnan(rel_quat)) or np.linalg.norm(rel_quat) < 0.5:
            rel_quat = np.array([0, 0, 0, 1])
        
        rel_rotation = R.from_quat(rel_quat)
        
        # Apply transformation
        world_trans = current_rotation.apply(rel_trans)
        current_position += world_trans
        current_rotation = current_rotation * rel_rotation
        
        absolute_positions[i + 1] = current_position
        absolute_rotations[i + 1] = current_rotation.as_quat()
    
    return absolute_positions, absolute_rotations


def evaluate_model(model, test_loader, device, output_dir, test_sequences):
    """Evaluate model on test data"""
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    sequence_results = {seq_id: [] for seq_id in test_sequences}
    
    print("\n🔍 Evaluating on test sequences...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing batches")):
            # Move to device
            images = batch['images'].to(device)
            imu = batch['imu'].to(device)
            gt_poses = batch['gt_poses'].to(device)
            
            # Get predictions
            predictions = model({'images': images, 'imu': imu})  # {'poses': [B, 10, 7]}
            pred_poses_batch = predictions['poses']  # [B, 10, 7]
            
            # Process each sample in batch
            batch_size = pred_poses_batch.shape[0]
            for i in range(batch_size):
                # For multi-step prediction, we predict all 10 transitions
                pred_poses = pred_poses_batch[i].cpu().numpy()  # [10, 7]
                gt_poses_sample = gt_poses[i].cpu().numpy()  # [10, 7]
                
                # Compute errors for all transitions
                trans_errors = []
                rot_errors = []
                
                for j in range(10):
                    # Translation error
                    trans_error = np.linalg.norm(pred_poses[j, :3] - gt_poses_sample[j, :3])
                    trans_errors.append(trans_error)
                    
                    # Rotation error
                    pred_q = pred_poses[j, 3:]
                    pred_q = pred_q / (np.linalg.norm(pred_q) + 1e-8)
                    gt_q = gt_poses_sample[j, 3:]
                    gt_q = gt_q / (np.linalg.norm(gt_q) + 1e-8)
                    
                    # Compute angle between quaternions
                    dot = np.abs(np.dot(pred_q, gt_q))
                    dot = np.clip(dot, -1.0, 1.0)
                    angle = 2 * np.arccos(dot)
                    rot_error = np.rad2deg(angle)
                    rot_errors.append(rot_error)
                
                result = {
                    'pred_poses': pred_poses,  # [10, 7]
                    'gt_poses': gt_poses_sample,  # [10, 7]
                    'trans_errors': np.array(trans_errors),  # [10]
                    'rot_errors': np.array(rot_errors),  # [10]
                    'seq_name': batch['seq_name'][i],
                    'start_idx': batch['start_idx'][i].item()
                }
                all_results.append(result)
                
                # Group by sequence
                seq_id = batch['seq_name'][i]
                if seq_id in sequence_results:
                    sequence_results[seq_id].append(result)
    
    # Compute overall statistics
    all_trans_errors = np.concatenate([r['trans_errors'] for r in all_results])
    all_rot_errors = np.concatenate([r['rot_errors'] for r in all_results])
    
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    print(f"\n📏 Translation Error:")
    print(f"   Mean:   {np.mean(all_trans_errors) * 100:.2f} cm")
    print(f"   Std:    {np.std(all_trans_errors) * 100:.2f} cm")
    print(f"   Median: {np.median(all_trans_errors) * 100:.2f} cm")
    print(f"   95%:    {np.percentile(all_trans_errors, 95) * 100:.2f} cm")
    print(f"   Max:    {np.max(all_trans_errors) * 100:.2f} cm")
    
    print(f"\n🔄 Rotation Error (degrees):")
    print(f"   Mean:   {np.mean(all_rot_errors):.2f}")
    print(f"   Std:    {np.std(all_rot_errors):.2f}")
    print(f"   Median: {np.median(all_rot_errors):.2f}")
    print(f"   95%:    {np.percentile(all_rot_errors, 95):.2f}")
    print(f"   Max:    {np.max(all_rot_errors):.2f}")
    print("="*50)
    
    # Save sample trajectories
    plot_sample_trajectories(all_results[:8], output_dir)
    
    # Generate 3D plots and CSV files for each test sequence
    print("\n📊 Generating 3D trajectory plots and CSV files...")
    fps = 20  # Aria dataset is 20 FPS
    
    for seq_id in test_sequences:
        if seq_id not in sequence_results or not sequence_results[seq_id]:
            continue
        
        # Sort by start index to maintain chronological order
        seq_results = sorted(sequence_results[seq_id], key=lambda x: x['start_idx'])
        
        print(f"\nSequence {seq_id}:")
        print(f"  Total windows: {len(seq_results)}")
        
        # No need to sample since we already use stride=10 in the dataset
        seq_results_sampled = seq_results
        print(f"  Non-overlapping windows: {len(seq_results_sampled)}")
        
        # Concatenate sampled results
        all_pred = np.concatenate([r['pred_poses'] for r in seq_results_sampled], axis=0)
        all_gt = np.concatenate([r['gt_poses'] for r in seq_results_sampled], axis=0)
        
        # Integrate full trajectory
        pred_positions_full, pred_rotations_full = integrate_trajectory(all_pred)
        gt_positions_full, gt_rotations_full = integrate_trajectory(all_gt)
        
        # Save trajectories to CSV
        save_trajectory_csv(pred_positions_full, gt_positions_full, 
                           pred_rotations_full, gt_rotations_full,
                           seq_id, output_dir)
        
        # Plot full trajectory
        plot_trajectory_3d(
            pred_positions_full,
            gt_positions_full,
            seq_id,
            os.path.join(output_dir, f'trajectory_3d_{seq_id}.png')
        )
        
        # Plot rotation trajectory
        plot_rotation_3d(
            pred_rotations_full[1:],
            gt_rotations_full[1:],
            seq_id,
            os.path.join(output_dir, f'rotation_3d_{seq_id}.png')
        )
        
        # Generate 1-second plots
        frames_1s = min(fps, len(all_pred))
        if frames_1s > 0:
            pred_positions_1s, pred_rotations_1s = integrate_trajectory(all_pred[:frames_1s])
            gt_positions_1s, gt_rotations_1s = integrate_trajectory(all_gt[:frames_1s])
            
            plot_trajectory_3d(
                pred_positions_1s,
                gt_positions_1s,
                seq_id,
                os.path.join(output_dir, f'trajectory_3d_{seq_id}_1s.png'),
                time_window='1s'
            )
            
            plot_rotation_3d(
                pred_rotations_1s[1:],
                gt_rotations_1s[1:],
                seq_id,
                os.path.join(output_dir, f'rotation_3d_{seq_id}_1s.png'),
                time_window='1s'
            )
        
        # Generate 5-second plots
        frames_5s = min(5 * fps, len(all_pred))
        if frames_5s > 0:
            pred_positions_5s, pred_rotations_5s = integrate_trajectory(all_pred[:frames_5s])
            gt_positions_5s, gt_rotations_5s = integrate_trajectory(all_gt[:frames_5s])
            
            plot_trajectory_3d(
                pred_positions_5s,
                gt_positions_5s,
                seq_id,
                os.path.join(output_dir, f'trajectory_3d_{seq_id}_5s.png'),
                time_window='5s'
            )
            
            plot_rotation_3d(
                pred_rotations_5s[1:],
                gt_rotations_5s[1:],
                seq_id,
                os.path.join(output_dir, f'rotation_3d_{seq_id}_5s.png'),
                time_window='5s'
            )
    
    # Save numerical results
    save_evaluation_metrics(all_trans_errors, all_rot_errors, output_dir)
    
    print(f"\n📊 Generated outputs:")
    for seq_id in test_sequences:
        if seq_id in sequence_results and sequence_results[seq_id]:
            print(f"   Sequence {seq_id}:")
            print(f"   - trajectory_{seq_id}_gt.csv (ground truth)")
            print(f"   - trajectory_{seq_id}_pred.csv (predictions with errors)")
            print(f"   - trajectory_3d_{seq_id}.png, trajectory_3d_{seq_id}_1s.png, trajectory_3d_{seq_id}_5s.png")
            print(f"   - rotation_3d_{seq_id}.png, rotation_3d_{seq_id}_1s.png, rotation_3d_{seq_id}_5s.png")
    
    return all_results


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
    gt_csv_path = os.path.join(output_dir, f'trajectory_{seq_id}_gt.csv')
    gt_df.to_csv(gt_csv_path, index=False)
    print(f"Saved ground truth to {gt_csv_path}")
    
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
    pred_csv_path = os.path.join(output_dir, f'trajectory_{seq_id}_pred.csv')
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"Saved predictions to {pred_csv_path}")


def plot_trajectory_3d(pred_positions, gt_positions, sequence_name, output_path, time_window=None):
    """Create 3D trajectory plot"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert to centimeters for display
    pred_positions_cm = pred_positions * 100
    gt_positions_cm = gt_positions * 100
    
    # Plot trajectories
    ax.plot(gt_positions_cm[:, 0], gt_positions_cm[:, 1], gt_positions_cm[:, 2], 
            'b-', linewidth=2, label='Ground Truth')
    ax.plot(pred_positions_cm[:, 0], pred_positions_cm[:, 1], pred_positions_cm[:, 2], 
            'r--', linewidth=2, label='Prediction', alpha=0.8)
    
    # Mark start and end
    ax.scatter(*gt_positions_cm[0], color='green', s=100, marker='o', label='Start')
    ax.scatter(*gt_positions_cm[-1], color='red', s=100, marker='x', label='End')
    
    # Calculate path lengths
    gt_length = np.sum(np.linalg.norm(np.diff(gt_positions, axis=0), axis=1)) * 100
    pred_length = np.sum(np.linalg.norm(np.diff(pred_positions, axis=0), axis=1)) * 100
    
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    
    if time_window:
        ax.set_title(f'Sequence {sequence_name} - First {time_window}\nGT Length: {gt_length:.1f}cm, Pred Length: {pred_length:.1f}cm')
    else:
        ax.set_title(f'Sequence {sequence_name} - Full Trajectory\nGT Length: {gt_length:.1f}cm, Pred Length: {pred_length:.1f}cm')
    
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_rotation_3d(pred_rotations, gt_rotations, sequence_name, output_path, time_window=None):
    """Create 3D rotation trajectory visualization"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert quaternions to axis-angle
    gt_axis_angles = np.array([R.from_quat(q).as_rotvec() for q in gt_rotations])
    pred_axis_angles = np.array([R.from_quat(q).as_rotvec() for q in pred_rotations])
    
    # Downsample for clarity if sequence is very long
    stride = max(1, len(gt_axis_angles) // 1000)
    
    # Plot rotation trajectories
    ax.plot(gt_axis_angles[::stride, 0], gt_axis_angles[::stride, 1], gt_axis_angles[::stride, 2], 
            'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(pred_axis_angles[::stride, 0], pred_axis_angles[::stride, 1], pred_axis_angles[::stride, 2], 
            'r--', linewidth=2, label='Prediction', alpha=0.8)
    
    # Mark start and end
    ax.scatter(*gt_axis_angles[0], color='green', s=100, marker='o', label='Start')
    ax.scatter(*gt_axis_angles[-1], color='red', s=100, marker='x', label='End')
    
    # Calculate rotation statistics
    rot_errors = []
    for pred_q, gt_q in zip(pred_rotations, gt_rotations):
        gt_rot = R.from_quat(gt_q)
        pred_rot = R.from_quat(pred_q)
        rel_rot = gt_rot.inv() * pred_rot
        angle_error = np.abs(rel_rot.magnitude() * 180 / np.pi)
        rot_errors.append(angle_error)
    
    mean_error = np.mean(rot_errors)
    max_error = np.max(rot_errors)
    
    ax.set_xlabel('X (rad)')
    ax.set_ylabel('Y (rad)')
    ax.set_zlabel('Z (rad)')
    
    if time_window:
        ax.set_title(f'Sequence {sequence_name} - Rotation First {time_window}\n'
                     f'Mean Error: {mean_error:.2f}°, Max Error: {max_error:.2f}°')
    else:
        ax.set_title(f'Sequence {sequence_name} - Full Rotation Trajectory\n'
                     f'Mean Error: {mean_error:.2f}°, Max Error: {max_error:.2f}°')
    
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_sample_trajectories(results, output_dir):
    """Plot sample trajectories"""
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot first 8 sequences
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, result in enumerate(results[:8]):
        ax = axes[idx]
        
        # Integrate trajectories
        pred_positions, _ = integrate_trajectory(result['pred_poses'])
        gt_positions, _ = integrate_trajectory(result['gt_poses'])
        
        # Convert to cm
        gt_positions_cm = gt_positions * 100
        pred_positions_cm = pred_positions * 100
        
        # Plot 2D trajectories
        ax.plot(gt_positions_cm[:, 0], gt_positions_cm[:, 1], 'b-', label='Ground Truth', linewidth=2)
        ax.plot(pred_positions_cm[:, 0], pred_positions_cm[:, 1], 'r--', label='Prediction', linewidth=2)
        
        ax.scatter(gt_positions_cm[0, 0], gt_positions_cm[0, 1], c='green', s=100, marker='o')
        ax.scatter(gt_positions_cm[-1, 0], gt_positions_cm[-1, 1], c='black', s=100, marker='x')
        
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_title(f'Sample {idx+1} - Seq {result["seq_name"]}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'test_trajectories_2d.png'), dpi=150)
    plt.close()
    
    # Plot error distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    all_trans_errors = np.concatenate([r['trans_errors'] for r in results]) * 100
    all_rot_errors = np.concatenate([r['rot_errors'] for r in results])
    
    # Translation error histogram
    ax1.hist(all_trans_errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Translation Error (cm)')
    ax1.set_ylabel('Count')
    ax1.set_title('Translation Error Distribution')
    ax1.axvline(np.mean(all_trans_errors), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_trans_errors):.2f}cm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rotation error histogram
    ax2.hist(all_rot_errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Rotation Error (degrees)')
    ax2.set_ylabel('Count')
    ax2.set_title('Rotation Error Distribution')
    ax2.axvline(np.mean(all_rot_errors), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_rot_errors):.2f}°')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'error_distributions.png'), dpi=150)
    plt.close()
    
    print(f"\n📊 Plots saved to: {plot_dir}/")


def save_evaluation_metrics(trans_errors, rot_errors, output_dir):
    """Save evaluation metrics to file"""
    results_file = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(results_file, 'w') as f:
        f.write("VIFT FROM SCRATCH EVALUATION RESULTS\n")
        f.write("=====================================\n\n")
        f.write("Translation Error (cm):\n")
        f.write(f"  Mean:   {np.mean(trans_errors) * 100:.2f}\n")
        f.write(f"  Std:    {np.std(trans_errors) * 100:.2f}\n")
        f.write(f"  Median: {np.median(trans_errors) * 100:.2f}\n")
        f.write(f"  95%:    {np.percentile(trans_errors, 95) * 100:.2f}\n")
        f.write(f"  Max:    {np.max(trans_errors) * 100:.2f}\n\n")
        f.write("Rotation Error (degrees):\n")
        f.write(f"  Mean:   {np.mean(rot_errors):.2f}\n")
        f.write(f"  Std:    {np.std(rot_errors):.2f}\n")
        f.write(f"  Median: {np.median(rot_errors):.2f}\n")
        f.write(f"  95%:    {np.percentile(rot_errors, 95):.2f}\n")
        f.write(f"  Max:    {np.max(rot_errors):.2f}\n\n")
        f.write("Note: High rotation error is expected due to Euler angle representation.\n")
        f.write("Consider using quaternion representation for better results.\n")
    
    print(f"\n💾 Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate VIFT model trained from scratch')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='aria_processed',
                        help='Directory with processed Aria data')
    parser.add_argument('--output-dir', type=str, default='evaluation_from_scratch',
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Evaluating VIFT Model Trained from Scratch")
    print("="*60)
    
    # Load model
    model = load_checkpoint(args.checkpoint, device)
    
    # Test sequences
    test_sequences = ['016', '017', '018', '019']
    print(f"\n📁 Test sequences: {test_sequences}")
    
    # Create test dataset
    test_dir = Path(args.data_dir) / 'test'
    if not test_dir.exists():
        print(f"\n❌ Test directory not found: {test_dir}")
        print("Please ensure test sequences are in the test directory")
        return
    
    test_dataset = AriaRawDataset(test_dir, sequence_length=11, stride=10)  # Use stride=10 to match training
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"📁 Test dataset loaded: {len(test_dataset)} samples")
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, str(output_dir), test_sequences)
    
    print("\n✅ Evaluation complete!")
    print(f"📂 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()