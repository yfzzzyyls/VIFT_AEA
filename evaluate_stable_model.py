#!/usr/bin/env python3
"""
Evaluate stable VIFT model on test data.
Automatically generates test features if they don't exist.
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
import subprocess

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from train_vift_aria_stable import VIFTStable
from src.data.components.aria_latent_dataset import AriaLatentDataset


def check_and_generate_test_features(data_dir, processed_dir):
    """Check if test features exist, generate if not"""
    test_dir = os.path.join(data_dir, 'test')
    
    if not os.path.exists(test_dir) or len(os.listdir(test_dir)) == 0:
        print("\n⚠️  Test features not found. Generating them now...")
        print("This may take a few minutes...\n")
        
        # Run the feature generation script
        cmd = [
            'python', 'generate_all_pretrained_latents_fixed.py',
            '--processed-dir', processed_dir,
            '--output-dir', data_dir,
            '--stride', '10',
            '--pose-scale', '100.0',
            '--process-test'
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Test features generated successfully!")
            print(f"Test features saved to: {test_dir}")
            
            # Print some statistics from the output
            if 'test: ' in result.stdout:
                for line in result.stdout.split('\n'):
                    if 'test: ' in line:
                        print(line.strip())
        except subprocess.CalledProcessError as e:
            print(f"❌ Error generating test features: {e}")
            print(f"Error output: {e.stderr}")
            raise
    else:
        print(f"✅ Test features found in: {test_dir}")
        # Count test samples
        test_samples = len([f for f in os.listdir(test_dir) if f.endswith('_gt.npy')])
        print(f"   Found {test_samples} test samples")


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = VIFTStable().to(device)
    # Load with strict=False to ignore the output_norm layer from old checkpoint
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"\n📊 Model Information:")
    print(f"   Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"   Validation loss: {checkpoint['val_metrics']['loss']:.4f}")
    if 'val_metrics' in checkpoint:
        metrics = checkpoint['val_metrics']
        if 'trans_error' in metrics:
            print(f"   Val translation error: {metrics['trans_error']:.3f} m")
        if 'rot_error_deg' in metrics:
            print(f"   Val rotation error: {metrics['rot_error_deg']:.2f}°")
    
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
        rel_quat = relative_poses[i, 3:]
        
        rel_quat = rel_quat / (np.linalg.norm(rel_quat) + 1e-8)
        
        # Check for valid quaternion
        if np.any(np.isnan(rel_quat)) or np.linalg.norm(rel_quat) < 0.5:
            rel_quat = np.array([0, 0, 0, 1])
        
        rel_rotation = R.from_quat(rel_quat)
        
        world_trans = current_rotation.apply(rel_trans)
        current_position += world_trans
        current_rotation = current_rotation * rel_rotation
        
        absolute_positions[i + 1] = current_position
        absolute_rotations[i + 1] = current_rotation.as_quat()
    
    return absolute_positions, absolute_rotations


def evaluate_model(model, test_loader, device, output_dir, test_sequences, test_dataset):
    """Evaluate model on test data"""
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    sequence_results = {}
    
    # Group results by sequence
    for seq_id in test_sequences:
        sequence_results[seq_id] = []
    
    print("\n🔍 Evaluating on test sequences...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing batches")):
            # Move to device
            batch_gpu = {
                'visual_features': batch['visual_features'].to(device),
                'imu_features': batch['imu_features'].to(device),
                'poses': batch['poses'].to(device)
            }
            
            # Get predictions
            predictions = model(batch_gpu)
            
            # Get predictions - now only single step prediction [B, 1, 7]
            pred_poses = predictions['poses'].cpu().numpy()  # [B, 1, 7]
            
            # For single-step prediction, we compare with the last ground truth pose
            # This represents the transition from current frame to next frame
            gt_poses = batch_gpu['poses'][:, -1:, :].cpu().numpy()  # [B, 1, 7]
            
            # Process each sequence in batch
            for i in range(pred_poses.shape[0]):
                # Get single sequence
                pred_seq = pred_poses[i]
                gt_seq = gt_poses[i]
                
                # Compute errors
                trans_error = np.linalg.norm(pred_seq[:, :3] - gt_seq[:, :3], axis=1)
                
                # Rotation error
                rot_errors = []
                for j in range(len(pred_seq)):
                    pred_q = pred_seq[j, 3:] / (np.linalg.norm(pred_seq[j, 3:]) + 1e-8)
                    gt_q = gt_seq[j, 3:] / (np.linalg.norm(gt_seq[j, 3:]) + 1e-8)
                    
                    # Compute angle between quaternions
                    dot = np.abs(np.dot(pred_q, gt_q))
                    dot = np.clip(dot, -1.0, 1.0)
                    angle = 2 * np.arccos(dot)
                    rot_errors.append(np.rad2deg(angle))
                
                result = {
                    'pred_poses': pred_seq,
                    'gt_poses': gt_seq,
                    'trans_errors': trans_error,
                    'rot_errors': np.array(rot_errors)
                }
                all_results.append(result)
                
                # Use sequence ID and frame index for proper ordering
                seq_id = batch['sequence_id'][i]
                frame_idx = batch['frame_idx'][i].item() if torch.is_tensor(batch['frame_idx'][i]) else batch['frame_idx'][i]
                sequence_results[seq_id].append((frame_idx, result))
    
    # Compute overall statistics
    all_trans_errors = np.concatenate([r['trans_errors'] for r in all_results])
    all_rot_errors = np.concatenate([r['rot_errors'] for r in all_results])
    
    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    print(f"\n📏 Translation Error (cm):")
    print(f"   Mean:   {np.mean(all_trans_errors) * 100:.3f}")
    print(f"   Std:    {np.std(all_trans_errors) * 100:.3f}")
    print(f"   Median: {np.median(all_trans_errors) * 100:.3f}")
    print(f"   95%:    {np.percentile(all_trans_errors, 95) * 100:.3f}")
    print(f"   Max:    {np.max(all_trans_errors) * 100:.3f}")
    
    print(f"\n🔄 Rotation Error (degrees):")
    print(f"   Mean:   {np.mean(all_rot_errors):.3f}")
    print(f"   Std:    {np.std(all_rot_errors):.3f}")
    print(f"   Median: {np.median(all_rot_errors):.3f}")
    print(f"   95%:    {np.percentile(all_rot_errors, 95):.3f}")
    print(f"   Max:    {np.max(all_rot_errors):.3f}")
    print("="*50)
    
    # Save some trajectory visualizations
    plot_sample_trajectories(all_results[:8], output_dir)
    
    # Generate 3D plots for each test sequence
    print("\n📊 Generating 3D trajectory and rotation plots...")
    fps = 20  # Aria dataset is 20 FPS
    
    for seq_id in test_sequences:
        if seq_id not in sequence_results or not sequence_results[seq_id]:
            continue
            
        # Sort results by frame index to ensure chronological order
        seq_results_with_idx = sequence_results[seq_id]
        seq_results_with_idx.sort(key=lambda x: x[0])  # Sort by frame index
        
        # Extract just the results after sorting
        seq_results = [r for _, r in seq_results_with_idx]
        
        # Concatenate all results for this sequence in proper temporal order
        all_pred = np.concatenate([r['pred_poses'] for r in seq_results], axis=0)
        all_gt = np.concatenate([r['gt_poses'] for r in seq_results], axis=0)
        
        # Integrate full trajectory
        pred_positions_full, pred_rotations_full = integrate_trajectory(all_pred)
        gt_positions_full, gt_rotations_full = integrate_trajectory(all_gt)
        
        # Plot full trajectory
        plot_trajectory_3d(
            pred_positions_full,
            gt_positions_full,
            seq_id,
            os.path.join(output_dir, f'trajectory_3d_{seq_id}.png')
        )
        
        # Plot full rotation trajectory
        plot_rotation_3d(
            pred_rotations_full[1:],  # Skip first identity rotation
            gt_rotations_full[1:],
            seq_id,
            os.path.join(output_dir, f'rotation_3d_{seq_id}.png')
        )
        
        # Generate 1-second plots (first 20 frames)
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
        
        # Generate 5-second plots (first 100 frames)
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
    results_file = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(results_file, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("==================\n\n")
        f.write("Translation Error (cm):\n")
        f.write(f"  Mean:   {np.mean(all_trans_errors) * 100:.3f}\n")
        f.write(f"  Std:    {np.std(all_trans_errors) * 100:.3f}\n")
        f.write(f"  Median: {np.median(all_trans_errors) * 100:.3f}\n")
        f.write(f"  95%:    {np.percentile(all_trans_errors, 95) * 100:.3f}\n")
        f.write(f"  Max:    {np.max(all_trans_errors) * 100:.3f}\n\n")
        f.write("Rotation Error (degrees):\n")
        f.write(f"  Mean:   {np.mean(all_rot_errors):.3f}\n")
        f.write(f"  Std:    {np.std(all_rot_errors):.3f}\n")
        f.write(f"  Median: {np.median(all_rot_errors):.3f}\n")
        f.write(f"  95%:    {np.percentile(all_rot_errors, 95):.3f}\n")
        f.write(f"  Max:    {np.max(all_rot_errors):.3f}\n")
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"\n📊 Generated 3D plots:")
    for seq_id in test_sequences:
        print(f"   - trajectory_3d_{seq_id}.png, trajectory_3d_{seq_id}_1s.png, trajectory_3d_{seq_id}_5s.png")
        print(f"   - rotation_3d_{seq_id}.png, rotation_3d_{seq_id}_1s.png, rotation_3d_{seq_id}_5s.png")
    
    return all_results


def plot_trajectory_3d(pred_positions, gt_positions, sequence_name, output_path, time_window=None):
    """Create 3D trajectory plot with optional time window"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert from meters to centimeters for display
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
    
    # Calculate path lengths (convert to cm)
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
    plt.savefig(output_path)
    plt.close()
    print(f"Saved 3D trajectory plot to {output_path}")


def plot_rotation_3d(pred_rotations, gt_rotations, sequence_name, output_path, time_window=None):
    """Create 3D rotation trajectory visualization in axis-angle space with optional time window"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert quaternions to axis-angle representation
    # This maps rotations to 3D space where direction = rotation axis, magnitude = rotation angle
    gt_axis_angles = np.array([R.from_quat(q).as_rotvec() for q in gt_rotations])
    pred_axis_angles = np.array([R.from_quat(q).as_rotvec() for q in pred_rotations])
    
    # Downsample for clarity if sequence is very long
    stride = max(1, len(gt_axis_angles) // 1000)
    
    # Plot rotation trajectories
    ax.plot(gt_axis_angles[::stride, 0], gt_axis_angles[::stride, 1], gt_axis_angles[::stride, 2], 
            'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(pred_axis_angles[::stride, 0], pred_axis_angles[::stride, 1], pred_axis_angles[::stride, 2], 
            'r--', linewidth=2, label='Prediction', alpha=0.8)
    
    # Mark start and end points
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
    
    # Set labels and title
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
    
    # Set equal aspect ratio for better visualization
    max_range = np.array([
        gt_axis_angles[:, 0].max() - gt_axis_angles[:, 0].min(),
        gt_axis_angles[:, 1].max() - gt_axis_angles[:, 1].min(),
        gt_axis_angles[:, 2].max() - gt_axis_angles[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (gt_axis_angles[:, 0].max() + gt_axis_angles[:, 0].min()) * 0.5
    mid_y = (gt_axis_angles[:, 1].max() + gt_axis_angles[:, 1].min()) * 0.5
    mid_z = (gt_axis_angles[:, 2].max() + gt_axis_angles[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved 3D rotation plot to {output_path}")


def plot_sample_trajectories(results, output_dir):
    """Plot sample trajectories"""
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot first 8 sequences in a 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, result in enumerate(results[:8]):
        ax = axes[idx]
        
        # Integrate trajectories
        pred_positions, _ = integrate_trajectory(result['pred_poses'])
        gt_positions, _ = integrate_trajectory(result['gt_poses'])
        
        # Convert to cm for display
        gt_positions_cm = gt_positions * 100
        pred_positions_cm = pred_positions * 100
        
        # Plot 2D trajectories
        ax.plot(gt_positions_cm[:, 0], gt_positions_cm[:, 1], 'b-', label='Ground Truth', linewidth=2)
        ax.plot(pred_positions_cm[:, 0], pred_positions_cm[:, 1], 'r--', label='Prediction', linewidth=2)
        
        ax.scatter(gt_positions_cm[0, 0], gt_positions_cm[0, 1], c='green', s=100, marker='o', label='Start')
        ax.scatter(gt_positions_cm[-1, 0], gt_positions_cm[-1, 1], c='black', s=100, marker='x', label='GT End')
        ax.scatter(pred_positions_cm[-1, 0], pred_positions_cm[-1, 1], c='red', s=100, marker='x', label='Pred End')
        
        # Compute trajectory length (convert to cm)
        gt_length = np.sum(np.linalg.norm(np.diff(gt_positions, axis=0), axis=1)) * 100
        pred_length = np.sum(np.linalg.norm(np.diff(pred_positions, axis=0), axis=1)) * 100
        
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_title(f'Test Sample {idx+1}\nGT: {gt_length:.1f}cm, Pred: {pred_length:.1f}cm')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'test_trajectories_2d.png'), dpi=150)
    plt.close()
    
    # Plot error distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    all_trans_errors = np.concatenate([r['trans_errors'] for r in results])
    all_rot_errors = np.concatenate([r['rot_errors'] for r in results])
    
    # Translation error histogram (convert to cm for display)
    all_trans_errors_cm = all_trans_errors * 100
    ax1.hist(all_trans_errors_cm, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Translation Error (cm)')
    ax1.set_ylabel('Count')
    ax1.set_title('Translation Error Distribution')
    ax1.axvline(np.mean(all_trans_errors_cm), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_trans_errors_cm):.2f}cm')
    ax1.axvline(np.median(all_trans_errors_cm), color='green', linestyle='--', 
                label=f'Median: {np.median(all_trans_errors_cm):.2f}cm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rotation error histogram
    ax2.hist(all_rot_errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Rotation Error (degrees)')
    ax2.set_ylabel('Count')
    ax2.set_title('Rotation Error Distribution')
    ax2.axvline(np.mean(all_rot_errors), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_rot_errors):.2f}°')
    ax2.axvline(np.median(all_rot_errors), color='green', linestyle='--', 
                label=f'Median: {np.median(all_rot_errors):.2f}°')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'error_distributions.png'), dpi=150)
    plt.close()
    
    print(f"\n📊 Plots saved to: {plot_dir}/")
    print(f"   - test_trajectories_2d.png")
    print(f"   - error_distributions.png")


def main():
    parser = argparse.ArgumentParser(description='Evaluate stable VIFT model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='aria_latent_full_frames',
                        help='Directory with latent features')
    parser.add_argument('--processed-dir', type=str, default='aria_processed_full_frames',
                        help='Directory with processed data (for generating features if needed)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    # Check and generate test features if needed
    check_and_generate_test_features(args.data_dir, args.processed_dir)
    
    # Load model
    model = load_checkpoint(args.checkpoint, device)
    
    # Load test dataset
    test_dataset = AriaLatentDataset(os.path.join(args.data_dir, 'test'))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=2)
    
    print(f"\n📁 Test dataset loaded: {len(test_dataset)} samples")
    
    # Load test sequences
    test_sequences = ['016', '017', '018', '019']  # From test_sequences.txt
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, args.output_dir, test_sequences, test_dataset)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()