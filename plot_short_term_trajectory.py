#!/usr/bin/env python3
"""Plot short-term trajectory for AR/VR evaluation (first 5 seconds)"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
from scipy.spatial.transform import Rotation

def plot_short_term_trajectory(npz_file, output_dir, duration_seconds=5, fps=20):
    """
    Plot trajectory for the first N seconds only
    
    Args:
        npz_file: Path to NPZ results file
        output_dir: Output directory for plots
        duration_seconds: Duration to plot in seconds
        fps: Frames per second (20 for Aria)
    """
    # Load data
    print(f"Loading {npz_file}...")
    data = np.load(npz_file, allow_pickle=True)
    
    gt_traj = data['ground_truth']
    pred_traj = data['absolute_trajectory']
    metrics = data['metrics'].item()
    
    # Calculate number of frames to plot
    num_frames = min(duration_seconds * fps, len(gt_traj))
    print(f"Plotting first {num_frames} frames ({duration_seconds} seconds at {fps} fps)")
    
    # Slice trajectories
    gt_short = gt_traj[:num_frames]
    pred_short = pred_traj[:num_frames]
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 3D Trajectory Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(gt_short[:, 0], gt_short[:, 1], gt_short[:, 2], 
            'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(pred_short[:, 0], pred_short[:, 1], pred_short[:, 2], 
            'r-', linewidth=2, label='Prediction', alpha=0.8)
    
    # Mark start and end
    ax.scatter(*gt_short[0, :3], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(*gt_short[-1, :3], c='red', s=100, marker='s', label=f'End ({duration_seconds}s)', zorder=5)
    
    # Calculate short-term error
    trans_errors = np.linalg.norm(pred_short[:, :3] - gt_short[:, :3], axis=1)
    mean_error = np.mean(trans_errors)
    
    ax.set_xlabel('X Position (cm)')
    ax.set_ylabel('Y Position (cm)')
    ax.set_zlabel('Z Position (cm)')
    ax.set_title(f'First {duration_seconds} Seconds Trajectory\nMean Translation Error: {mean_error:.2f} cm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    max_range = np.array([
        gt_short[:, 0].max() - gt_short[:, 0].min(),
        gt_short[:, 1].max() - gt_short[:, 1].min(),
        gt_short[:, 2].max() - gt_short[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (gt_short[:, 0].max() + gt_short[:, 0].min()) * 0.5
    mid_y = (gt_short[:, 1].max() + gt_short[:, 1].min()) * 0.5
    mid_z = (gt_short[:, 2].max() + gt_short[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.savefig(output_dir / f'trajectory_{duration_seconds}s_3d.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 3D Rotation Trajectory Plot (similar to translation but for rotations)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert quaternions to Euler angles for visualization
    gt_euler = np.zeros((num_frames, 3))
    pred_euler = np.zeros((num_frames, 3))
    
    for i in range(num_frames):
        # Ground truth
        gt_quat = gt_short[i, 3:7]  # [qx, qy, qz, qw]
        gt_rot = Rotation.from_quat(gt_quat)
        gt_euler[i] = gt_rot.as_euler('xyz', degrees=True)
        
        # Prediction
        pred_quat = pred_short[i, 3:7]
        pred_rot = Rotation.from_quat(pred_quat)
        pred_euler[i] = pred_rot.as_euler('xyz', degrees=True)
    
    # Plot rotation trajectories in 3D Euler space
    ax.plot(gt_euler[:, 0], gt_euler[:, 1], gt_euler[:, 2], 
            'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.plot(pred_euler[:, 0], pred_euler[:, 1], pred_euler[:, 2], 
            'r-', linewidth=2, label='Prediction', alpha=0.8)
    
    # Mark start and end
    ax.scatter(*gt_euler[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(*gt_euler[-1], c='red', s=100, marker='s', label=f'End ({duration_seconds}s)', zorder=5)
    
    # Calculate rotation errors
    rot_errors = []
    for i in range(num_frames):
        gt_rot = Rotation.from_quat(gt_short[i, 3:7])
        pred_rot = Rotation.from_quat(pred_short[i, 3:7])
        # Compute relative rotation
        rel_rot = gt_rot.inv() * pred_rot
        # Get angle of rotation
        angle = np.abs(rel_rot.magnitude() * 180 / np.pi)
        rot_errors.append(angle)
    
    mean_rot_error = np.mean(rot_errors)
    
    ax.set_xlabel('Roll (degrees)')
    ax.set_ylabel('Pitch (degrees)')
    ax.set_zlabel('Yaw (degrees)')
    ax.set_title(f'First {duration_seconds} Seconds Rotation Trajectory\nMean Angular Error: {mean_rot_error:.2f}°')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / f'rotation_trajectory_{duration_seconds}s_3d.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Translation components over time
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    time = np.arange(num_frames) / fps  # Time in seconds
    
    for i, (ax, label) in enumerate(zip(axes, ['X', 'Y', 'Z'])):
        ax.plot(time, gt_short[:, i], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.plot(time, pred_short[:, i], 'r-', linewidth=2, label='Prediction', alpha=0.8)
        
        # Calculate per-axis error
        axis_error = np.abs(pred_short[:, i] - gt_short[:, i])
        mean_axis_error = np.mean(axis_error)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(f'{label} Position (cm)')
        ax.set_title(f'{label}-axis Translation (Mean Error: {mean_axis_error:.2f} cm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'translation_components_{duration_seconds}s.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Frame-to-frame motion analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Calculate frame-to-frame displacements
    gt_displacements = np.diff(gt_short[:, :3], axis=0)
    pred_displacements = np.diff(pred_short[:, :3], axis=0)
    
    gt_speeds = np.linalg.norm(gt_displacements, axis=1)
    pred_speeds = np.linalg.norm(pred_displacements, axis=1)
    
    time_diff = time[1:]  # Time for differences
    
    # Plot speeds
    ax1.plot(time_diff, gt_speeds, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax1.plot(time_diff, pred_speeds, 'r-', linewidth=2, label='Prediction', alpha=0.8)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Speed (cm/frame)')
    ax1.set_title('Frame-to-Frame Speed')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot speed error
    speed_error = np.abs(pred_speeds - gt_speeds)
    ax2.plot(time_diff, speed_error, 'g-', linewidth=2)
    ax2.axhline(np.mean(speed_error), color='r', linestyle='--', 
                label=f'Mean Error: {np.mean(speed_error):.3f} cm/frame')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Speed Error (cm/frame)')
    ax2.set_title('Frame-to-Frame Speed Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'frame_to_frame_analysis_{duration_seconds}s.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Error over time
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(time, trans_errors, 'r-', linewidth=2, alpha=0.8)
    ax.axhline(mean_error, color='b', linestyle='--', 
               label=f'Mean Error: {mean_error:.2f} cm')
    
    # Add percentile lines
    p95 = np.percentile(trans_errors, 95)
    ax.axhline(p95, color='g', linestyle=':', 
               label=f'95th Percentile: {p95:.2f} cm')
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Translation Error (cm)')
    ax.set_title(f'Translation Error Over First {duration_seconds} Seconds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / f'error_over_time_{duration_seconds}s.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\n=== Short-term Statistics ({duration_seconds} seconds) ===")
    print(f"Frames analyzed: {num_frames}")
    print(f"Mean translation error: {mean_error:.2f} cm")
    print(f"Max translation error: {np.max(trans_errors):.2f} cm")
    print(f"95th percentile error: {p95:.2f} cm")
    print(f"Mean GT speed: {np.mean(gt_speeds):.3f} cm/frame")
    print(f"Mean pred speed: {np.mean(pred_speeds):.3f} cm/frame")
    
    # AR/VR specific metrics
    print(f"\n=== AR/VR Metrics ===")
    # Check if motion is stable enough for mask reuse
    # Typical threshold: < 2cm motion for stable viewing
    stable_frames = np.sum(trans_errors < 2.0)
    print(f"Frames with <2cm error (stable for mask reuse): {stable_frames}/{num_frames} ({stable_frames/num_frames*100:.1f}%)")
    
    # Check head motion detection accuracy
    motion_detected_gt = gt_speeds > 0.5  # Motion if >0.5cm/frame
    motion_detected_pred = pred_speeds > 0.5
    motion_accuracy = np.sum(motion_detected_gt == motion_detected_pred) / len(gt_speeds)
    print(f"Motion detection accuracy: {motion_accuracy*100:.1f}%")
    
    print(f"\nPlots saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Plot short-term trajectory for AR/VR evaluation')
    parser.add_argument('--npz-file', type=str, required=True, help='NPZ results file')
    parser.add_argument('--output-dir', type=str, default='short_term_plots', 
                        help='Output directory for plots')
    parser.add_argument('--duration', type=int, default=5, 
                        help='Duration in seconds to plot (default: 5)')
    parser.add_argument('--fps', type=int, default=20, 
                        help='Frames per second (default: 20)')
    
    args = parser.parse_args()
    
    plot_short_term_trajectory(args.npz_file, args.output_dir, args.duration, args.fps)

if __name__ == "__main__":
    main()