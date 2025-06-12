#!/usr/bin/env python3
"""
Generate both relative motion and absolute trajectory plots
Shows ground truth vs prediction for both cases
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def plot_relative_motion(pred_poses, gt_poses, output_dir, duration=5):
    """Plot relative frame-to-frame motion"""
    fps = 20
    frames = min(duration * fps, len(pred_poses) - 1)
    
    # Calculate relative motions
    pred_relative = []
    gt_relative = []
    
    for i in range(1, frames + 1):
        # Prediction relative motion
        pred_rel_trans = pred_poses[i, :3] - pred_poses[i-1, :3]
        pred_relative.append(pred_rel_trans * 100)  # Convert to cm
        
        # GT relative motion
        gt_rel_trans = gt_poses[i, :3] - gt_poses[i-1, :3]
        gt_relative.append(gt_rel_trans * 100)  # Convert to cm
    
    pred_relative = np.array(pred_relative)
    gt_relative = np.array(gt_relative)
    
    # Create 3D plot for relative motion
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each relative motion vector
    for i in range(len(pred_relative)):
        # GT relative motion (blue)
        ax.plot([0, gt_relative[i, 0]], 
                [0, gt_relative[i, 1]], 
                [0, gt_relative[i, 2]], 
                'b-', alpha=0.3, linewidth=1)
        
        # Predicted relative motion (red)
        ax.plot([0, pred_relative[i, 0]], 
                [0, pred_relative[i, 1]], 
                [0, pred_relative[i, 2]], 
                'r-', alpha=0.3, linewidth=1)
    
    # Add legend with statistics
    gt_mean_mag = np.mean(np.linalg.norm(gt_relative, axis=1))
    pred_mean_mag = np.mean(np.linalg.norm(pred_relative, axis=1))
    
    ax.text2D(0.05, 0.95, f'Mean GT motion: {gt_mean_mag:.2f} cm/frame\n'
                          f'Mean Pred motion: {pred_mean_mag:.2f} cm/frame', 
              transform=ax.transAxes, fontsize=12,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title(f'Relative Motion Vectors (First {duration}s)')
    
    # Set equal aspect ratio
    max_range = np.max(np.abs([gt_relative, pred_relative])) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Add origin marker
    ax.scatter([0], [0], [0], color='black', s=100, marker='o', label='Origin')
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='b', lw=2, label='GT Relative Motion'),
                      Line2D([0], [0], color='r', lw=2, label='Pred Relative Motion')]
    ax.legend(handles=legend_elements)
    
    plt.savefig(output_dir / f'relative_motion_{duration}s.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_absolute_trajectory(pred_poses, gt_poses, output_dir, duration=5):
    """Plot absolute accumulated trajectory"""
    fps = 20
    frames = min(duration * fps, len(pred_poses))
    
    # Extract positions in cm
    pred_traj = pred_poses[:frames, :3] * 100
    gt_traj = gt_poses[:frames, :3] * 100
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], 
            'b-', linewidth=3, label='Ground Truth', alpha=0.8)
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], 
            'r--', linewidth=2, label='Prediction', alpha=0.8)
    
    # Add markers
    ax.scatter(*gt_traj[0], color='green', s=200, marker='o', label='Start', zorder=5)
    ax.scatter(*gt_traj[-1], color='blue', s=200, marker='s', label='GT End', zorder=5)
    ax.scatter(*pred_traj[-1], color='red', s=200, marker='^', label='Pred End', zorder=5)
    
    # Add frame markers every second
    for i in range(0, frames, fps):
        ax.scatter(*gt_traj[i], color='blue', s=50, alpha=0.5)
        ax.scatter(*pred_traj[i], color='red', s=50, alpha=0.5)
    
    # Calculate error
    error = np.mean(np.linalg.norm(pred_traj - gt_traj, axis=1))
    ax.text2D(0.05, 0.95, f'Mean Error: {error:.2f} cm\n'
                          f'End Error: {np.linalg.norm(pred_traj[-1] - gt_traj[-1]):.2f} cm',
              transform=ax.transAxes, fontsize=12,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title(f'Absolute Trajectory (First {duration}s)')
    ax.legend()
    
    # Set equal aspect ratio
    all_points = np.vstack([gt_traj, pred_traj])
    center = all_points.mean(axis=0)
    max_range = np.max(np.abs(all_points - center)) * 1.1
    
    ax.set_xlim([center[0] - max_range, center[0] + max_range])
    ax.set_ylim([center[1] - max_range, center[1] + max_range])
    ax.set_zlim([center[2] - max_range, center[2] + max_range])
    
    plt.savefig(output_dir / f'absolute_trajectory_{duration}s.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_rotation_comparison(pred_poses, gt_poses, output_dir, duration=5):
    """Plot rotation comparison"""
    fps = 20
    frames = min(duration * fps, len(pred_poses))
    
    # Extract quaternions and convert to euler angles
    pred_euler = []
    gt_euler = []
    
    for i in range(frames):
        # Prediction
        pred_q = pred_poses[i, 3:]
        pred_r = R.from_quat(pred_q)
        pred_euler.append(pred_r.as_euler('xyz', degrees=True))
        
        # Ground truth
        gt_q = gt_poses[i, 3:]
        gt_r = R.from_quat(gt_q)
        gt_euler.append(gt_r.as_euler('xyz', degrees=True))
    
    pred_euler = np.array(pred_euler)
    gt_euler = np.array(gt_euler)
    
    # Create subplots for each axis
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    time = np.arange(frames) / fps
    
    labels = ['Roll', 'Pitch', 'Yaw']
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(time, gt_euler[:, i], 'b-', linewidth=2, label='Ground Truth')
        ax.plot(time, pred_euler[:, i], 'r--', linewidth=2, label='Prediction')
        ax.set_ylabel(f'{label} (degrees)')
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate error
        error = np.mean(np.abs(pred_euler[:, i] - gt_euler[:, i]))
        ax.text(0.02, 0.95, f'Mean Error: {error:.2f}°', 
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(f'Rotation Comparison (First {duration}s)')
    plt.tight_layout()
    plt.savefig(output_dir / f'rotation_comparison_{duration}s.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz-file', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='relative_absolute_plots')
    parser.add_argument('--duration', type=int, default=5, help='Duration in seconds')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.npz_file}...")
    data = np.load(args.npz_file)
    pred_poses = data['pred_poses']
    gt_poses = data['gt_poses']
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("Generating relative motion plot...")
    plot_relative_motion(pred_poses, gt_poses, output_dir, args.duration)
    
    print("Generating absolute trajectory plot...")
    plot_absolute_trajectory(pred_poses, gt_poses, output_dir, args.duration)
    
    print("Generating rotation comparison plot...")
    plot_rotation_comparison(pred_poses, gt_poses, output_dir, args.duration)
    
    print(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    main()