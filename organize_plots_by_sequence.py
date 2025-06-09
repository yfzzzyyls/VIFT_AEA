#!/usr/bin/env python3
"""Organize plots into sequence folders with translation and rotation visualizations."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
from scipy.spatial.transform import Rotation

def create_translation_plot(pred_traj, gt_traj, metrics, output_path, seq_id):
    """
    Create clean 3D trajectory visualization for translation only.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions and convert to cm
    gt_pos = gt_traj[:, :3] * 100  # Convert m to cm
    pred_pos = pred_traj[:, :3] * 100  # Convert m to cm
    
    # Plot ground truth trajectory with thicker line
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 
            'b-', linewidth=3, label='Ground Truth', alpha=0.9)
    
    # Plot predicted trajectory with thinner line to see overlap
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 
            'r-', linewidth=2, label='Prediction', alpha=0.8)
    
    # Mark start and end points
    ax.scatter(gt_pos[0, 0], gt_pos[0, 1], gt_pos[0, 2], 
              c='green', s=200, marker='o', edgecolor='black', 
              linewidth=2, label='Start', depthshade=False)
    ax.scatter(gt_pos[-1, 0], gt_pos[-1, 1], gt_pos[-1, 2], 
              c='red', s=200, marker='s', edgecolor='black', 
              linewidth=2, label='End', depthshade=False)
    
    # Get metrics
    ate_trans = metrics.get('ate_rmse_cm', 0)
    rpe_5s_trans = metrics.get('rpe_results', {}).get('5s', {}).get('trans_mean', 0)
    
    # Set title
    title = f'Sequence {seq_id} - Translation Trajectory\n'
    title += f'Translation ATE: {ate_trans:.3f} cm  |  '
    title += f'Translation RPE@5s: {rpe_5s_trans:.3f} cm'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Set labels
    ax.set_xlabel('X Position (cm)', fontsize=14, labelpad=10)
    ax.set_ylabel('Y Position (cm)', fontsize=14, labelpad=10)
    ax.set_zlabel('Z Position (cm)', fontsize=14, labelpad=10)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Grid settings
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Legend
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    
    # Set equal aspect ratio
    max_range = np.array([
        gt_pos[:, 0].max() - gt_pos[:, 0].min(),
        gt_pos[:, 1].max() - gt_pos[:, 1].min(),
        gt_pos[:, 2].max() - gt_pos[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (gt_pos[:, 0].max() + gt_pos[:, 0].min()) * 0.5
    mid_y = (gt_pos[:, 1].max() + gt_pos[:, 1].min()) * 0.5
    mid_z = (gt_pos[:, 2].max() + gt_pos[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved translation plot: {output_path}")


def create_rotation_euler_plot(pred_traj, gt_traj, metrics, output_path, seq_id):
    """
    Create 3D visualization of rotation trajectories using euler angles.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract quaternions and convert to euler angles for visualization
    gt_euler = []
    pred_euler = []
    
    for i in range(len(gt_traj)):
        if gt_traj.shape[1] >= 7 and pred_traj.shape[1] >= 7:
            try:
                # Convert quaternions to euler angles (in degrees)
                gt_rot = Rotation.from_quat(gt_traj[i, 3:7])
                pred_rot = Rotation.from_quat(pred_traj[i, 3:7])
                
                # Use euler angles in degrees for better visualization
                gt_angles = gt_rot.as_euler('xyz', degrees=True)
                pred_angles = pred_rot.as_euler('xyz', degrees=True)
                
                gt_euler.append(gt_angles)
                pred_euler.append(pred_angles)
            except:
                gt_euler.append([0, 0, 0])
                pred_euler.append([0, 0, 0])
    
    gt_euler = np.array(gt_euler)
    pred_euler = np.array(pred_euler)
    
    # Plot ground truth rotation trajectory
    ax.plot(gt_euler[:, 0], gt_euler[:, 1], gt_euler[:, 2], 
            'b-', linewidth=3, label='Ground Truth', alpha=0.9)
    
    # Plot predicted rotation trajectory
    ax.plot(pred_euler[:, 0], pred_euler[:, 1], pred_euler[:, 2], 
            'r-', linewidth=2, label='Prediction', alpha=0.8)
    
    # Mark start and end points
    ax.scatter(gt_euler[0, 0], gt_euler[0, 1], gt_euler[0, 2], 
              c='green', s=200, marker='o', edgecolor='black', 
              linewidth=2, label='Start', depthshade=False)
    ax.scatter(gt_euler[-1, 0], gt_euler[-1, 1], gt_euler[-1, 2], 
              c='red', s=200, marker='s', edgecolor='black', 
              linewidth=2, label='End', depthshade=False)
    
    # Get metrics
    ate_rot = metrics.get('rot_rmse_deg', 0)
    rpe_5s_rot = metrics.get('rpe_results', {}).get('5s', {}).get('rot_mean', 0)
    
    # Set title
    title = f'Sequence {seq_id} - Rotation Trajectory (Euler Angles)\n'
    title += f'Rotation ATE: {ate_rot:.3f}°  |  Rotation RPE@5s: {rpe_5s_rot:.3f}°'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Set labels
    ax.set_xlabel('Roll (degrees)', fontsize=14, labelpad=10)
    ax.set_ylabel('Pitch (degrees)', fontsize=14, labelpad=10)
    ax.set_zlabel('Yaw (degrees)', fontsize=14, labelpad=10)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Grid settings
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Legend
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved rotation plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Organize plots by sequence')
    parser.add_argument('--results-dir', type=str, 
                        default='inference_results_realtime_all_stride_1',
                        help='Directory containing sequence results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / 'sequence_plots'
    
    print(f"Loading results from: {results_dir}")
    print(f"Organizing plots in: {output_dir}")
    
    # Process all sequence files
    result_files = sorted(results_dir.glob('seq_*.npz'))
    
    if not result_files:
        print("No sequence results found!")
        return
    
    print(f"Found {len(result_files)} sequences")
    
    for result_file in result_files:
        seq_id = result_file.stem.replace('seq_', '')
        print(f"\nProcessing sequence {seq_id}...")
        
        # Create sequence-specific folder
        seq_dir = output_dir / f'seq_{seq_id}'
        seq_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        data = np.load(result_file, allow_pickle=True)
        
        # Get trajectories
        gt_traj = data['ground_truth']
        pred_traj = data['absolute_trajectory']
        metrics = data['metrics'].item()
        
        # Generate translation plot
        trans_plot_path = seq_dir / 'translation.png'
        create_translation_plot(pred_traj, gt_traj, metrics, trans_plot_path, seq_id)
        
        # Generate rotation euler plot
        rot_plot_path = seq_dir / 'rotation_euler.png'
        create_rotation_euler_plot(pred_traj, gt_traj, metrics, rot_plot_path, seq_id)
    
    print(f"\n✅ All visualizations organized! Check {output_dir}/")


if __name__ == '__main__':
    main()