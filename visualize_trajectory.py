#!/usr/bin/env python3
"""
Visualize trajectory results from full sequence inference.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path


def plot_trajectory(results_path):
    """Plot predicted vs ground truth trajectory."""
    
    # Load results
    data = np.load(results_path, allow_pickle=True)
    pred_traj = data['absolute_trajectory']
    gt_traj = data['ground_truth']
    
    # Handle metrics - might be a dict or not present
    if 'metrics' in data:
        try:
            metrics = data['metrics'].item()
        except:
            # Calculate metrics if not stored properly
            errors = np.linalg.norm(pred_traj[:, :3] - gt_traj[:, :3], axis=1)
            metrics = {
                'ate_mean': errors.mean(),
                'ate_std': errors.std()
            }
    else:
        # Calculate metrics if not present
        errors = np.linalg.norm(pred_traj[:, :3] - gt_traj[:, :3], axis=1)
        metrics = {
            'ate_mean': errors.mean(),
            'ate_std': errors.std()
        }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], 'g-', label='Ground Truth', linewidth=2)
    ax1.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], 'b--', label='Predicted', linewidth=2)
    ax1.set_xlabel('X (cm)')
    ax1.set_ylabel('Y (cm)')
    ax1.set_zlabel('Z (cm)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # Top-down view (X-Y plane)
    ax2 = fig.add_subplot(222)
    ax2.plot(gt_traj[:, 0], gt_traj[:, 1], 'g-', label='Ground Truth', linewidth=2)
    ax2.plot(pred_traj[:, 0], pred_traj[:, 1], 'b--', label='Predicted', linewidth=2)
    ax2.set_xlabel('X (cm)')
    ax2.set_ylabel('Y (cm)')
    ax2.set_title('Top-Down View (X-Y)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # Side view (X-Z plane)
    ax3 = fig.add_subplot(223)
    ax3.plot(gt_traj[:, 0], gt_traj[:, 2], 'g-', label='Ground Truth', linewidth=2)
    ax3.plot(pred_traj[:, 0], pred_traj[:, 2], 'b--', label='Predicted', linewidth=2)
    ax3.set_xlabel('X (cm)')
    ax3.set_ylabel('Z (cm)')
    ax3.set_title('Side View (X-Z)')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
    # Error over time
    ax4 = fig.add_subplot(224)
    errors = np.linalg.norm(pred_traj[:, :3] - gt_traj[:, :3], axis=1)
    frames = np.arange(len(errors))
    ax4.plot(frames, errors, 'r-', linewidth=2)
    ax4.axhline(y=metrics['ate_mean'], color='k', linestyle='--', label=f'Mean: {metrics["ate_mean"]:.2f} cm')
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Error (cm)')
    ax4.set_title('Trajectory Error Over Time')
    ax4.legend()
    ax4.grid(True)
    
    # Add overall title with metrics
    fig.suptitle(f'Trajectory Visualization - ATE: {metrics["ate_mean"]:.2f} ± {metrics["ate_std"]:.2f} cm', 
                 fontsize=16)
    
    plt.tight_layout()
    
    # Save figure
    output_path = results_path.stem + '_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    
    plt.show()


def plot_error_distribution(results_path):
    """Plot error distribution histogram."""
    
    # Load results
    data = np.load(results_path)
    pred_traj = data['absolute_trajectory']
    gt_traj = data['ground_truth']
    
    # Calculate errors
    errors = np.linalg.norm(pred_traj[:, :3] - gt_traj[:, :3], axis=1)
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    
    # Add statistics
    mean_error = errors.mean()
    median_error = np.median(errors)
    p95_error = np.percentile(errors, 95)
    
    plt.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.2f} cm')
    plt.axvline(median_error, color='green', linestyle='--', linewidth=2, label=f'Median: {median_error:.2f} cm')
    plt.axvline(p95_error, color='orange', linestyle='--', linewidth=2, label=f'95th %ile: {p95_error:.2f} cm')
    
    plt.xlabel('Trajectory Error (cm)')
    plt.ylabel('Density')
    plt.title('Trajectory Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_path = results_path.stem + '_error_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved error distribution to {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize trajectory results')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to results .npz file')
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file {results_path} not found!")
        return
    
    print(f"Visualizing results from {results_path}")
    
    # Create visualizations
    plot_trajectory(results_path)
    plot_error_distribution(results_path)


if __name__ == "__main__":
    main()