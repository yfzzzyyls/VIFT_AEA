#!/usr/bin/env python3
"""
Visualize trajectory comparison between ground truth and predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def plot_trajectory_2d(gt_poses, pred_poses, sequence_id, output_dir):
    """Create 2D trajectory plot comparing ground truth vs prediction"""
    
    # Extract x, y positions
    gt_x, gt_y = gt_poses[:, 0], gt_poses[:, 1]
    pred_x, pred_y = pred_poses[:, 0], pred_poses[:, 1]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot trajectories
    plt.plot(gt_x, gt_y, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
    plt.plot(pred_x, pred_y, 'r--', linewidth=2, label='Prediction', alpha=0.8)
    
    # Mark start and end points
    plt.scatter(gt_x[0], gt_y[0], c='green', s=100, marker='o', label='Start', zorder=5)
    plt.scatter(gt_x[-1], gt_y[-1], c='red', s=100, marker='s', label='End', zorder=5)
    
    # Calculate error statistics
    errors = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
    rmse = np.sqrt(np.mean(errors**2))
    
    # Add title and labels
    plt.title(f'Trajectory Comparison - Sequence {sequence_id}\nRMSE: {rmse:.6f} m', fontsize=14)
    plt.xlabel('X Position (meters)', fontsize=12)
    plt.ylabel('Y Position (meters)', fontsize=12)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.axis('equal')
    
    # Save plot
    output_path = Path(output_dir) / f'trajectory_{sequence_id}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved trajectory plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize VIO trajectories')
    parser.add_argument('--results-dir', type=str, 
                        default='inference_results_realtime_all_stride_1',
                        help='Directory containing inference results')
    parser.add_argument('--output-dir', type=str, 
                        default='trajectory_plots',
                        help='Directory to save plots')
    parser.add_argument('--sequence-id', type=str, default=None,
                        help='Specific sequence to visualize (default: all)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find result files
    results_dir = Path(args.results_dir)
    if args.sequence_id:
        result_files = [results_dir / f'seq_{args.sequence_id}.npz']
    else:
        result_files = sorted(results_dir.glob('seq_*.npz'))
    
    # Process each sequence
    for result_file in result_files:
        if not result_file.exists():
            print(f"Skipping {result_file} - not found")
            continue
            
        # Extract sequence ID
        seq_id = result_file.stem.replace('seq_', '')
        
        # Load data
        data = np.load(result_file)
        gt_poses = data['gt_aligned'] if 'gt_aligned' in data else data['ground_truth']
        pred_poses = data['pred_aligned'] if 'pred_aligned' in data else data['absolute_trajectory']
        
        # Create plot
        plot_trajectory_2d(gt_poses, pred_poses, seq_id, output_dir)
    
    print(f"\nAll plots saved to {output_dir}/")

if __name__ == '__main__':
    main()