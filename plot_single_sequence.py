#!/usr/bin/env python3
"""Plot a single sequence result"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Import the plotting functions from organize_plots_by_sequence
from organize_plots_by_sequence import create_translation_plot, create_rotation_euler_plot

def main():
    parser = argparse.ArgumentParser(description='Plot single sequence')
    parser.add_argument('--npz-file', type=str, required=True,
                        help='Path to NPZ result file')
    parser.add_argument('--output-dir', type=str, default='single_seq_plots',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.npz_file}...")
    data = np.load(args.npz_file, allow_pickle=True)
    
    # Extract sequence ID from filename
    filename = Path(args.npz_file).stem
    if 'seq_' in filename:
        seq_id = filename.split('seq_')[1].split('_')[0]
    else:
        seq_id = 'unknown'
    
    print(f"Sequence ID: {seq_id}")
    
    # Get trajectories and metrics
    gt_traj = data['ground_truth']
    pred_traj = data['absolute_trajectory']
    metrics = data['metrics'].item()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("Generating translation plot...")
    trans_plot_path = output_dir / f'seq_{seq_id}_translation.png'
    create_translation_plot(pred_traj, gt_traj, metrics, trans_plot_path, seq_id)
    
    print("Generating rotation plot...")
    rot_plot_path = output_dir / f'seq_{seq_id}_rotation_euler.png'
    create_rotation_euler_plot(pred_traj, gt_traj, metrics, rot_plot_path, seq_id)
    
    print(f"\nPlots saved to {output_dir}/")

if __name__ == '__main__':
    main()