#!/usr/bin/env python3
"""
Visualize multiple VIO trajectory results with comprehensive plots.
Creates paper-style figures, 2D trajectory comparisons, and error analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from matplotlib.gridspec import GridSpec
try:
    import seaborn as sns
    sns.set_palette("husl")
except ImportError:
    sns = None

try:
    from scipy.stats import gaussian_kde
except ImportError:
    gaussian_kde = None

# Set style for publication-quality plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('ggplot')


def load_sequence_results(results_dir, sequence_ids=None):
    """Load all sequence results from directory."""
    results_dir = Path(results_dir)
    
    if sequence_ids:
        # Load specific sequences
        result_files = [results_dir / f'seq_{seq_id}.npz' for seq_id in sequence_ids]
    else:
        # Load all sequences
        result_files = sorted(results_dir.glob('seq_*.npz'))
    
    sequences = {}
    for result_file in result_files:
        if not result_file.exists():
            print(f"Warning: {result_file} not found, skipping")
            continue
            
        seq_id = result_file.stem.replace('seq_', '')
        data = np.load(result_file, allow_pickle=True)
        
        # Extract trajectories and metrics
        sequences[seq_id] = {
            'gt': data['gt_aligned'] if 'gt_aligned' in data else data['ground_truth'],
            'pred': data['pred_aligned'] if 'pred_aligned' in data else data['absolute_trajectory'],
            'metrics': data['metrics'].item() if 'metrics' in data else {}
        }
    
    return sequences


def create_paper_style_figure(sequences, output_dir):
    """Create publication-ready multi-panel figure."""
    output_dir = Path(output_dir)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Collect all sequences for overview
    all_sequences = list(sequences.keys())
    
    # 1. Overview trajectory plot (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (seq_id, data) in enumerate(sequences.items()):
        gt = data['gt']
        pred = data['pred']
        
        # Plot with different colors for each sequence
        color = plt.cm.tab10(i % 10)
        ax1.plot(gt[:, 0], gt[:, 1], '-', color=color, alpha=0.6, linewidth=1.5)
        ax1.plot(pred[:, 0], pred[:, 1], '--', color=color, alpha=0.8, linewidth=1)
    
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title('All Trajectories Overview', fontsize=14, fontweight='bold')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='gray', linewidth=2, label='Ground Truth'),
                      Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='Prediction')]
    ax1.legend(handles=legend_elements, loc='best')
    
    # 2. ATE distribution (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    ate_values = [data['metrics'].get('ate_rmse_cm', 0) for data in sequences.values()]
    
    # Create violin plot
    parts = ax2.violinplot([ate_values], positions=[1], showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('skyblue')
        pc.set_alpha(0.7)
    
    # Add individual points
    y_positions = np.random.normal(1, 0.04, size=len(ate_values))
    ax2.scatter(y_positions, ate_values, alpha=0.8, s=50, color='darkblue')
    
    ax2.set_ylabel('ATE RMSE (m)', fontsize=12)
    ax2.set_xticks([1])
    ax2.set_xticklabels(['Test Sequences'])
    ax2.set_title('Absolute Trajectory Error Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add statistics text
    mean_ate = np.mean(ate_values)
    std_ate = np.std(ate_values)
    ax2.text(0.95, 0.95, f'Mean: {mean_ate:.6f} m\nStd: {std_ate:.6f} m',
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 3. Rotation error distribution (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    rot_values = [data['metrics'].get('rot_mean_deg', 0) for data in sequences.values()]
    
    # Histogram with KDE
    ax3.hist(rot_values, bins=15, density=True, alpha=0.6, color='coral', edgecolor='black')
    
    # Add KDE if we have enough data and gaussian_kde is available
    if len(rot_values) > 5 and gaussian_kde is not None:
        kde = gaussian_kde(rot_values)
        x_range = np.linspace(min(rot_values), max(rot_values), 100)
        ax3.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    ax3.set_xlabel('Mean Rotation Error (degrees)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Rotation Error Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Per-sequence performance (bottom left)
    ax4 = fig.add_subplot(gs[1, 0])
    seq_ids = list(sequences.keys())
    ate_values = [sequences[seq_id]['metrics'].get('ate_rmse_cm', 0) for seq_id in seq_ids]
    
    bars = ax4.bar(range(len(seq_ids)), ate_values, color='lightgreen', edgecolor='darkgreen')
    
    # Color bars based on performance
    for i, (bar, ate) in enumerate(zip(bars, ate_values)):
        if ate > 0.1:  # More than 10cm error
            bar.set_facecolor('lightcoral')
        elif ate > 0.05:  # 5-10cm error
            bar.set_facecolor('lightyellow')
    
    ax4.set_xlabel('Sequence ID', fontsize=12)
    ax4.set_ylabel('ATE RMSE (m)', fontsize=12)
    ax4.set_title('Per-Sequence Performance', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(seq_ids)))
    ax4.set_xticklabels(seq_ids, rotation=45)
    ax4.grid(True, axis='y', alpha=0.3)
    
    # Add performance threshold line
    ax4.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='5cm threshold')
    ax4.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='10cm threshold')
    ax4.legend()
    
    # 5. Time-based RPE analysis (bottom middle)
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Collect RPE data at different time scales
    time_windows = ['50ms', '1s', '5s']
    rpe_means = {window: [] for window in time_windows}
    
    for seq_data in sequences.values():
        if 'rpe_results' in seq_data['metrics']:
            rpe = seq_data['metrics']['rpe_results']
            for window in time_windows:
                if window in rpe:
                    rpe_means[window].append(rpe[window]['trans_mean'])
    
    # Plot RPE trends
    x_pos = np.arange(len(time_windows))
    means = [np.mean(rpe_means[w]) if rpe_means[w] else 0 for w in time_windows]
    stds = [np.std(rpe_means[w]) if rpe_means[w] else 0 for w in time_windows]
    
    ax5.errorbar(x_pos, means, yerr=stds, marker='o', markersize=8, 
                 capsize=5, capthick=2, linewidth=2, color='darkblue')
    
    ax5.set_xlabel('Time Window', fontsize=12)
    ax5.set_ylabel('Translation RPE (m)', fontsize=12)
    ax5.set_title('Relative Pose Error vs Time Scale', fontsize=14, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(time_windows)
    ax5.grid(True, alpha=0.3)
    
    # 6. Best vs Worst trajectory comparison (bottom right)
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Find best and worst performing sequences
    ate_dict = {seq_id: data['metrics'].get('ate_rmse_cm', float('inf')) 
                for seq_id, data in sequences.items()}
    best_seq = min(ate_dict, key=ate_dict.get)
    worst_seq = max(ate_dict, key=ate_dict.get)
    
    # Plot best sequence
    best_gt = sequences[best_seq]['gt']
    best_pred = sequences[best_seq]['pred']
    ax6.plot(best_gt[:, 0], best_gt[:, 1], 'g-', linewidth=2, 
             label=f'Best GT (Seq {best_seq})', alpha=0.8)
    ax6.plot(best_pred[:, 0], best_pred[:, 1], 'g--', linewidth=2, 
             label=f'Best Pred (ATE: {ate_dict[best_seq]:.3f}m)', alpha=0.8)
    
    # Plot worst sequence
    worst_gt = sequences[worst_seq]['gt']
    worst_pred = sequences[worst_seq]['pred']
    ax6.plot(worst_gt[:, 0], worst_gt[:, 1], 'r-', linewidth=2, 
             label=f'Worst GT (Seq {worst_seq})', alpha=0.8)
    ax6.plot(worst_pred[:, 0], worst_pred[:, 1], 'r--', linewidth=2, 
             label=f'Worst Pred (ATE: {ate_dict[worst_seq]:.3f}m)', alpha=0.8)
    
    ax6.set_xlabel('X Position (m)', fontsize=12)
    ax6.set_ylabel('Y Position (m)', fontsize=12)
    ax6.set_title('Best vs Worst Performance', fontsize=14, fontweight='bold')
    ax6.axis('equal')
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='best', fontsize=10)
    
    # Add overall title
    fig.suptitle('Visual-Inertial Odometry Performance Analysis', fontsize=16, fontweight='bold')
    
    # Save figure
    output_path = output_dir / 'paper_style_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved paper-style figure to {output_path}")


def create_2d_trajectory_plots(sequences, output_dir):
    """Create individual 2D trajectory plots for each sequence."""
    output_dir = Path(output_dir)
    (output_dir / '2d_trajectories').mkdir(exist_ok=True)
    
    for seq_id, data in sequences.items():
        gt = data['gt']
        pred = data['pred']
        
        # Create figure with subplots - 3x3 grid for better separation
        fig, axes = plt.subplots(3, 3, figsize=(22, 18))
        fig.suptitle(f'Sequence {seq_id} - Trajectory Analysis', fontsize=16, fontweight='bold')
        
        # 1. Full trajectory plot with Translation ATE (top left)
        ax1 = axes[0, 0]
        ax1.plot(gt[:, 0], gt[:, 1], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax1.plot(pred[:, 0], pred[:, 1], 'r--', linewidth=2, label='Prediction', alpha=0.8)
        
        # Mark start and end
        ax1.scatter(gt[0, 0], gt[0, 1], c='green', s=100, marker='o', 
                   label='Start', zorder=5, edgecolor='black')
        ax1.scatter(gt[-1, 0], gt[-1, 1], c='red', s=100, marker='s', 
                   label='End', zorder=5, edgecolor='black')
        
        # Add translation ATE information
        trans_ate = data['metrics'].get('ate_rmse_cm', 0)
        ax1.set_title(f'Full Trajectory - Translation ATE: {trans_ate:.6f} m', fontsize=12)
        ax1.set_xlabel('X Position (meters)', fontsize=10)
        ax1.set_ylabel('Y Position (meters)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        ax1.axis('equal')
        
        # 2. Translation RPE@5s distribution analysis (top middle)
        ax2 = axes[0, 1]
        fps = 20  # 20 fps for Aria
        segment_5s = 5 * fps  # 100 frames
        
        # Calculate translation errors for ALL 5-second intervals
        trans_errors_5s = []
        interval_positions = []
        
        # Slide through entire sequence with 1-frame steps
        for start_idx in range(len(gt) - segment_5s + 1):
            end_idx = start_idx + segment_5s
            
            # Calculate translation error for this 5s window
            gt_start, gt_end = gt[start_idx, :3], gt[end_idx-1, :3]
            pred_start, pred_end = pred[start_idx, :3], pred[end_idx-1, :3]
            
            # Relative motion over 5 seconds
            gt_motion = gt_end - gt_start
            pred_motion = pred_end - pred_start
            trans_error = np.linalg.norm(pred_motion - gt_motion)
            
            trans_errors_5s.append(trans_error)
            interval_positions.append(start_idx / fps)  # Convert to seconds
        
        # Plot histogram of translation errors
        ax2.hist(trans_errors_5s, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Add statistics
        mean_trans = np.mean(trans_errors_5s)
        std_trans = np.std(trans_errors_5s)
        median_trans = np.median(trans_errors_5s)
        p95_trans = np.percentile(trans_errors_5s, 95)
        
        # Add vertical lines for statistics
        ax2.axvline(mean_trans, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_trans:.4f} m')
        ax2.axvline(median_trans, color='green', linestyle='--', linewidth=2, label=f'Median: {median_trans:.4f} m')
        ax2.axvline(p95_trans, color='orange', linestyle='--', linewidth=2, label=f'95%: {p95_trans:.4f} m')
        
        # Add RPE@5s value from metrics
        rpe_5s_trans = data['metrics'].get('rpe_results', {}).get('5s', {}).get('trans_mean', 0)
        
        ax2.set_xlabel('Translation Error (meters)', fontsize=10)
        ax2.set_ylabel('Density', fontsize=10)
        ax2.set_title(f'Translation RPE@5s Distribution ({len(trans_errors_5s)} intervals)\nRPE: {rpe_5s_trans:.6f} m', 
                     fontsize=12, color='blue')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        
        # Add text box with statistics
        stats_text = f'Intervals: {len(trans_errors_5s)}\nStd: {std_trans:.4f} m'
        ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=8)
        
        # 3. Rotation ATE visualization (top right)
        ax3 = axes[0, 2]
        # Calculate rotation error distribution
        rot_errors = []
        for i in range(1, min(len(gt), len(pred))):
            if gt.shape[1] >= 7 and pred.shape[1] >= 7:
                from scipy.spatial.transform import Rotation
                try:
                    gt_rot = Rotation.from_quat(gt[i, 3:7])
                    pred_rot = Rotation.from_quat(pred[i, 3:7])
                    rel_rot = pred_rot * gt_rot.inv()
                    angle_error = np.degrees(np.abs(rel_rot.magnitude()))
                    rot_errors.append(angle_error)
                except:
                    pass
        
        if rot_errors:
            # Histogram of rotation errors
            ax3.hist(rot_errors, bins=30, density=True, alpha=0.7, color='purple', edgecolor='black')
            ax3.axvline(np.mean(rot_errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rot_errors):.2f}°')
            ax3.axvline(np.sqrt(np.mean(np.square(rot_errors))), color='orange', linestyle='--', linewidth=2, label=f'RMSE: {np.sqrt(np.mean(np.square(rot_errors))):.2f}°')
            
            rot_ate = data['metrics'].get('rot_rmse_deg', np.sqrt(np.mean(np.square(rot_errors))))
            ax3.set_title(f'Rotation Error Distribution - ATE (RMSE): {rot_ate:.2f}°', fontsize=12)
            ax3.set_xlabel('Rotation Error (degrees)', fontsize=10)
            ax3.set_ylabel('Density', fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # 4. Rotation RPE@5s distribution analysis (middle left)
        ax4 = axes[1, 0]
        
        # Calculate rotation errors for ALL 5-second intervals
        rot_errors_5s = []
        
        # Slide through entire sequence with 1-frame steps
        for start_idx in range(len(gt) - segment_5s + 1):
            end_idx = start_idx + segment_5s
            
            if gt.shape[1] >= 7 and pred.shape[1] >= 7:
                try:
                    from scipy.spatial.transform import Rotation
                    # Get rotations at start and end of 5s window
                    gt_rot_start = Rotation.from_quat(gt[start_idx, 3:7])
                    gt_rot_end = Rotation.from_quat(gt[end_idx-1, 3:7])
                    pred_rot_start = Rotation.from_quat(pred[start_idx, 3:7])
                    pred_rot_end = Rotation.from_quat(pred[end_idx-1, 3:7])
                    
                    # Relative rotation over 5 seconds
                    gt_rel_rot = gt_rot_start.inv() * gt_rot_end
                    pred_rel_rot = pred_rot_start.inv() * pred_rot_end
                    
                    # Error between relative rotations
                    error_rot = pred_rel_rot * gt_rel_rot.inv()
                    angle_error = np.degrees(np.abs(error_rot.magnitude()))
                    rot_errors_5s.append(angle_error)
                except:
                    pass
        
        if rot_errors_5s:
            # Plot histogram of rotation errors
            ax4.hist(rot_errors_5s, bins=50, density=True, alpha=0.7, color='coral', edgecolor='black')
            
            # Add statistics
            mean_rot = np.mean(rot_errors_5s)
            std_rot = np.std(rot_errors_5s)
            median_rot = np.median(rot_errors_5s)
            p95_rot = np.percentile(rot_errors_5s, 95)
            
            # Add vertical lines for statistics
            ax4.axvline(mean_rot, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rot:.2f}°')
            ax4.axvline(median_rot, color='green', linestyle='--', linewidth=2, label=f'Median: {median_rot:.2f}°')
            ax4.axvline(p95_rot, color='orange', linestyle='--', linewidth=2, label=f'95%: {p95_rot:.2f}°')
            
            ax4.set_xlabel('Rotation Error (degrees)', fontsize=10)
            ax4.set_ylabel('Density', fontsize=10)
            ax4.legend(fontsize=8)
            
            # Add text box with statistics
            stats_text = f'Intervals: {len(rot_errors_5s)}\nStd: {std_rot:.2f}°'
            ax4.text(0.95, 0.95, stats_text, transform=ax4.transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=8)
        
        # Add RPE@5s rotation value from metrics
        rpe_5s_rot = data['metrics'].get('rpe_results', {}).get('5s', {}).get('rot_mean', 0)
        ax4.set_title(f'Rotation RPE@5s Distribution ({len(rot_errors_5s)} intervals)\nRPE: {rpe_5s_rot:.2f}°', 
                     fontsize=12, color='red')
        ax4.grid(True, alpha=0.3)
        
        # 5. Translation RPE@10s distribution analysis (middle middle)
        ax5 = axes[1, 1]
        segment_10s = 10 * fps  # 200 frames
        
        # Calculate translation errors for ALL 10-second intervals
        trans_errors_10s = []
        
        # Slide through entire sequence with 1-frame steps
        for start_idx in range(len(gt) - segment_10s + 1):
            end_idx = start_idx + segment_10s
            
            # Calculate translation error for this 10s window
            gt_start, gt_end = gt[start_idx, :3], gt[end_idx-1, :3]
            pred_start, pred_end = pred[start_idx, :3], pred[end_idx-1, :3]
            
            # Relative motion over 10 seconds
            gt_motion = gt_end - gt_start
            pred_motion = pred_end - pred_start
            trans_error = np.linalg.norm(pred_motion - gt_motion)
            
            trans_errors_10s.append(trans_error)
        
        # Plot histogram of translation errors
        ax5.hist(trans_errors_10s, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Add statistics
        mean_trans_10s = np.mean(trans_errors_10s)
        std_trans_10s = np.std(trans_errors_10s)
        median_trans_10s = np.median(trans_errors_10s)
        p95_trans_10s = np.percentile(trans_errors_10s, 95)
        
        # Add vertical lines for statistics
        ax5.axvline(mean_trans_10s, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_trans_10s:.4f} m')
        ax5.axvline(median_trans_10s, color='green', linestyle='--', linewidth=2, label=f'Median: {median_trans_10s:.4f} m')
        ax5.axvline(p95_trans_10s, color='orange', linestyle='--', linewidth=2, label=f'95%: {p95_trans_10s:.4f} m')
        
        # Add RPE@10s value from metrics
        rpe_10s_trans = data['metrics'].get('rpe_results', {}).get('10s', {}).get('trans_mean', 0)
        
        ax5.set_xlabel('Translation Error (meters)', fontsize=10)
        ax5.set_ylabel('Density', fontsize=10)
        ax5.set_title(f'Translation RPE@10s Distribution ({len(trans_errors_10s)} intervals)\nRPE: {rpe_10s_trans:.6f} m', 
                     fontsize=12, color='blue')
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=8)
        
        # Add text box with statistics
        stats_text = f'Intervals: {len(trans_errors_10s)}\nStd: {std_trans_10s:.4f} m'
        ax5.text(0.95, 0.95, stats_text, transform=ax5.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=8)
        
        # 6. Rotation RPE@10s distribution analysis (middle right)
        ax6 = axes[1, 2]
        
        # Calculate rotation errors for ALL 10-second intervals
        rot_errors_10s = []
        
        # Slide through entire sequence with 1-frame steps
        for start_idx in range(len(gt) - segment_10s + 1):
            end_idx = start_idx + segment_10s
            
            if gt.shape[1] >= 7 and pred.shape[1] >= 7:
                try:
                    # Get rotations at start and end of 10s window
                    gt_rot_start = Rotation.from_quat(gt[start_idx, 3:7])
                    gt_rot_end = Rotation.from_quat(gt[end_idx-1, 3:7])
                    pred_rot_start = Rotation.from_quat(pred[start_idx, 3:7])
                    pred_rot_end = Rotation.from_quat(pred[end_idx-1, 3:7])
                    
                    # Relative rotation over 10 seconds
                    gt_rel_rot = gt_rot_start.inv() * gt_rot_end
                    pred_rel_rot = pred_rot_start.inv() * pred_rot_end
                    
                    # Error between relative rotations
                    error_rot = pred_rel_rot * gt_rel_rot.inv()
                    angle_error = np.degrees(np.abs(error_rot.magnitude()))
                    rot_errors_10s.append(angle_error)
                except:
                    pass
        
        if rot_errors_10s:
            # Plot histogram of rotation errors
            ax6.hist(rot_errors_10s, bins=50, density=True, alpha=0.7, color='coral', edgecolor='black')
            
            # Add statistics
            mean_rot_10s = np.mean(rot_errors_10s)
            std_rot_10s = np.std(rot_errors_10s)
            median_rot_10s = np.median(rot_errors_10s)
            p95_rot_10s = np.percentile(rot_errors_10s, 95)
            
            # Add vertical lines for statistics
            ax6.axvline(mean_rot_10s, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rot_10s:.2f}°')
            ax6.axvline(median_rot_10s, color='green', linestyle='--', linewidth=2, label=f'Median: {median_rot_10s:.2f}°')
            ax6.axvline(p95_rot_10s, color='orange', linestyle='--', linewidth=2, label=f'95%: {p95_rot_10s:.2f}°')
            
            ax6.set_xlabel('Rotation Error (degrees)', fontsize=10)
            ax6.set_ylabel('Density', fontsize=10)
            ax6.legend(fontsize=8)
            
            # Add text box with statistics
            stats_text = f'Intervals: {len(rot_errors_10s)}\nStd: {std_rot_10s:.2f}°'
            ax6.text(0.95, 0.95, stats_text, transform=ax6.transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=8)
        
        # Add RPE@10s rotation value from metrics
        rpe_10s_rot = data['metrics'].get('rpe_results', {}).get('10s', {}).get('rot_mean', 0)
        ax6.set_title(f'Rotation RPE@10s Distribution ({len(rot_errors_10s)} intervals)\nRPE: {rpe_10s_rot:.2f}°', 
                     fontsize=12, color='red')
        ax6.grid(True, alpha=0.3)
        
        # 7. Translation RPE over different time windows (bottom left)
        ax7 = axes[2, 0]
        
        # Get RPE data for different time windows
        rpe_data = data['metrics'].get('rpe_results', {})
        time_windows = ['50ms', '500ms', '1s', '5s', '10s']
        trans_rpe_values = []
        window_labels = []
        
        for window in time_windows:
            if window in rpe_data:
                trans_rpe_values.append(rpe_data[window]['trans_mean'])
                window_labels.append(window)
        
        if trans_rpe_values:
            x_pos = np.arange(len(window_labels))
            bars = ax7.bar(x_pos, trans_rpe_values, color='steelblue', alpha=0.8)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, trans_rpe_values)):
                ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.4f}m', ha='center', va='bottom', fontsize=8)
            
            ax7.set_xlabel('Time Window', fontsize=10)
            ax7.set_ylabel('Translation RPE (meters)', fontsize=10)
            ax7.set_title('Translation RPE at Different Time Scales', fontsize=12)
            ax7.set_xticks(x_pos)
            ax7.set_xticklabels(window_labels)
            ax7.grid(True, axis='y', alpha=0.3)
        
        # 8. Rotation RPE over different time windows (bottom middle)
        ax8 = axes[2, 1]
        
        rot_rpe_values = []
        for window in time_windows:
            if window in rpe_data:
                rot_rpe_values.append(rpe_data[window]['rot_mean'])
        
        if rot_rpe_values:
            x_pos = np.arange(len(window_labels))
            bars = ax8.bar(x_pos, rot_rpe_values, color='coral', alpha=0.8)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, rot_rpe_values)):
                ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.2f}°', ha='center', va='bottom', fontsize=8)
            
            ax8.set_xlabel('Time Window', fontsize=10)
            ax8.set_ylabel('Rotation RPE (degrees)', fontsize=10)
            ax8.set_title('Rotation RPE at Different Time Scales', fontsize=12)
            ax8.set_xticks(x_pos)
            ax8.set_xticklabels(window_labels)
            ax8.grid(True, axis='y', alpha=0.3)
        
        # 9. Rotation trajectory visualization (bottom right)
        ax9 = axes[2, 2]
        
        # Extract heading angles from quaternions
        if gt.shape[1] >= 7 and pred.shape[1] >= 7:
            from scipy.spatial.transform import Rotation
            
            # Convert quaternions to Euler angles (we'll use yaw/heading)
            gt_headings = []
            pred_headings = []
            
            for i in range(len(gt)):
                try:
                    # Get yaw angle (rotation around z-axis)
                    gt_euler = Rotation.from_quat(gt[i, 3:7]).as_euler('xyz', degrees=True)
                    pred_euler = Rotation.from_quat(pred[i, 3:7]).as_euler('xyz', degrees=True)
                    gt_headings.append(gt_euler[2])  # z-axis rotation (yaw)
                    pred_headings.append(pred_euler[2])
                except:
                    gt_headings.append(0)
                    pred_headings.append(0)
            
            # Create rotation trajectory plot
            # Plot trajectory colored by heading difference
            heading_diff = np.array(pred_headings) - np.array(gt_headings)
            # Normalize angle differences to [-180, 180]
            heading_diff = (heading_diff + 180) % 360 - 180
            
            # Create scatter plot with color based on heading error
            scatter = ax9.scatter(gt[:, 0], gt[:, 1], c=np.abs(heading_diff), 
                                cmap='YlOrRd', s=20, alpha=0.8, vmin=0, vmax=30)
            
            # Add start and end orientation indicators
            # Start position with arrow showing initial heading
            start_length = 0.5  # Arrow length in meters
            ax9.arrow(gt[0, 0], gt[0, 1], 
                     start_length * np.cos(np.radians(gt_headings[0])), 
                     start_length * np.sin(np.radians(gt_headings[0])),
                     head_width=0.2, head_length=0.1, fc='green', ec='darkgreen',
                     linewidth=2, zorder=10)
            ax9.arrow(pred[0, 0], pred[0, 1], 
                     start_length * np.cos(np.radians(pred_headings[0])), 
                     start_length * np.sin(np.radians(pred_headings[0])),
                     head_width=0.2, head_length=0.1, fc='lightgreen', ec='darkgreen',
                     linewidth=2, linestyle='--', zorder=9)
            
            # End position with arrow showing final heading
            ax9.arrow(gt[-1, 0], gt[-1, 1], 
                     start_length * np.cos(np.radians(gt_headings[-1])), 
                     start_length * np.sin(np.radians(gt_headings[-1])),
                     head_width=0.2, head_length=0.1, fc='red', ec='darkred',
                     linewidth=2, zorder=10)
            ax9.arrow(pred[-1, 0], pred[-1, 1], 
                     start_length * np.cos(np.radians(pred_headings[-1])), 
                     start_length * np.sin(np.radians(pred_headings[-1])),
                     head_width=0.2, head_length=0.1, fc='lightcoral', ec='darkred',
                     linewidth=2, linestyle='--', zorder=9)
            
            # Add orientation arrows along the path at regular intervals
            step = max(1, len(gt) // 20)  # Show ~20 arrows along the path
            arrow_length = 0.3  # Smaller arrows for path
            
            for i in range(0, len(gt), step):
                if i > 0 and i < len(gt) - 1:  # Skip start and end
                    # Ground truth orientation
                    ax9.arrow(gt[i, 0], gt[i, 1], 
                             arrow_length * np.cos(np.radians(gt_headings[i])), 
                             arrow_length * np.sin(np.radians(gt_headings[i])),
                             head_width=0.1, head_length=0.05, fc='blue', ec='blue',
                             alpha=0.6, zorder=5)
                    # Prediction orientation
                    ax9.arrow(pred[i, 0], pred[i, 1], 
                             arrow_length * np.cos(np.radians(pred_headings[i])), 
                             arrow_length * np.sin(np.radians(pred_headings[i])),
                             head_width=0.1, head_length=0.05, fc='orange', ec='orange',
                             alpha=0.6, zorder=4)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax9)
            cbar.set_label('Heading Error (°)', fontsize=8)
            
            # Calculate heading RMSE
            heading_rmse = np.sqrt(np.mean(heading_diff**2))
            
            ax9.set_title(f'Rotation Trajectory - Heading RMSE: {heading_rmse:.2f}°', fontsize=12)
            ax9.set_xlabel('X Position (meters)', fontsize=10)
            ax9.set_ylabel('Y Position (meters)', fontsize=10)
            ax9.grid(True, alpha=0.3)
            ax9.axis('equal')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', edgecolor='darkgreen', label='Start (GT)'),
                Patch(facecolor='lightgreen', edgecolor='darkgreen', label='Start (Pred)'),
                Patch(facecolor='red', edgecolor='darkred', label='End (GT)'),
                Patch(facecolor='lightcoral', edgecolor='darkred', label='End (Pred)'),
                Patch(facecolor='blue', alpha=0.6, label='GT orientations'),
                Patch(facecolor='orange', alpha=0.6, label='Pred orientations')
            ]
            ax9.legend(handles=legend_elements, loc='best', fontsize=8)
        else:
            ax9.text(0.5, 0.5, 'No rotation data available', 
                    transform=ax9.transAxes, ha='center', va='center', fontsize=12)
            ax9.axis('off')
        
        plt.tight_layout()
        
        # Save
        output_path = output_dir / '2d_trajectories' / f'trajectory_{seq_id}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved individual 2D plots to {output_dir / '2d_trajectories'}")


def create_error_analysis_plots(sequences, output_dir):
    """Create detailed error analysis plots."""
    output_dir = Path(output_dir)
    
    # Create multi-panel error analysis figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Detailed Error Analysis', fontsize=16, fontweight='bold')
    
    # 1. Frame-wise error plot (for first few sequences)
    ax1 = axes[0, 0]
    for i, (seq_id, data) in enumerate(list(sequences.items())[:5]):  # First 5 sequences
        gt = data['gt']
        pred = data['pred']
        
        # Compute frame-wise error
        errors = np.sqrt(np.sum((gt[:, :3] - pred[:, :3])**2, axis=1))
        frames = np.arange(len(errors))
        
        ax1.plot(frames[::10], errors[::10], alpha=0.7, linewidth=1.5, 
                label=f'Seq {seq_id}')
    
    ax1.set_xlabel('Frame Index', fontsize=12)
    ax1.set_ylabel('Position Error (m)', fontsize=12)
    ax1.set_title('Frame-wise Position Error', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Error vs trajectory length
    ax2 = axes[0, 1]
    lengths = []
    ate_values = []
    
    for seq_id, data in sequences.items():
        gt = data['gt']
        # Compute trajectory length
        length = np.sum(np.sqrt(np.sum(np.diff(gt[:, :3], axis=0)**2, axis=1)))
        lengths.append(length)
        ate_values.append(data['metrics'].get('ate_rmse_cm', 0))
    
    ax2.scatter(lengths, ate_values, s=60, alpha=0.7, edgecolor='black')
    
    # Add trend line
    if len(lengths) > 2:
        z = np.polyfit(lengths, ate_values, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(lengths), max(lengths), 100)
        ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8, label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
    
    ax2.set_xlabel('Trajectory Length (m)', fontsize=12)
    ax2.set_ylabel('ATE RMSE (m)', fontsize=12)
    ax2.set_title('Error vs Trajectory Length', fontsize=14)
    ax2.grid(True, alpha=0.3)
    if len(lengths) > 2:
        ax2.legend()
    
    # 3. Translation vs Rotation error scatter
    ax3 = axes[1, 0]
    trans_errors = [data['metrics'].get('ate_rmse_cm', 0) for data in sequences.values()]
    rot_errors = [data['metrics'].get('rot_mean_deg', 0) for data in sequences.values()]
    
    scatter = ax3.scatter(trans_errors, rot_errors, s=80, alpha=0.7, 
                         c=range(len(trans_errors)), cmap='viridis', edgecolor='black')
    
    # Add sequence labels
    for i, seq_id in enumerate(sequences.keys()):
        ax3.annotate(seq_id, (trans_errors[i], rot_errors[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax3.set_xlabel('Translation Error (m)', fontsize=12)
    ax3.set_ylabel('Rotation Error (degrees)', fontsize=12)
    ax3.set_title('Translation vs Rotation Error', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # 4. RPE heatmap
    ax4 = axes[1, 1]
    
    # Collect RPE data
    time_windows = ['50ms', '100ms', '250ms', '500ms', '1s']
    rpe_matrix = []
    
    for seq_id in sorted(sequences.keys()):
        seq_rpe = []
        if 'rpe_results' in sequences[seq_id]['metrics']:
            rpe = sequences[seq_id]['metrics']['rpe_results']
            for window in time_windows:
                if window in rpe:
                    seq_rpe.append(rpe[window]['trans_mean'])
                else:
                    seq_rpe.append(0)
        else:
            seq_rpe = [0] * len(time_windows)
        rpe_matrix.append(seq_rpe)
    
    # Create heatmap
    if rpe_matrix:
        im = ax4.imshow(rpe_matrix, aspect='auto', cmap='YlOrRd')
        ax4.set_xticks(range(len(time_windows)))
        ax4.set_xticklabels(time_windows)
        ax4.set_yticks(range(len(sequences)))
        ax4.set_yticklabels(sorted(sequences.keys()))
        ax4.set_xlabel('Time Window', fontsize=12)
        ax4.set_ylabel('Sequence ID', fontsize=12)
        ax4.set_title('RPE Heatmap (Translation)', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('RPE (m)', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'error_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved error analysis to {output_path}")


def create_summary_statistics(sequences, output_dir):
    """Generate summary statistics and save to JSON."""
    output_dir = Path(output_dir)
    
    # Collect all metrics
    ate_values = [data['metrics'].get('ate_rmse_cm', 0) for data in sequences.values()]
    rot_values = [data['metrics'].get('rot_mean_deg', 0) for data in sequences.values()]
    
    # Compute statistics
    stats = {
        'num_sequences': len(sequences),
        'ate_rmse': {
            'mean': float(np.mean(ate_values)),
            'std': float(np.std(ate_values)),
            'min': float(np.min(ate_values)),
            'max': float(np.max(ate_values)),
            'median': float(np.median(ate_values))
        },
        'rotation_error': {
            'mean': float(np.mean(rot_values)),
            'std': float(np.std(rot_values)),
            'min': float(np.min(rot_values)),
            'max': float(np.max(rot_values)),
            'median': float(np.median(rot_values))
        }
    }
    
    # Add RPE statistics
    rpe_stats = {}
    for window in ['50ms', '1s', '5s']:
        window_values = []
        for data in sequences.values():
            if 'rpe_results' in data['metrics'] and window in data['metrics']['rpe_results']:
                window_values.append(data['metrics']['rpe_results'][window]['trans_mean'])
        
        if window_values:
            rpe_stats[f'rpe_{window}'] = {
                'mean': float(np.mean(window_values)),
                'std': float(np.std(window_values))
            }
    
    stats['rpe'] = rpe_stats
    
    # Save to JSON
    with open(output_dir / 'error_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ATE statistics
    ax1.bar(['Mean', 'Median', 'Min', 'Max'], 
            [stats['ate_rmse']['mean'], stats['ate_rmse']['median'],
             stats['ate_rmse']['min'], stats['ate_rmse']['max']],
            color=['blue', 'green', 'lightgreen', 'red'])
    ax1.set_ylabel('ATE RMSE (m)')
    ax1.set_title('ATE Statistics Summary')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (label, value) in enumerate(zip(['Mean', 'Median', 'Min', 'Max'],
                                          [stats['ate_rmse']['mean'], stats['ate_rmse']['median'],
                                           stats['ate_rmse']['min'], stats['ate_rmse']['max']])):
        ax1.text(i, value + 0.001, f'{value:.6f}', ha='center', va='bottom')
    
    # RPE trend
    ax2.set_title('RPE Trend Over Time')
    if rpe_stats:
        windows = []
        means = []
        stds = []
        
        for window, values in rpe_stats.items():
            windows.append(window.replace('rpe_', ''))
            means.append(values['mean'])
            stds.append(values['std'])
        
        x_pos = np.arange(len(windows))
        ax2.errorbar(x_pos, means, yerr=stds, marker='o', markersize=8,
                    capsize=5, capthick=2, linewidth=2)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(windows)
        ax2.set_xlabel('Time Window')
        ax2.set_ylabel('Translation RPE (m)')
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Summary Statistics - {len(sequences)} Sequences', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'summary_statistics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary statistics to {output_dir}/error_statistics.json")
    print(f"Saved summary plot to {output_path}")


def create_multi_trajectory_comparison(sequences, output_dir, max_sequences=8):
    """Create a grid of trajectory comparisons."""
    output_dir = Path(output_dir)
    
    # Limit number of sequences for readability
    seq_items = list(sequences.items())[:max_sequences]
    n_seqs = len(seq_items)
    
    # Calculate grid dimensions
    cols = min(4, n_seqs)
    rows = (n_seqs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Individual Trajectory Comparisons', fontsize=16, fontweight='bold')
    
    for idx, (seq_id, data) in enumerate(seq_items):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        gt = data['gt']
        pred = data['pred']
        
        # Plot trajectories
        ax.plot(gt[:, 0], gt[:, 1], 'b-', linewidth=1.5, label='GT', alpha=0.8)
        ax.plot(pred[:, 0], pred[:, 1], 'r--', linewidth=1.5, label='Pred', alpha=0.8)
        
        # Mark start
        ax.scatter(gt[0, 0], gt[0, 1], c='green', s=50, marker='o', zorder=5)
        
        # Add metrics
        ate = data['metrics'].get('ate_rmse_cm', 0)
        ax.set_title(f'Seq {seq_id} (ATE: {ate:.3f}m)', fontsize=12)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        if idx == 0:
            ax.legend(loc='best', fontsize=8)
    
    # Hide empty subplots
    for idx in range(n_seqs, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'multi_trajectory_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved multi-trajectory comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize multiple VIO trajectory results')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing sequence results')
    parser.add_argument('--output-dir', type=str, default='trajectory_plots',
                        help='Directory to save plots')
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
                        help='Specific sequence IDs to visualize')
    parser.add_argument('--plot-types', type=str, nargs='+', 
                        default=['paper', '2d', 'error', 'multi'],
                        choices=['paper', '2d', 'error', 'multi', 'all'],
                        help='Types of plots to generate')
    
    args = parser.parse_args()
    
    # Handle 'all' plot type
    if 'all' in args.plot_types:
        args.plot_types = ['paper', '2d', 'error', 'multi']
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sequences
    print(f"Loading sequences from {args.results_dir}...")
    sequences = load_sequence_results(args.results_dir, args.sequences)
    
    if not sequences:
        print("No sequences found!")
        return
    
    print(f"Loaded {len(sequences)} sequences")
    
    # Generate requested plots
    if 'paper' in args.plot_types:
        print("\nGenerating paper-style figure...")
        create_paper_style_figure(sequences, output_dir)
    
    if '2d' in args.plot_types:
        print("\nGenerating 2D trajectory plots...")
        create_2d_trajectory_plots(sequences, output_dir)
    
    if 'error' in args.plot_types:
        print("\nGenerating error analysis plots...")
        create_error_analysis_plots(sequences, output_dir)
    
    if 'multi' in args.plot_types:
        print("\nGenerating multi-trajectory comparison...")
        create_multi_trajectory_comparison(sequences, output_dir)
    
    # Always generate summary statistics
    print("\nGenerating summary statistics...")
    create_summary_statistics(sequences, output_dir)
    
    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == '__main__':
    main()