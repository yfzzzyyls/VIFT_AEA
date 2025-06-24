"""
Visualization utilities for trajectories and results.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional, Tuple
import seaborn as sns


def plot_trajectory_comparison(
    pred_trajectory: np.ndarray,
    gt_trajectory: np.ndarray,
    title: str = "Trajectory Comparison",
    save_path: Optional[str] = None
) -> None:
    """
    Plot predicted vs ground truth trajectory in 2D (bird's eye view).
    
    Args:
        pred_trajectory: Predicted trajectory [N, 3]
        gt_trajectory: Ground truth trajectory [N, 3]
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot trajectories
    plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'g-', 
             linewidth=2, label='Ground Truth')
    plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r--', 
             linewidth=2, label='Predicted')
    
    # Mark start and end
    plt.scatter(gt_trajectory[0, 0], gt_trajectory[0, 1], 
                c='green', s=100, marker='o', label='Start')
    plt.scatter(gt_trajectory[-1, 0], gt_trajectory[-1, 1], 
                c='blue', s=100, marker='s', label='End (GT)')
    plt.scatter(pred_trajectory[-1, 0], pred_trajectory[-1, 1], 
                c='red', s=100, marker='^', label='End (Pred)')
    
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_3d_trajectory(
    trajectories: List[Tuple[np.ndarray, str, str]],
    title: str = "3D Trajectory",
    save_path: Optional[str] = None
) -> None:
    """
    Plot multiple trajectories in 3D.
    
    Args:
        trajectories: List of (trajectory, label, color) tuples
        title: Plot title
        save_path: Path to save the plot
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for traj, label, color in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=color, linewidth=2, label=label)
    
    # Mark start
    if trajectories:
        ax.scatter(trajectories[0][0][0, 0], 
                  trajectories[0][0][0, 1], 
                  trajectories[0][0][0, 2], 
                  c='green', s=100, marker='o', label='Start')
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title(title)
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([
        traj[:, 0].max() - traj[:, 0].min() 
        for traj, _, _ in trajectories
    ]).max()
    
    mid_x = trajectories[0][0][:, 0].mean()
    mid_y = trajectories[0][0][:, 1].mean()
    mid_z = trajectories[0][0][:, 2].mean()
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_error_over_time(
    errors: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    error_type: str = "Translation Error",
    save_path: Optional[str] = None
) -> None:
    """
    Plot error evolution over time.
    
    Args:
        errors: Error values [N]
        timestamps: Optional timestamps [N]
        error_type: Type of error for labeling
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    if timestamps is None:
        timestamps = np.arange(len(errors))
    
    plt.plot(timestamps, errors, 'b-', linewidth=2)
    plt.fill_between(timestamps, errors, alpha=0.3)
    
    plt.xlabel('Time (s)' if timestamps is not None else 'Frame')
    plt.ylabel(f'{error_type}')
    plt.title(f'{error_type} Over Time')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.axhline(y=mean_error, color='r', linestyle='--', 
                label=f'Mean: {mean_error:.3f}')
    plt.axhline(y=mean_error + std_error, color='r', linestyle=':', 
                alpha=0.5, label=f'±1 STD: {std_error:.3f}')
    plt.axhline(y=mean_error - std_error, color='r', linestyle=':', 
                alpha=0.5)
    
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_metrics_summary(
    metrics: dict,
    save_path: Optional[str] = None
) -> None:
    """
    Create a summary plot of all metrics.
    
    Args:
        metrics: Dictionary of metric values
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Visual-Inertial Odometry Metrics Summary', fontsize=16)
    
    # Translation errors
    ax = axes[0, 0]
    trans_metrics = {k: v for k, v in metrics.items() if 'trans' in k and 'rpe' in k}
    if trans_metrics:
        bars = ax.bar(range(len(trans_metrics)), list(trans_metrics.values()))
        ax.set_xticks(range(len(trans_metrics)))
        ax.set_xticklabels([k.replace('rpe_trans_', 'Δ') for k in trans_metrics.keys()])
        ax.set_ylabel('Translation Error (m)')
        ax.set_title('Translation RPE at Different Scales')
        for bar, val in zip(bars, trans_metrics.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom')
    
    # Rotation errors
    ax = axes[0, 1]
    rot_metrics = {k: v for k, v in metrics.items() if 'rot' in k and 'rpe' in k}
    if rot_metrics:
        bars = ax.bar(range(len(rot_metrics)), list(rot_metrics.values()), color='orange')
        ax.set_xticks(range(len(rot_metrics)))
        ax.set_xticklabels([k.replace('rpe_rot_', 'Δ') for k in rot_metrics.keys()])
        ax.set_ylabel('Rotation Error (degrees)')
        ax.set_title('Rotation RPE at Different Scales')
        for bar, val in zip(bars, rot_metrics.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{val:.2f}°', ha='center', va='bottom')
    
    # Overall metrics
    ax = axes[1, 0]
    overall_metrics = {
        'ATE': metrics.get('ate', 0),
        'Scale Error': metrics.get('scale_error', 1.0),
        'Drift Rate (%)': metrics.get('drift_rate', 0)
    }
    bars = ax.bar(range(len(overall_metrics)), list(overall_metrics.values()), 
                   color=['green', 'blue', 'red'])
    ax.set_xticks(range(len(overall_metrics)))
    ax.set_xticklabels(list(overall_metrics.keys()))
    ax.set_ylabel('Value')
    ax.set_title('Overall Trajectory Metrics')
    for bar, (key, val) in zip(bars, overall_metrics.items()):
        format_str = f'{val:.3f}' if 'Scale' not in key else f'{val:.3f}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               format_str, ha='center', va='bottom')
    
    # Info text
    ax = axes[1, 1]
    ax.axis('off')
    info_text = f"""
    Trajectory Length: {metrics.get('trajectory_length', 0):.1f} m
    
    ATE: Absolute Trajectory Error
    RPE: Relative Pose Error
    Δ1: Between consecutive frames
    Δ10: Between frames 10 apart
    Δ100: Between frames 100 apart
    
    Scale Error: 1.0 = perfect scale
    Drift Rate: Final error / trajectory length
    """
    ax.text(0.1, 0.5, info_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_video_visualization(
    pred_trajectory: np.ndarray,
    gt_trajectory: np.ndarray,
    output_path: str,
    fps: int = 30
) -> None:
    """
    Create a video visualization of trajectory evolution.
    Requires opencv-python.
    
    Args:
        pred_trajectory: Predicted trajectory [N, 3]
        gt_trajectory: Ground truth trajectory [N, 3]
        output_path: Path to save video
        fps: Frames per second
    """
    try:
        import cv2
    except ImportError:
        print("opencv-python not installed. Skipping video creation.")
        return
    
    # Create frames
    frames = []
    for i in range(len(pred_trajectory)):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot trajectories up to current point
        ax.plot(gt_trajectory[:i+1, 0], gt_trajectory[:i+1, 1], 
                'g-', linewidth=2, label='Ground Truth')
        ax.plot(pred_trajectory[:i+1, 0], pred_trajectory[:i+1, 1], 
                'r--', linewidth=2, label='Predicted')
        
        # Current position
        ax.scatter(gt_trajectory[i, 0], gt_trajectory[i, 1], 
                  c='green', s=100, marker='o')
        ax.scatter(pred_trajectory[i, 0], pred_trajectory[i, 1], 
                  c='red', s=100, marker='o')
        
        # Set fixed limits
        all_points = np.vstack([pred_trajectory, gt_trajectory])
        margin = 5
        ax.set_xlim(all_points[:, 0].min() - margin, 
                    all_points[:, 0].max() + margin)
        ax.set_ylim(all_points[:, 1].min() - margin, 
                    all_points[:, 1].max() + margin)
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(f'Trajectory Evolution - Frame {i+1}/{len(pred_trajectory)}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        plt.close(fig)
    
    # Write video
    if frames:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        print(f"Video saved to {output_path}")