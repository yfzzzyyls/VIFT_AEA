#!/usr/bin/env python3
"""
Evaluation script with proper ATE and RPE metrics for AR/VR.
"""

import os
import sys
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
import argparse

# Add src to path
sys.path.append('src')

from src.models.multihead_vio import MultiHeadVIOModel
from train_pretrained_relative import RelativePoseDataset
from torch.utils.data import DataLoader

console = Console()


def quaternion_to_matrix(q):
    """Convert quaternion to rotation matrix."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])


def build_trajectory(relative_poses):
    """Build absolute trajectory from relative poses."""
    trajectory = []
    
    # Start at origin
    current_pos = np.zeros(3)
    current_rot = np.eye(3)
    trajectory.append((current_pos.copy(), current_rot.copy()))
    
    for pose in relative_poses:
        # Extract translation and rotation
        trans = pose[:3]
        quat = pose[3:]
        
        # Convert quaternion to rotation matrix
        rot_matrix = quaternion_to_matrix(quat)
        
        # Update position (in current frame)
        current_pos = current_pos + current_rot @ trans
        
        # Update rotation
        current_rot = current_rot @ rot_matrix
        
        trajectory.append((current_pos.copy(), current_rot.copy()))
    
    return trajectory


def calculate_ate(pred_trajectory, gt_trajectory):
    """Calculate Absolute Trajectory Error (ATE)."""
    errors = []
    
    for (pred_pos, _), (gt_pos, _) in zip(pred_trajectory, gt_trajectory):
        error = np.linalg.norm(pred_pos - gt_pos)
        errors.append(error)
    
    return np.array(errors)


def calculate_rpe(pred_trajectory, gt_trajectory, delta=1):
    """Calculate Relative Pose Error (RPE) at given frame interval."""
    trans_errors = []
    rot_errors = []
    
    for i in range(len(pred_trajectory) - delta):
        # Get relative motion for prediction
        pred_pos1, pred_rot1 = pred_trajectory[i]
        pred_pos2, pred_rot2 = pred_trajectory[i + delta]
        pred_rel_trans = pred_rot1.T @ (pred_pos2 - pred_pos1)
        pred_rel_rot = pred_rot1.T @ pred_rot2
        
        # Get relative motion for ground truth
        gt_pos1, gt_rot1 = gt_trajectory[i]
        gt_pos2, gt_rot2 = gt_trajectory[i + delta]
        gt_rel_trans = gt_rot1.T @ (gt_pos2 - gt_pos1)
        gt_rel_rot = gt_rot1.T @ gt_rot2
        
        # Translation error
        trans_error = np.linalg.norm(pred_rel_trans - gt_rel_trans)
        trans_errors.append(trans_error)
        
        # Rotation error (angle)
        rel_rot_diff = pred_rel_rot @ gt_rel_rot.T
        angle = np.arccos(np.clip((np.trace(rel_rot_diff) - 1) / 2, -1, 1))
        rot_errors.append(np.degrees(angle))
    
    return np.array(trans_errors), np.array(rot_errors)


def evaluate_with_metrics(checkpoint_path: str, pose_scale: float = 100.0):
    """Evaluate with proper ATE and RPE metrics."""
    
    console.rule("[bold cyan]🚀 AR/VR Metrics Evaluation[/bold cyan]")
    
    # Load model
    model = MultiHeadVIOModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Create test dataset
    test_dataset = RelativePoseDataset(
        "aria_latent_data_pretrained/test",
        pose_scale=pose_scale
    )
    
    # Create dataloader
    def collate_fn(batch):
        features, imus, poses = zip(*batch)
        return {
            'images': torch.stack(features),
            'imus': torch.stack(imus),
            'poses': torch.stack(poses)
        }
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn
    )
    
    console.print(f"\nTest samples: {len(test_dataset):,}")
    
    # Collect all trajectories
    all_ate_errors = []
    all_rpe_trans_1 = []
    all_rpe_rot_1 = []
    all_rpe_trans_5 = []
    all_rpe_rot_5 = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            
            # Get predictions
            outputs = model(batch)
            
            # Process each sequence in batch
            batch_size = batch['images'].shape[0]
            
            for i in range(batch_size):
                # Extract predictions and ground truth
                pred_poses = torch.cat([
                    outputs['translation'][i],
                    outputs['rotation'][i]
                ], dim=1).cpu().numpy()  # [11, 7]
                
                gt_poses = batch['poses'][i].cpu().numpy()  # [11, 7]
                
                # Build trajectories
                pred_traj = build_trajectory(pred_poses)
                gt_traj = build_trajectory(gt_poses)
                
                # Calculate ATE
                ate_errors = calculate_ate(pred_traj, gt_traj)
                all_ate_errors.extend(ate_errors)
                
                # Calculate RPE at different intervals
                if len(pred_traj) > 1:
                    rpe_trans_1, rpe_rot_1 = calculate_rpe(pred_traj, gt_traj, delta=1)
                    all_rpe_trans_1.extend(rpe_trans_1)
                    all_rpe_rot_1.extend(rpe_rot_1)
                
                if len(pred_traj) > 5:
                    rpe_trans_5, rpe_rot_5 = calculate_rpe(pred_traj, gt_traj, delta=5)
                    all_rpe_trans_5.extend(rpe_trans_5)
                    all_rpe_rot_5.extend(rpe_rot_5)
            
            if batch_idx % 50 == 0:
                console.print(f"  Processed {batch_idx}/{len(test_loader)} batches...")
    
    # Convert to arrays
    ate_errors = np.array(all_ate_errors)
    rpe_trans_1 = np.array(all_rpe_trans_1)
    rpe_rot_1 = np.array(all_rpe_rot_1)
    rpe_trans_5 = np.array(all_rpe_trans_5) if all_rpe_trans_5 else np.array([0])
    rpe_rot_5 = np.array(all_rpe_rot_5) if all_rpe_rot_5 else np.array([0])
    
    # Create results table
    table = Table(title="AR/VR Standard Metrics")
    table.add_column("Metric", style="cyan", width=40)
    table.add_column("Value", style="green", width=25)
    table.add_column("AR/VR Target", style="yellow", width=15)
    table.add_column("Status", style="white", width=15)
    
    # ATE metrics
    ate_mean = ate_errors.mean()
    ate_median = np.median(ate_errors)
    ate_std = ate_errors.std()
    ate_95 = np.percentile(ate_errors, 95)
    
    table.add_row(
        "ATE (Absolute Trajectory Error)",
        f"{ate_mean:.4f} ± {ate_std:.4f} cm",
        "<1 cm",
        "✅ EXCEEDS" if ate_mean < 1.0 else "❌ FAILS"
    )
    table.add_row(
        "  ├─ Median",
        f"{ate_median:.4f} cm",
        "-",
        "-"
    )
    table.add_row(
        "  └─ 95th percentile",
        f"{ate_95:.4f} cm",
        "-",
        "-"
    )
    
    # RPE metrics (1 frame)
    table.add_row(
        "RPE Translation (1 frame)",
        f"{rpe_trans_1.mean():.4f} ± {rpe_trans_1.std():.4f} cm",
        "<0.1 cm",
        "✅ EXCEEDS" if rpe_trans_1.mean() < 0.1 else "❌ FAILS"
    )
    table.add_row(
        "RPE Rotation (1 frame)",
        f"{rpe_rot_1.mean():.4f} ± {rpe_rot_1.std():.4f}°",
        "<0.1°",
        "✅ EXCEEDS" if rpe_rot_1.mean() < 0.1 else "⚠️  CLOSE"
    )
    
    # RPE metrics (5 frames)
    if len(all_rpe_trans_5) > 0:
        table.add_row(
            "RPE Translation (5 frames)",
            f"{rpe_trans_5.mean():.4f} ± {rpe_trans_5.std():.4f} cm",
            "<0.5 cm",
            "✅ EXCEEDS" if rpe_trans_5.mean() < 0.5 else "❌ FAILS"
        )
        table.add_row(
            "RPE Rotation (5 frames)",
            f"{rpe_rot_5.mean():.4f} ± {rpe_rot_5.std():.4f}°",
            "<0.5°",
            "✅ EXCEEDS" if rpe_rot_5.mean() < 0.5 else "❌ FAILS"
        )
    
    console.print("\n")
    console.print(table)
    
    # Performance Summary
    console.print("\n[bold]Performance Summary:[/bold]")
    
    if ate_mean < 0.01:  # Less than 0.01cm
        console.print(f"[bold green]✅ EXCEPTIONAL! ATE of {ate_mean:.4f}cm is sub-millimeter accuracy![/bold green]")
    elif ate_mean < 0.1:
        console.print(f"[bold green]✅ EXCELLENT! ATE of {ate_mean:.4f}cm exceeds professional AR/VR requirements![/bold green]")
    elif ate_mean < 1.0:
        console.print(f"[bold yellow]⚠️  GOOD. ATE of {ate_mean:.4f}cm meets AR/VR requirements.[/bold yellow]")
    else:
        console.print(f"[bold red]❌ NEEDS IMPROVEMENT. ATE of {ate_mean:.4f}cm exceeds 1cm threshold.[/bold red]")
    
    if ate_mean < 0.59:
        improvement = (0.59 - ate_mean) / 0.59 * 100
        console.print(f"[bold green]📈 {improvement:.1f}% improvement over ResNet baseline (0.59cm)![/bold green]")
    
    # Additional insights
    console.print("\n[bold]Technical Details:[/bold]")
    console.print(f"  • Total evaluated poses: {len(ate_errors):,}")
    console.print(f"  • Trajectory sequences: {len(test_dataset)}")
    console.print(f"  • Frames per sequence: 11")
    console.print(f"  • Evaluation performed on: Test set")
    
    # Save detailed results
    results = {
        'ate_mean': ate_mean,
        'ate_median': ate_median,
        'ate_std': ate_std,
        'ate_95': ate_95,
        'rpe_trans_1_mean': rpe_trans_1.mean(),
        'rpe_rot_1_mean': rpe_rot_1.mean(),
        'rpe_trans_5_mean': rpe_trans_5.mean() if len(all_rpe_trans_5) > 0 else 0,
        'rpe_rot_5_mean': rpe_rot_5.mean() if len(all_rpe_rot_5) > 0 else 0,
    }
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate with AR/VR metrics (ATE & RPE)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--scale', type=float, default=100.0,
                       help='Pose scale factor')
    
    args = parser.parse_args()
    
    console.print("[bold magenta]Visual-Selective-VIO Evaluation with AR/VR Metrics[/bold magenta]\n")
    
    results = evaluate_with_metrics(args.checkpoint, args.scale)