#!/usr/bin/env python3
"""
Evaluation script for FlowNet-LSTM-Transformer architecture on Aria dataset.
Computes comprehensive metrics and generates visualizations.

Usage:
    python evaluate_flownet_lstm_transformer.py \
        --checkpoint new_architecture/checkpoints/exp_name/best_model.pt \
        --output-dir new_architecture/evaluation/exp_name
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from models.flownet_lstm_transformer import FlowNetLSTMTransformer
from data.aria_variable_imu_dataset import AriaVariableIMUDataset, collate_variable_imu
from configs.flownet_lstm_transformer_config import Config, ModelConfig
from utils.metrics import compute_trajectory_metrics, compute_ate, compute_rpe
from utils.visualization import plot_trajectory_comparison
from utils.pose_utils import integrate_poses, poses_to_trajectory


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Config]:
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default configuration if not saved
        print("Warning: No config found in checkpoint, using defaults")
        from configs.flownet_lstm_transformer_config import get_config
        config = get_config()
    
    # Create model
    if isinstance(config, dict):
        # Convert dict to Config object
        model_config = ModelConfig(**config.get('model', {}))
        model = FlowNetLSTMTransformer(model_config)
    else:
        model = FlowNetLSTMTransformer(config.model)
    
    # Load state dict
    state_dict = checkpoint['model_state_dict']
    # Handle DDP wrapped models
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    return model, config


def evaluate_sequence(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    sequence_name: str
) -> Dict[str, any]:
    """Evaluate model on a single sequence."""
    all_predictions = []
    all_ground_truth = []
    all_translations_pred = []
    all_translations_gt = []
    all_rotations_pred = []
    all_rotations_gt = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {sequence_name}")):
            # Move data to device
            images = batch['images'].to(device)
            poses_gt = batch['poses'].to(device)
            
            # Move IMU sequences to device
            imu_sequences = batch['imu_sequences']
            for b in range(len(imu_sequences)):
                for t in range(len(imu_sequences[b])):
                    imu_sequences[b][t] = imu_sequences[b][t].to(device)
            
            # Forward pass
            outputs = model(images, imu_sequences)
            
            # Store predictions
            all_predictions.append(outputs['poses'].cpu())
            all_ground_truth.append(poses_gt.cpu())
            all_translations_pred.append(outputs['translation'].cpu())
            all_translations_gt.append(poses_gt[:, :, :3].cpu())
            all_rotations_pred.append(outputs['rotation'].cpu())
            all_rotations_gt.append(poses_gt[:, :, 3:].cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_ground_truth = torch.cat(all_ground_truth, dim=0).numpy()
    all_translations_pred = torch.cat(all_translations_pred, dim=0).numpy()
    all_translations_gt = torch.cat(all_translations_gt, dim=0).numpy()
    all_rotations_pred = torch.cat(all_rotations_pred, dim=0).numpy()
    all_rotations_gt = torch.cat(all_rotations_gt, dim=0).numpy()
    
    # Compute trajectories from relative poses
    trajectories_pred = []
    trajectories_gt = []
    
    for i in range(len(all_predictions)):
        # Integrate relative poses to get absolute trajectory
        traj_pred = integrate_poses(all_predictions[i])
        traj_gt = integrate_poses(all_ground_truth[i])
        trajectories_pred.append(traj_pred)
        trajectories_gt.append(traj_gt)
    
    # Compute metrics
    metrics = {}
    
    # Average metrics over all sequences
    ate_values = []
    rpe_trans_values = []
    rpe_rot_values = []
    
    for traj_pred, traj_gt in zip(trajectories_pred, trajectories_gt):
        # ATE (Absolute Trajectory Error)
        ate = compute_ate(traj_pred[:, :3], traj_gt[:, :3])
        ate_values.append(ate)
        
        # RPE (Relative Pose Error)
        rpe_trans, rpe_rot = compute_rpe(traj_pred, traj_gt, delta=1)
        rpe_trans_values.append(rpe_trans)
        rpe_rot_values.append(rpe_rot)
    
    # Aggregate metrics
    metrics['ate_mean'] = np.mean(ate_values)
    metrics['ate_std'] = np.std(ate_values)
    metrics['ate_median'] = np.median(ate_values)
    metrics['rpe_trans_mean'] = np.mean(rpe_trans_values)
    metrics['rpe_trans_std'] = np.std(rpe_trans_values)
    metrics['rpe_rot_mean'] = np.mean(rpe_rot_values)
    metrics['rpe_rot_std'] = np.std(rpe_rot_values)
    
    # Scale consistency
    pred_lengths = [np.sum(np.linalg.norm(t[1:, :3] - t[:-1, :3], axis=1)) 
                   for t in trajectories_pred]
    gt_lengths = [np.sum(np.linalg.norm(t[1:, :3] - t[:-1, :3], axis=1)) 
                 for t in trajectories_gt]
    scale_errors = np.abs(np.array(pred_lengths) - np.array(gt_lengths)) / np.array(gt_lengths)
    metrics['scale_error_mean'] = np.mean(scale_errors)
    metrics['scale_error_std'] = np.std(scale_errors)
    
    # Drift analysis
    drift_percentages = []
    for traj_pred, traj_gt in zip(trajectories_pred, trajectories_gt):
        end_error = np.linalg.norm(traj_pred[-1, :3] - traj_gt[-1, :3])
        path_length = np.sum(np.linalg.norm(traj_gt[1:, :3] - traj_gt[:-1, :3], axis=1))
        drift_percentage = (end_error / path_length) * 100
        drift_percentages.append(drift_percentage)
    
    metrics['drift_mean'] = np.mean(drift_percentages)
    metrics['drift_std'] = np.std(drift_percentages)
    
    # Store raw data for visualization
    results = {
        'metrics': metrics,
        'trajectories_pred': trajectories_pred,
        'trajectories_gt': trajectories_gt,
        'translations_pred': all_translations_pred,
        'translations_gt': all_translations_gt,
        'rotations_pred': all_rotations_pred,
        'rotations_gt': all_rotations_gt,
        'predictions': all_predictions,  # Raw relative poses for plotting
        'ground_truth': all_ground_truth  # Raw relative poses for plotting
    }
    
    return results


def create_visualizations(results: Dict, output_dir: Path, sequence_name: str):
    """Create and save visualization plots."""
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # 1. Trajectory comparison plot
    if len(results['trajectories_pred']) > 0:
        # Plot first few trajectories
        for i in range(min(5, len(results['trajectories_pred']))):
            fig = plot_trajectory_comparison(
                results['trajectories_pred'][i],
                results['trajectories_gt'][i],
                title=f"{sequence_name} - Trajectory {i}"
            )
            if fig is not None:
                fig.savefig(vis_dir / f'{sequence_name}_trajectory_{i}.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
            else:
                print(f"Warning: Failed to create trajectory plot {i}")
    
    # 2. Error distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ATE distribution
    ate_values = []
    for traj_pred, traj_gt in zip(results['trajectories_pred'], results['trajectories_gt']):
        ate = compute_ate(traj_pred[:, :3], traj_gt[:, :3])
        ate_values.append(ate)
    
    axes[0, 0].hist(ate_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('ATE (m)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Absolute Trajectory Error Distribution')
    axes[0, 0].axvline(np.mean(ate_values), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(ate_values):.3f}m')
    axes[0, 0].legend()
    
    # Translation error per frame
    trans_errors = np.linalg.norm(
        results['translations_pred'] - results['translations_gt'], 
        axis=-1
    ).flatten()
    axes[0, 1].hist(trans_errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Translation Error (m)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Per-Frame Translation Error')
    axes[0, 1].axvline(np.mean(trans_errors), color='red', linestyle='--',
                      label=f'Mean: {np.mean(trans_errors):.3f}m')
    axes[0, 1].legend()
    
    # Rotation error per frame (in degrees)
    # Compute quaternion distance
    rot_pred = results['rotations_pred'].reshape(-1, 4)
    rot_gt = results['rotations_gt'].reshape(-1, 4)
    # Normalize quaternions
    rot_pred = rot_pred / np.linalg.norm(rot_pred, axis=1, keepdims=True)
    rot_gt = rot_gt / np.linalg.norm(rot_gt, axis=1, keepdims=True)
    # Compute angle difference
    dot_products = np.clip(np.sum(rot_pred * rot_gt, axis=1), -1, 1)
    rot_errors_rad = 2 * np.arccos(np.abs(dot_products))
    rot_errors_deg = np.degrees(rot_errors_rad)
    
    axes[1, 0].hist(rot_errors_deg, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Rotation Error (degrees)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Per-Frame Rotation Error')
    axes[1, 0].axvline(np.mean(rot_errors_deg), color='red', linestyle='--',
                      label=f'Mean: {np.mean(rot_errors_deg):.2f}°')
    axes[1, 0].legend()
    
    # Scale error distribution
    pred_lengths = [np.sum(np.linalg.norm(t[1:, :3] - t[:-1, :3], axis=1)) 
                   for t in results['trajectories_pred']]
    gt_lengths = [np.sum(np.linalg.norm(t[1:, :3] - t[:-1, :3], axis=1)) 
                 for t in results['trajectories_gt']]
    scale_ratios = np.array(pred_lengths) / np.array(gt_lengths)
    
    axes[1, 1].hist(scale_ratios, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_xlabel('Scale Ratio (Predicted/GT)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Trajectory Scale Distribution')
    axes[1, 1].axvline(1.0, color='green', linestyle='--', label='Perfect Scale')
    axes[1, 1].axvline(np.mean(scale_ratios), color='red', linestyle='--',
                      label=f'Mean: {np.mean(scale_ratios):.3f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    fig.savefig(vis_dir / f'{sequence_name}_error_distributions.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    """Main evaluation function."""
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate FlowNet-LSTM-Transformer")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='aria_processed',
                       help='Path to Aria dataset')
    parser.add_argument('--output-dir', type=str, default='new_architecture/evaluation',
                       help='Output directory for results')
    parser.add_argument('--split', type=str, default='test',
                       choices=['val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--sequence-length', type=int, default=11,
                       help='Sequence length for evaluation')
    parser.add_argument('--stride', type=int, default=10,
                       help='Stride for sliding window')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("FlowNet-LSTM-Transformer Evaluation")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {device}")
    print("="*80 + "\n")
    
    # Load model
    model, config = load_checkpoint(args.checkpoint, device)
    
    # Create dataset
    dataset = AriaVariableIMUDataset(
        data_dir=args.data_dir,
        split=args.split,
        variable_length=False,  # Fixed length for evaluation
        sequence_length=args.sequence_length,
        stride=args.stride,
        image_size=(704, 704)  # Match training image size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_variable_imu
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Evaluate
    results = evaluate_sequence(model, dataloader, device, args.split)
    
    # Print metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    metrics = results['metrics']
    print("\nTrajectory Metrics:")
    print(f"  ATE (Absolute Trajectory Error):")
    print(f"    Mean: {metrics['ate_mean']:.4f} m")
    print(f"    Std:  {metrics['ate_std']:.4f} m")
    print(f"    Median: {metrics['ate_median']:.4f} m")
    
    print(f"\n  RPE (Relative Pose Error):")
    print(f"    Translation - Mean: {metrics['rpe_trans_mean']:.4f} m, Std: {metrics['rpe_trans_std']:.4f} m")
    print(f"    Rotation - Mean: {metrics['rpe_rot_mean']:.4f} rad ({np.degrees(metrics['rpe_rot_mean']):.2f}°)")
    
    print(f"\n  Scale Consistency:")
    print(f"    Mean Error: {metrics['scale_error_mean']*100:.2f}%")
    print(f"    Std: {metrics['scale_error_std']*100:.2f}%")
    
    print(f"\n  Drift:")
    print(f"    Mean: {metrics['drift_mean']:.2f}% of path length")
    print(f"    Std: {metrics['drift_std']:.2f}%")
    
    # Save metrics
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(results, output_dir, args.split)
    print(f"Visualizations saved to: {output_dir / 'visualizations'}")
    
    # Save detailed results
    results_file = output_dir / 'detailed_results.npz'
    np.savez_compressed(
        results_file,
        trajectories_pred=results['trajectories_pred'],
        trajectories_gt=results['trajectories_gt'],
        translations_pred=results['translations_pred'],
        translations_gt=results['translations_gt'],
        rotations_pred=results['rotations_pred'],
        rotations_gt=results['rotations_gt']
    )
    print(f"Detailed results saved to: {results_file}")
    
    # Also save as .pt for the plotting script
    torch.save({
        'predictions': results['predictions'],
        'ground_truth': results['ground_truth']
    }, output_dir / 'results.pt')
    print(f"Results saved to: {output_dir / 'results.pt'}")
    
    print("\n" + "="*80)
    print("Evaluation completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()