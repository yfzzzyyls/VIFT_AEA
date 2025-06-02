#!/usr/bin/env python3
"""
Test different aggregation strategies for overlapping windows.
Compare using first prediction vs middle frames vs averaging.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def simulate_predictions_with_noise(n_frames=50, window_size=11, stride=1, noise_std=0.01):
    """
    Simulate pose predictions with different noise levels at window edges.
    Middle frames should have lower error.
    """
    # True trajectory: simple forward motion
    true_poses = np.zeros((n_frames, 7))
    true_poses[:, 3:] = [0, 0, 0, 1]  # Identity rotation
    for i in range(n_frames):
        true_poses[i, 0] = i * 0.1  # 0.1 units per frame in X
    
    # Generate predictions for each window
    windows = []
    for start_idx in range(0, n_frames - window_size + 1, stride):
        window_pred = np.zeros((window_size, 7))
        window_pred[:, 3:] = [0, 0, 0, 1]  # Identity rotation
        
        for i in range(window_size):
            frame_idx = start_idx + i
            
            # Add noise that's higher at edges, lower in middle
            distance_from_center = abs(i - window_size // 2)
            noise_scale = 1.0 + distance_from_center * 0.2  # 20% more noise per frame from center
            
            # True position + noise
            window_pred[i, 0] = true_poses[frame_idx, 0] + np.random.normal(0, noise_std * noise_scale)
            window_pred[i, 1] = 0 + np.random.normal(0, noise_std * noise_scale * 0.1)  # Small Y noise
            window_pred[i, 2] = 0 + np.random.normal(0, noise_std * noise_scale * 0.1)  # Small Z noise
        
        windows.append((start_idx, window_pred))
    
    return true_poses, windows


def aggregate_first_only(windows, n_frames):
    """Use only the first prediction for each frame."""
    aggregated = np.zeros((n_frames, 7))
    aggregated[:, 3:] = [0, 0, 0, 1]
    counts = np.zeros(n_frames)
    
    for start_idx, pred in windows:
        for i in range(len(pred)):
            frame_idx = start_idx + i
            if frame_idx < n_frames and counts[frame_idx] == 0:
                aggregated[frame_idx] = pred[i]
                counts[frame_idx] = 1
    
    return aggregated


def aggregate_middle_priority(windows, n_frames, window_size=11):
    """Prioritize predictions from middle of windows."""
    aggregated = np.zeros((n_frames, 7))
    aggregated[:, 3:] = [0, 0, 0, 1]
    quality_scores = np.full(n_frames, -1.0)  # Track best quality so far
    
    middle = window_size // 2
    
    for start_idx, pred in windows:
        for i in range(len(pred)):
            frame_idx = start_idx + i
            if frame_idx < n_frames:
                # Quality score: higher for frames closer to window center
                quality = 1.0 - abs(i - middle) / middle
                
                if quality > quality_scores[frame_idx]:
                    aggregated[frame_idx] = pred[i]
                    quality_scores[frame_idx] = quality
    
    return aggregated


def aggregate_weighted_average(windows, n_frames, window_size=11):
    """Weighted average based on position in window."""
    aggregated = np.zeros((n_frames, 7))
    aggregated[:, 3:] = [0, 0, 0, 1]
    weights_sum = np.zeros(n_frames)
    
    middle = window_size // 2
    
    for start_idx, pred in windows:
        for i in range(len(pred)):
            frame_idx = start_idx + i
            if frame_idx < n_frames:
                # Weight: higher for frames closer to window center
                weight = 1.0 - abs(i - middle) / middle
                
                aggregated[frame_idx, :3] += weight * pred[i, :3]
                weights_sum[frame_idx] += weight
    
    # Normalize by total weights
    for i in range(n_frames):
        if weights_sum[i] > 0:
            aggregated[i, :3] /= weights_sum[i]
    
    return aggregated


def evaluate_strategies():
    """Compare different aggregation strategies."""
    np.random.seed(42)
    
    n_frames = 100
    window_size = 11
    stride = 1
    noise_std = 0.02
    
    # Generate simulated data
    true_poses, windows = simulate_predictions_with_noise(n_frames, window_size, stride, noise_std)
    
    # Test different strategies
    agg_first = aggregate_first_only(windows, n_frames)
    agg_middle = aggregate_middle_priority(windows, n_frames, window_size)
    agg_weighted = aggregate_weighted_average(windows, n_frames, window_size)
    
    # Calculate errors
    errors_first = np.linalg.norm(agg_first[:, :3] - true_poses[:, :3], axis=1)
    errors_middle = np.linalg.norm(agg_middle[:, :3] - true_poses[:, :3], axis=1)
    errors_weighted = np.linalg.norm(agg_weighted[:, :3] - true_poses[:, :3], axis=1)
    
    # Print statistics
    print("=== Aggregation Strategy Comparison ===\n")
    print(f"First-only strategy:")
    print(f"  Mean error: {errors_first.mean():.6f}")
    print(f"  Std error:  {errors_first.std():.6f}")
    print(f"  Max error:  {errors_first.max():.6f}")
    
    print(f"\nMiddle-priority strategy:")
    print(f"  Mean error: {errors_middle.mean():.6f}")
    print(f"  Std error:  {errors_middle.std():.6f}")
    print(f"  Max error:  {errors_middle.max():.6f}")
    
    print(f"\nWeighted average strategy:")
    print(f"  Mean error: {errors_weighted.mean():.6f}")
    print(f"  Std error:  {errors_weighted.std():.6f}")
    print(f"  Max error:  {errors_weighted.max():.6f}")
    
    # Improvement percentages
    improve_middle = (errors_first.mean() - errors_middle.mean()) / errors_first.mean() * 100
    improve_weighted = (errors_first.mean() - errors_weighted.mean()) / errors_first.mean() * 100
    
    print(f"\nImprovement over first-only:")
    print(f"  Middle-priority: {improve_middle:.1f}%")
    print(f"  Weighted average: {improve_weighted:.1f}%")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(errors_first, label='First-only', alpha=0.7)
    plt.plot(errors_middle, label='Middle-priority', alpha=0.7)
    plt.plot(errors_weighted, label='Weighted average', alpha=0.7)
    plt.xlabel('Frame')
    plt.ylabel('Position Error')
    plt.title('Aggregation Strategy Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    x = np.arange(3)
    means = [errors_first.mean(), errors_middle.mean(), errors_weighted.mean()]
    stds = [errors_first.std(), errors_middle.std(), errors_weighted.std()]
    
    plt.bar(x, means, yerr=stds, capsize=10)
    plt.xticks(x, ['First-only', 'Middle-priority', 'Weighted'])
    plt.ylabel('Mean Error ± Std')
    plt.title('Average Performance')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('aggregation_comparison.png', dpi=150)
    print(f"\nPlot saved to aggregation_comparison.png")


if __name__ == "__main__":
    evaluate_strategies()