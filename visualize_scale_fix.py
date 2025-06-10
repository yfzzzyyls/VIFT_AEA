#!/usr/bin/env python3
"""Visualize trajectory results after scale fix."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load results
data = np.load('inference_results_realtime_seq_009_stride_20.npz', allow_pickle=True)

# Get trajectories
gt_aligned = data['gt_aligned']
pred_aligned = data['pred_aligned']
metrics = data['metrics'].item()

# Create figure with subplots
fig = plt.figure(figsize=(15, 10))

# 3D trajectory plot
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot(gt_aligned[:, 0], gt_aligned[:, 1], gt_aligned[:, 2], 'b-', label='Ground Truth', linewidth=2)
ax1.plot(pred_aligned[:, 0], pred_aligned[:, 1], pred_aligned[:, 2], 'r--', label='Predicted', linewidth=2)
ax1.set_xlabel('X (cm)')
ax1.set_ylabel('Y (cm)')
ax1.set_zlabel('Z (cm)')
ax1.set_title('3D Trajectory Comparison (Aligned)')
ax1.legend()

# XY plane view
ax2 = fig.add_subplot(222)
ax2.plot(gt_aligned[:, 0], gt_aligned[:, 1], 'b-', label='Ground Truth', linewidth=2)
ax2.plot(pred_aligned[:, 0], pred_aligned[:, 1], 'r--', label='Predicted', linewidth=2)
ax2.set_xlabel('X (cm)')
ax2.set_ylabel('Y (cm)')
ax2.set_title('XY Plane View')
ax2.axis('equal')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Translation error over time
ax3 = fig.add_subplot(223)
trans_errors = np.linalg.norm(gt_aligned[:, :3] - pred_aligned[:, :3], axis=1)
time_seconds = np.arange(len(trans_errors)) / 20.0  # 20 fps
ax3.plot(time_seconds, trans_errors, 'g-', linewidth=1)
ax3.axhline(y=metrics['ate_mean_cm'], color='r', linestyle='--', label=f'Mean: {metrics["ate_mean_cm"]:.2f} cm')
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Translation Error (cm)')
ax3.set_title('Translation Error Over Time')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Metrics summary
ax4 = fig.add_subplot(224)
ax4.axis('off')
metrics_text = f"""Scale Fix Results Summary:

Absolute Trajectory Error (ATE):
  Mean: {metrics['ate_mean_cm']:.2f} cm
  RMSE: {metrics['ate_rmse_cm']:.2f} cm
  Median: {metrics['ate_median_cm']:.2f} cm
  95%: {metrics['ate_95_cm']:.2f} cm

Raw ATE (before alignment):
  Mean: {metrics['ate_mean_raw_cm']:.2f} cm

Alignment Info:
  Scale: {metrics['alignment_info']['scale']:.4f}
  (Predictions are {1/metrics['alignment_info']['scale']:.1f}x GT scale)
  
Rotation Error:
  Mean: {metrics['rot_mean_deg']:.1f}°
  RMSE: {metrics['rot_rmse_deg']:.1f}°

Note: Person is moving very slowly in this sequence
(~0.05-0.2 cm/s)"""

ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, 
         fontsize=11, verticalalignment='top', fontfamily='monospace')

plt.suptitle('Sequence 009 - Scale Fix Visualization', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('scale_fix_visualization_seq_009.png', dpi=150, bbox_inches='tight')
print("Saved visualization to scale_fix_visualization_seq_009.png")

# Also create a simple 2D comparison
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Before fix (approximate from previous results)
ax1.text(0.5, 0.5, """Before Scale Fix:

Raw ATE: 89.44 cm
Aligned ATE: 0.08 cm
Scale ratio: 0.0045

Predictions were 222x larger
than ground truth!""", 
         transform=ax1.transAxes, ha='center', va='center',
         fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
ax1.set_title('Before Fix', fontsize=16, fontweight='bold')
ax1.axis('off')

# After fix
ax2.text(0.5, 0.5, """After Scale Fix:

Raw ATE: 20.95 cm
Aligned ATE: 8.36 cm
Scale ratio: 0.45

Predictions are now only 2.2x
larger than ground truth""", 
         transform=ax2.transAxes, ha='center', va='center',
         fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
ax2.set_title('After Fix', fontsize=16, fontweight='bold')
ax2.axis('off')

plt.suptitle('Scale Fix Impact Summary', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('scale_fix_before_after.png', dpi=150, bbox_inches='tight')
print("Saved before/after comparison to scale_fix_before_after.png")