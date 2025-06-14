from .base_metrics_calculator import BaseMetricsCalculator
from typing import Dict, Any
from src.utils.kitti_eval import kitti_err_cal
from src.utils.kitti_utils import path_accu
import numpy as np

class ARVRKITTIMetricsCalculator(BaseMetricsCalculator):
    """Metrics calculator that provides both KITTI and AR/VR metrics"""
    
    def calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        metrics = {}
        
        # Collect all frame-to-frame errors for AR/VR metrics
        all_trans_errors_cm = []
        all_rot_errors_deg = []
        
        for seq_name, seq_data in results.items():
            pose_est = seq_data['estimated_poses']
            pose_gt = seq_data['gt_poses']
            
            # Convert to global poses for KITTI metrics
            pose_est_global = path_accu(pose_est)
            pose_gt_global = path_accu(pose_gt)
            
            # Calculate KITTI errors (percentage-based)
            err, t_rel, r_rel, speed = kitti_err_cal(pose_est_global, pose_gt_global)
            t_rmse, r_rmse = self.calculate_rmse(pose_est, pose_gt)
            
            # Store KITTI metrics
            metrics[f'{seq_name}_t_rel'] = t_rel * 100  # Percentage error
            metrics[f'{seq_name}_r_rel'] = r_rel / np.pi * 180 * 100
            metrics[f'{seq_name}_t_rmse'] = t_rmse
            metrics[f'{seq_name}_r_rmse'] = r_rmse / np.pi * 180
            
            # Calculate AR/VR metrics (absolute frame-to-frame errors)
            # Translation error in meters
            trans_errors = np.linalg.norm(pose_est[:, 3:] - pose_gt[:, 3:], axis=1)
            # Convert to centimeters
            trans_errors_cm = trans_errors * 100
            all_trans_errors_cm.extend(trans_errors_cm)
            
            # Rotation error in radians
            rot_errors_rad = np.linalg.norm(pose_est[:, :3] - pose_gt[:, :3], axis=1)
            # Convert to degrees
            rot_errors_deg = rot_errors_rad * 180 / np.pi
            all_rot_errors_deg.extend(rot_errors_deg)
            
            # Per-sequence AR/VR metrics
            metrics[f'{seq_name}_trans_mean_cm'] = np.mean(trans_errors_cm)
            metrics[f'{seq_name}_trans_median_cm'] = np.median(trans_errors_cm)
            metrics[f'{seq_name}_trans_95_cm'] = np.percentile(trans_errors_cm, 95)
            metrics[f'{seq_name}_rot_mean_deg'] = np.mean(rot_errors_deg)
            metrics[f'{seq_name}_rot_median_deg'] = np.median(rot_errors_deg)
            metrics[f'{seq_name}_rot_95_deg'] = np.percentile(rot_errors_deg, 95)
        
        # Overall AR/VR metrics across all sequences
        if all_trans_errors_cm:
            metrics['overall_trans_mean_cm'] = np.mean(all_trans_errors_cm)
            metrics['overall_trans_median_cm'] = np.median(all_trans_errors_cm)
            metrics['overall_trans_std_cm'] = np.std(all_trans_errors_cm)
            metrics['overall_trans_95_cm'] = np.percentile(all_trans_errors_cm, 95)
            metrics['overall_trans_max_cm'] = np.max(all_trans_errors_cm)
            
            metrics['overall_rot_mean_deg'] = np.mean(all_rot_errors_deg)
            metrics['overall_rot_median_deg'] = np.median(all_rot_errors_deg)
            metrics['overall_rot_std_deg'] = np.std(all_rot_errors_deg)
            metrics['overall_rot_95_deg'] = np.percentile(all_rot_errors_deg, 95)
            metrics['overall_rot_max_deg'] = np.max(all_rot_errors_deg)
        
        # Print AR/VR summary
        self.print_arvr_summary(metrics)
        
        return metrics
    
    def print_arvr_summary(self, metrics):
        """Print AR/VR-specific metrics summary"""
        print("\n" + "="*60)
        print("🥽 AR/VR METRICS SUMMARY (Absolute Frame-to-Frame Errors)")
        print("="*60)
        
        if 'overall_trans_mean_cm' in metrics:
            print(f"\n📏 Translation Error (cm):")
            print(f"   Mean:   {metrics['overall_trans_mean_cm']:.3f}")
            print(f"   Median: {metrics['overall_trans_median_cm']:.3f}")
            print(f"   Std:    {metrics['overall_trans_std_cm']:.3f}")
            print(f"   95%:    {metrics['overall_trans_95_cm']:.3f}")
            print(f"   Max:    {metrics['overall_trans_max_cm']:.3f}")
            
            print(f"\n🔄 Rotation Error (degrees):")
            print(f"   Mean:   {metrics['overall_rot_mean_deg']:.3f}")
            print(f"   Median: {metrics['overall_rot_median_deg']:.3f}")
            print(f"   Std:    {metrics['overall_rot_std_deg']:.3f}")
            print(f"   95%:    {metrics['overall_rot_95_deg']:.3f}")
            print(f"   Max:    {metrics['overall_rot_max_deg']:.3f}")
        
        print("\n📊 Per-Sequence AR/VR Metrics:")
        sequences = set()
        for key in metrics.keys():
            if '_trans_mean_cm' in key:
                seq = key.replace('_trans_mean_cm', '')
                if seq != 'overall':
                    sequences.add(seq)
        
        for seq in sorted(sequences):
            print(f"\nSequence {seq}:")
            print(f"   Translation: {metrics[f'{seq}_trans_mean_cm']:.3f} cm (mean), "
                  f"{metrics[f'{seq}_trans_median_cm']:.3f} cm (median)")
            print(f"   Rotation:    {metrics[f'{seq}_rot_mean_deg']:.3f}° (mean), "
                  f"{metrics[f'{seq}_rot_median_deg']:.3f}° (median)")
        
        print("\n" + "="*60)
        print("Note: AR/VR metrics show absolute frame-to-frame errors")
        print("KITTI metrics show percentage error normalized by distance")
        print("="*60 + "\n")

    def calculate_rmse(self, pose_est, pose_gt):
        t_rmse = np.sqrt(np.mean(np.sum((pose_est[:, 3:] - pose_gt[:, 3:])**2, -1)))
        r_rmse = np.sqrt(np.mean(np.sum((pose_est[:, :3] - pose_gt[:, :3])**2, -1)))
        return t_rmse, r_rmse