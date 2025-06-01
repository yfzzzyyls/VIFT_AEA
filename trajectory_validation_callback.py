#!/usr/bin/env python3
"""
Trajectory-based validation callback for Lightning training.
Provides real-world trajectory metrics during training.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import lightning as L
from lightning.pytorch.callbacks import Callback


class TrajectoryValidationCallback(Callback):
    """
    Lightning callback that computes trajectory-based metrics during validation.
    Provides honest assessment of model performance with error accumulation.
    """
    
    def __init__(self, log_every_n_epochs: int = 5):
        """
        Args:
            log_every_n_epochs: How often to compute trajectory metrics (expensive operation)
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        
    def quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])
        return R

    def pose_to_matrix(self, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """Convert rotation (quaternion) and translation to 4x4 transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self.quaternion_to_rotation_matrix(rotation)
        T[:3, 3] = translation
        return T

    def accumulate_trajectory(self, relative_poses: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        """
        Accumulate relative poses into global trajectory.
        
        Args:
            relative_poses: List of (rotation_quat, translation) tuples
            
        Returns:
            List of 4x4 transformation matrices representing global poses
        """
        trajectory = [np.eye(4)]  # Start at origin
        
        for rotation, translation in relative_poses:
            # Convert relative pose to transformation matrix
            T_rel = self.pose_to_matrix(rotation, translation)
            
            # Accumulate: T_global_new = T_global_prev @ T_rel
            T_global = trajectory[-1] @ T_rel
            trajectory.append(T_global)
        
        return trajectory

    def compute_ate(self, traj_est: List[np.ndarray], traj_gt: List[np.ndarray]) -> float:
        """Compute Absolute Trajectory Error (ATE)."""
        assert len(traj_est) == len(traj_gt), "Trajectories must have same length"
        
        # Extract translations
        trans_est = np.array([T[:3, 3] for T in traj_est])
        trans_gt = np.array([T[:3, 3] for T in traj_gt])
        
        # Compute ATE
        errors = np.linalg.norm(trans_est - trans_gt, axis=1)
        ate = np.sqrt(np.mean(errors**2))
        
        return ate

    def compute_rpe(self, traj_est: List[np.ndarray], traj_gt: List[np.ndarray], 
                    delta_t: int = 1) -> Tuple[float, float]:
        """Compute Relative Pose Error (RPE)."""
        trans_errors = []
        rot_errors = []
        
        for i in range(len(traj_est) - delta_t):
            # Compute relative poses
            T_rel_est = np.linalg.inv(traj_est[i]) @ traj_est[i + delta_t]
            T_rel_gt = np.linalg.inv(traj_gt[i]) @ traj_gt[i + delta_t]
            
            # Compute relative error
            T_error = np.linalg.inv(T_rel_gt) @ T_rel_est
            
            # Translation error
            trans_error = np.linalg.norm(T_error[:3, 3])
            trans_errors.append(trans_error)
            
            # Rotation error
            R_error = T_error[:3, :3]
            trace = np.clip((np.trace(R_error) - 1) / 2, -1, 1)
            rot_error = np.arccos(trace) * 180 / np.pi
            rot_errors.append(rot_error)
        
        trans_rpe = np.sqrt(np.mean(np.array(trans_errors)**2))
        rot_rpe = np.sqrt(np.mean(np.array(rot_errors)**2))
        
        return trans_rpe, rot_rpe

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Compute trajectory metrics at end of validation epoch."""
        
        # Only compute trajectory metrics every N epochs (expensive)
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
            
        if not hasattr(trainer, 'datamodule') or trainer.datamodule is None:
            return
            
        # Get validation dataloader
        val_dataloader = trainer.datamodule.val_dataloader()
        if val_dataloader is None:
            return
            
        print(f"\n🎯 Computing trajectory metrics at epoch {trainer.current_epoch}...")
        
        pl_module.eval()
        device = next(pl_module.parameters()).device
        
        all_ates = []
        all_rpe_trans = []
        all_rpe_rot = []
        
        with torch.no_grad():
            # Sample a few batches for trajectory evaluation (expensive operation)
            for batch_idx, batch in enumerate(val_dataloader):
                if batch_idx >= 5:  # Limit to 5 batches for speed
                    break
                    
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_size = batch['poses'].shape[0]
                seq_len = batch['poses'].shape[1]
                
                # Get predictions for sequence
                predictions_list = []
                targets_list = []
                
                # Predict for each frame in sequence
                for i in range(1, min(seq_len, 6)):  # Limit sequence length for speed
                    window_end = i + 1
                    window_start = max(0, window_end - 11)
                    
                    if window_end <= seq_len:
                        window_batch = {
                            'images': batch['images'][:, window_start:window_end],
                            'imus': batch['imus'][:, window_start:window_end],
                            'poses': batch['poses'][:, window_start:window_end]
                        }
                        
                        pred = pl_module(window_batch)
                        
                        predictions_list.append({
                            'rotation': pred['rotation'].cpu().numpy(),
                            'translation': pred['translation'].cpu().numpy()
                        })
                        
                        targets_list.append({
                            'rotation': batch['poses'][:, i, 3:7].cpu().numpy(),
                            'translation': batch['poses'][:, i, :3].cpu().numpy()
                        })
                
                # Compute trajectory metrics for each sequence in batch
                for b in range(min(batch_size, 2)):  # Limit to 2 sequences per batch
                    if len(predictions_list) < 2:
                        continue
                        
                    # Extract predictions and targets for this sequence
                    pred_poses = [(pred['rotation'][b], pred['translation'][b]) 
                                 for pred in predictions_list]
                    gt_poses = [(target['rotation'][b], target['translation'][b]) 
                               for target in targets_list]
                    
                    try:
                        # Accumulate trajectories
                        traj_est = self.accumulate_trajectory(pred_poses)
                        traj_gt = self.accumulate_trajectory(gt_poses)
                        
                        # Compute metrics
                        ate = self.compute_ate(traj_est, traj_gt)
                        all_ates.append(ate)
                        
                        trans_rpe, rot_rpe = self.compute_rpe(traj_est, traj_gt, delta_t=1)
                        all_rpe_trans.append(trans_rpe)
                        all_rpe_rot.append(rot_rpe)
                        
                    except Exception as e:
                        # Skip sequences that cause numerical issues
                        continue
        
        # Log trajectory metrics
        if all_ates:
            ate_mean = np.mean(all_ates)
            rpe_trans_mean = np.mean(all_rpe_trans)
            rpe_rot_mean = np.mean(all_rpe_rot)
            
            # Log to trainer
            pl_module.log('val/trajectory_ate', ate_mean, on_epoch=True, prog_bar=True)
            pl_module.log('val/trajectory_rpe_trans', rpe_trans_mean, on_epoch=True, prog_bar=True)
            pl_module.log('val/trajectory_rpe_rot', rpe_rot_mean, on_epoch=True, prog_bar=True)
            
            # Convert to AR/VR relevant units
            ate_cm = ate_mean * 100
            rpe_trans_cm = rpe_trans_mean * 100
            
            print(f"📊 Trajectory Metrics (Epoch {trainer.current_epoch}):")
            print(f"   ATE: {ate_cm:.2f}cm")
            print(f"   RPE Translation: {rpe_trans_cm:.2f}cm")
            print(f"   RPE Rotation: {rpe_rot_mean:.2f}°")
            
            # Assessment for AR/VR suitability
            if ate_cm < 5.0:
                print("   ✅ Professional AR/VR grade trajectory accuracy!")
            elif ate_cm < 20.0:
                print("   🟡 Good for AR/VR demos and research")
            else:
                print("   ❌ Needs improvement for AR/VR applications")
        
        pl_module.train()  # Return to training mode