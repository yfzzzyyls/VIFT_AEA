from .base_tester import BaseTester
import torch
import numpy as np
from typing import Dict, Any
from src.utils.kitti_utils import read_pose_from_text
from src.utils.kitti_latent_eval import KITTI_tester_latent
from scipy.spatial.transform import Rotation as R
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dataclasses import dataclass
import os

class KITTILatentTester(BaseTester):
    def __init__(self, val_seqs, data_dir, seq_len, folder, img_w, img_h, wrapper_weights_path, device, v_f_len, i_f_len, use_history_in_eval=False):
        super().__init__()
        self.val_seq = val_seqs
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.folder = folder
        self.img_w = img_w
        self.img_h = img_h
        self.wrapper_weights_path = wrapper_weights_path
        self.device = device
        self.v_f_len = v_f_len
        self.i_f_len = i_f_len

        @dataclass
        class Args:
            val_seq: list
            data_dir: str
            seq_len: int
            folder: str
            img_w: int
            img_h: int
            device: str
            v_f_len: int
            i_f_len: int
            imu_dropout: float

        self.args = Args(self.val_seq, self.data_dir, self.seq_len, self.folder, self.img_w, self.img_h, self.device, self.v_f_len, self.i_f_len, 0.1)

        self.kitti_latent_tester = KITTI_tester_latent(self.args, self.wrapper_weights_path, use_history_in_eval=use_history_in_eval)
    
    def test(self, model: torch.nn.Module) -> Dict[str, Any]:
        results = {}
        for i, seq in enumerate(self.val_seq):
            print(f"Testing sequence {i+1} of {len(self.val_seq)}")
            pose_est = self.kitti_latent_tester.test_one_path(model, self.kitti_latent_tester.dataloader[i])
            pose_gt = self.kitti_latent_tester.dataloader[i].poses_rel
            
            results[seq] = {
                'estimated_poses': pose_est,
                'gt_poses': pose_gt
            }

        return results

    def accumulate_poses_with_rotation(self, relative_poses):
        """Convert relative poses to absolute trajectory with both position and rotation"""
        trajectory = []
        current_pos = np.array([0.0, 0.0, 0.0])
        current_rot = np.eye(3)
        
        # Add initial pose
        trajectory.append({
            'position': current_pos.copy(),
            'rotation_matrix': current_rot.copy(),
            'rotation_euler': np.array([0.0, 0.0, 0.0]),  # Initial rotation
        })
        
        for i, rel_pose in enumerate(relative_poses):
            # Extract relative rotation and translation
            rel_euler = rel_pose[:3]  # [roll, pitch, yaw] in radians
            rel_trans = rel_pose[3:]  # [x, y, z] in meters
            
            # Convert Euler to rotation matrix
            rel_rot_matrix = R.from_euler('xyz', rel_euler).as_matrix()
            
            # Apply transformation
            current_pos = current_pos + current_rot @ rel_trans
            current_rot = current_rot @ rel_rot_matrix
            
            # Get absolute rotation in different formats
            abs_rotation = R.from_matrix(current_rot)
            abs_euler = abs_rotation.as_euler('xyz')  # radians
            
            trajectory.append({
                'position': current_pos.copy(),
                'rotation_matrix': current_rot.copy(),
                'rotation_euler': abs_euler,
            })
        
        return trajectory

    def print_trajectory_table(self, trajectory, label, max_frames=100):
        """Print trajectory in a formatted table"""
        print(f"\n{'='*120}")
        print(f"{label} - Absolute Trajectory (First {min(len(trajectory), max_frames)} frames)")
        print(f"{'='*120}")
        print(f"{'Frame':>5} | {'X (m)':>10} {'Y (m)':>10} {'Z (m)':>10} | "
              f"{'Roll (°)':>10} {'Pitch (°)':>10} {'Yaw (°)':>10} | "
              f"{'Dist (m)':>10}")
        print(f"{'-'*120}")
        
        prev_pos = np.array([0.0, 0.0, 0.0])
        
        for i, pose in enumerate(trajectory[:max_frames]):
            pos = pose['position']
            euler_deg = np.rad2deg(pose['rotation_euler'])
            
            # Calculate distance from previous frame
            dist_from_prev = np.linalg.norm(pos - prev_pos)
            prev_pos = pos.copy()
            
            print(f"{i:>5} | {pos[0]:>10.6f} {pos[1]:>10.6f} {pos[2]:>10.6f} | "
                  f"{euler_deg[0]:>10.3f} {euler_deg[1]:>10.3f} {euler_deg[2]:>10.3f} | "
                  f"{dist_from_prev:>10.6f}")
        
        # Print summary statistics
        positions = np.array([pose['position'] for pose in trajectory[:max_frames]])
        total_distance = 0
        for i in range(1, len(positions)):
            total_distance += np.linalg.norm(positions[i] - positions[i-1])
        
        print(f"{'-'*120}")
        print(f"Summary: Total distance traveled: {total_distance:.3f}m, "
              f"Final position: ({positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}, {positions[-1, 2]:.3f})m")

    def save_trajectory_to_csv(self, trajectory, filename, max_frames=100):
        """Save trajectory to CSV file for further analysis"""
        data = []
        
        for i, pose in enumerate(trajectory[:max_frames]):
            pos = pose['position']
            euler_deg = np.rad2deg(pose['rotation_euler'])
            
            data.append({
                'frame': i,
                'x_m': pos[0],
                'y_m': pos[1],
                'z_m': pos[2],
                'roll_deg': euler_deg[0],
                'pitch_deg': euler_deg[1],
                'yaw_deg': euler_deg[2],
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, float_format='%.6f')
        print(f"Saved trajectory to: {filename}")

    def plot_3d_trajectory(self, gt_trajectory, pred_trajectory, sequence, save_dir):
        """Plot 3D trajectory comparison"""
        fig = plt.figure(figsize=(12, 10))
        
        # Extract positions
        gt_positions = np.array([pose['position'] for pose in gt_trajectory])
        pred_positions = np.array([pose['position'] for pose in pred_trajectory])
        
        # Create 3D plot
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 
                'g-', linewidth=3, label='Ground Truth', alpha=0.9)
        ax.plot(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2], 
                'r--', linewidth=3, label='KITTI Model Prediction', alpha=0.9)
        
        # Mark key points
        ax.scatter(*gt_positions[0], color='green', s=200, marker='o', 
                  edgecolors='black', linewidth=2, label='Start', zorder=5)
        ax.scatter(*gt_positions[-1], color='darkgreen', s=200, marker='s', 
                  edgecolors='black', linewidth=2, label='GT End', zorder=5)
        ax.scatter(*pred_positions[-1], color='darkred', s=200, marker='^', 
                  edgecolors='black', linewidth=2, label='Pred End', zorder=5)
        
        # Calculate metrics
        final_error = np.linalg.norm(gt_positions[-1] - pred_positions[-1]) * 100  # cm
        total_distance = 0
        for i in range(1, len(gt_positions)):
            total_distance += np.linalg.norm(gt_positions[i] - gt_positions[i-1])
        
        ax.set_xlabel('X (m)', fontsize=14)
        ax.set_ylabel('Y (m)', fontsize=14)
        ax.set_zlabel('Z (m)', fontsize=14)
        ax.set_title(f'Sequence {sequence} - 5s Trajectory Comparison\n'
                    f'Final Error: {final_error:.1f} cm, GT travels {total_distance*100:.1f} cm total', 
                    fontsize=16)
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'trajectory_3d_{sequence}_5s.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved trajectory plot: {save_path}")

    def plot_3d_rotation(self, gt_trajectory, pred_trajectory, sequence, save_dir):
        """Plot 3D rotation comparison over time"""
        fig = plt.figure(figsize=(14, 10))
        
        # Extract rotations (in degrees)
        gt_rotations = np.array([np.rad2deg(pose['rotation_euler']) for pose in gt_trajectory])
        pred_rotations = np.array([np.rad2deg(pose['rotation_euler']) for pose in pred_trajectory])
        
        # Time axis (20 FPS = 0.05s per frame)
        time_steps = np.arange(len(gt_rotations)) * 0.05
        
        # Create 3 subplots for roll, pitch, yaw
        for i, (component, label) in enumerate([(0, 'Roll'), (1, 'Pitch'), (2, 'Yaw')]):
            ax = plt.subplot(3, 1, i+1)
            ax.plot(time_steps, gt_rotations[:, component], 'g-', linewidth=2, label='Ground Truth')
            ax.plot(time_steps, pred_rotations[:, component], 'r--', linewidth=2, label='KITTI Prediction')
            ax.set_ylabel(f'{label} (degrees)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # Add zero line for reference
            ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)
            
            if i == 0:
                ax.set_title(f'Sequence {sequence} - 5s Rotation Components', fontsize=14)
            if i == 2:
                ax.set_xlabel('Time (s)', fontsize=12)
        
        # Calculate rotation errors
        rot_errors = []
        for i in range(len(gt_rotations)):
            # Convert back to rotation matrices for proper error calculation
            gt_rot = gt_trajectory[i]['rotation_matrix']
            pred_rot = pred_trajectory[i]['rotation_matrix']
            
            # Calculate geodesic distance
            rel_rot = gt_rot.T @ pred_rot
            trace = np.clip(np.trace(rel_rot), -3, 3)
            angle_error = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            rot_errors.append(np.rad2deg(angle_error))
        
        mean_rot_error = np.mean(rot_errors)
        
        # Add overall title with error
        fig.text(0.5, 0.95, f'Mean Rotation Error: {mean_rot_error:.2f}°', 
                 ha='center', fontsize=12, transform=fig.transFigure)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        save_path = os.path.join(save_dir, f'rotation_3d_{sequence}_5s.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved rotation plot: {save_path}")

    def save_results(self, results: Dict[str, Any], save_dir: str):
        # Create subdirectory for trajectory CSVs
        trajectory_dir = os.path.join(save_dir, 'absolute_trajectories')
        os.makedirs(trajectory_dir, exist_ok=True)
        
        for seq_name, seq_data in results.items():
            # Save raw results as before
            np.save(os.path.join(save_dir, f'{seq_name}_estimated_poses.npy'), seq_data['estimated_poses'])
            np.save(os.path.join(save_dir, f'{seq_name}_gt_poses.npy'), seq_data['gt_poses'])
            
            predictions = seq_data['estimated_poses']
            ground_truth = seq_data['gt_poses']
            
            # Get 5 seconds worth of data (99 relative poses = 100 frames at 20 FPS)
            fps = 20
            num_frames = 5 * fps - 1  # 99 transitions for 100 frames
            
            if num_frames > len(predictions):
                num_frames = len(predictions)
                print(f"\nWarning: Only {num_frames+1} frames available for sequence {seq_name}")
            
            pred_poses = predictions[:num_frames]
            gt_poses = ground_truth[:num_frames]
            
            # Accumulate to get absolute trajectories
            print(f"\n\n{'#'*120}")
            print(f"SEQUENCE {seq_name} - ABSOLUTE TRAJECTORIES")
            print(f"{'#'*120}")
            
            gt_trajectory = self.accumulate_poses_with_rotation(gt_poses)
            pred_trajectory = self.accumulate_poses_with_rotation(pred_poses)
            
            # Print ground truth trajectory
            self.print_trajectory_table(gt_trajectory, f"GROUND TRUTH - Sequence {seq_name}", max_frames=100)
            
            # Print predicted trajectory
            self.print_trajectory_table(pred_trajectory, f"KITTI MODEL PREDICTION - Sequence {seq_name}", max_frames=100)
            
            # Calculate and print error statistics
            print(f"\n{'='*120}")
            print(f"ERROR ANALYSIS - Sequence {seq_name}")
            print(f"{'='*120}")
            
            errors = []
            for i in range(min(len(gt_trajectory), len(pred_trajectory))):
                gt_pos = gt_trajectory[i]['position']
                pred_pos = pred_trajectory[i]['position']
                pos_error = np.linalg.norm(gt_pos - pred_pos) * 100  # cm
                
                # Rotation error (geodesic distance)
                gt_rot = gt_trajectory[i]['rotation_matrix']
                pred_rot = pred_trajectory[i]['rotation_matrix']
                rel_rot = gt_rot.T @ pred_rot
                trace = np.clip(np.trace(rel_rot), -3, 3)
                rot_error = np.rad2deg(np.arccos(np.clip((trace - 1) / 2, -1, 1)))
                
                errors.append({'position': pos_error, 'rotation': rot_error})
                
                if i < 10 or i % 10 == 0:  # Print first 10 and then every 10th frame
                    print(f"Frame {i:>3}: Position error: {pos_error:>8.2f} cm, Rotation error: {rot_error:>8.2f}°")
            
            # Summary statistics
            pos_errors = [e['position'] for e in errors]
            rot_errors = [e['rotation'] for e in errors]
            
            print(f"\nPosition Error Statistics (cm):")
            print(f"  Mean: {np.mean(pos_errors):.2f}, Median: {np.median(pos_errors):.2f}, "
                  f"Std: {np.std(pos_errors):.2f}, Max: {np.max(pos_errors):.2f}")
            
            print(f"Rotation Error Statistics (degrees):")
            print(f"  Mean: {np.mean(rot_errors):.2f}, Median: {np.median(rot_errors):.2f}, "
                  f"Std: {np.std(rot_errors):.2f}, Max: {np.max(rot_errors):.2f}")
            
            # Save to CSV files
            self.save_trajectory_to_csv(gt_trajectory, 
                                      os.path.join(trajectory_dir, f'sequence_{seq_name}_ground_truth.csv'))
            self.save_trajectory_to_csv(pred_trajectory, 
                                      os.path.join(trajectory_dir, f'sequence_{seq_name}_kitti_prediction.csv'))
            
            # Generate 3D plots
            print(f"\nGenerating 3D plots for sequence {seq_name}...")
            self.plot_3d_trajectory(gt_trajectory, pred_trajectory, seq_name, save_dir)
            self.plot_3d_rotation(gt_trajectory, pred_trajectory, seq_name, save_dir)
