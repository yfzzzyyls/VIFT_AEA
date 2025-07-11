#!/usr/bin/env python3
"""
Standalone evaluation script for TUM VI dataset.
Handles the specific directory structure of TUM VI downloads.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from scipy import interpolate
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Interactive 3D plots will be disabled.")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.components.vsvio import TransformerVIO
from umeyama_alignment import align_trajectory, compute_ate


class TUMVIDatasetDirect(Dataset):
    """Direct TUM VI dataset loader for extracted sequences."""
    
    def __init__(self, 
                 sequence_dir,
                 sequence_length=11,
                 stride=10):
        """
        Args:
            sequence_dir: Direct path to extracted sequence (e.g., /path/to/dataset-room1_512_16)
            sequence_length: Number of frames per sample (default: 11)
            stride: Stride between samples (default: 10)
        """
        self.sequence_dir = Path(sequence_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Verify structure
        self.mav0_dir = self.sequence_dir / 'mav0'
        if not self.mav0_dir.exists():
            raise ValueError(f"Invalid TUM VI structure: {self.mav0_dir} not found")
        
        # Camera is always cam0 (left)
        self.camera = 'cam0'
        
        # Load data
        self._load_timestamps()
        self._load_imu_data()
        self._load_ground_truth()
        self._create_valid_indices()
        
    def _load_timestamps(self):
        """Load image timestamps."""
        cam_data_path = self.mav0_dir / self.camera / 'data.csv'
        
        if not cam_data_path.exists():
            raise FileNotFoundError(f"Camera data file not found: {cam_data_path}")
        
        # Read CSV (timestamp [ns], filename)
        df = pd.read_csv(cam_data_path, header=0, names=['timestamp', 'filename'])
        self.image_timestamps = df['timestamp'].values  # in nanoseconds
        self.image_filenames = df['filename'].values
        
        # Convert to seconds
        self.image_timestamps_sec = self.image_timestamps * 1e-9
        
    def _load_imu_data(self):
        """Load and preprocess IMU data."""
        imu_data_path = self.mav0_dir / 'imu0' / 'data.csv'
        
        if not imu_data_path.exists():
            raise FileNotFoundError(f"IMU data file not found: {imu_data_path}")
        
        # Read IMU data (timestamp [ns], wx, wy, wz, ax, ay, az)
        df = pd.read_csv(imu_data_path, header=0, 
                        names=['timestamp', 'wx', 'wy', 'wz', 'ax', 'ay', 'az'])
        
        self.imu_timestamps = df['timestamp'].values * 1e-9  # Convert to seconds
        
        # Stack as [N, 6] array with order matching VIFT (accel first, then gyro)
        self.imu_data = np.stack([
            df['ax'].values, df['ay'].values, df['az'].values,
            df['wx'].values, df['wy'].values, df['wz'].values
        ], axis=1).astype(np.float32)
        
    def _load_ground_truth(self):
        """Load ground truth poses."""
        gt_path = self.mav0_dir / 'mocap0' / 'data.csv'
        
        if not gt_path.exists():
            # Try alternative GT path
            gt_path = self.mav0_dir / 'state_groundtruth_estimate0' / 'data.csv'
        
        if not gt_path.exists():
            print(f"Warning: Ground truth not found")
            self.has_gt = False
            return
        
        # Read ground truth (timestamp [ns], px, py, pz, qw, qx, qy, qz, ...)
        df = pd.read_csv(gt_path, header=0)
        
        self.gt_timestamps = df.iloc[:, 0].values * 1e-9  # Convert to seconds
        self.gt_positions = df.iloc[:, 1:4].values  # px, py, pz
        
        # TUM VI uses Hamilton convention (qw, qx, qy, qz)
        # Convert to our convention (qx, qy, qz, qw)
        self.gt_quaternions = np.stack([
            df.iloc[:, 5].values,  # qx
            df.iloc[:, 6].values,  # qy
            df.iloc[:, 7].values,  # qz
            df.iloc[:, 4].values   # qw
        ], axis=1)
        
        self.has_gt = True
        
    def _create_valid_indices(self):
        """Create valid starting indices for sequences."""
        max_start_idx = len(self.image_timestamps) - self.sequence_length
        
        valid_indices = []
        for idx in range(0, max_start_idx, self.stride):
            # Check if we have IMU data for this time range
            start_time = self.image_timestamps_sec[idx]
            end_time = self.image_timestamps_sec[idx + self.sequence_length - 1]
            
            # Need IMU data covering this range
            imu_mask = (self.imu_timestamps >= start_time - 0.1) & \
                      (self.imu_timestamps <= end_time + 0.1)
            
            if np.sum(imu_mask) >= 110:  # Need at least 110 IMU samples
                valid_indices.append(idx)
        
        self.valid_indices = valid_indices
        
    def _sync_imu_to_images(self, start_idx):
        """Synchronize IMU data to image timestamps."""
        img_times = self.image_timestamps_sec[start_idx:start_idx + self.sequence_length]
        
        synced_imu = []
        for i in range(self.sequence_length - 1):
            t_start = img_times[i]
            t_end = img_times[i + 1]
            
            # VIFT expects 11 IMU samples per transition
            # With 20Hz camera and 200Hz IMU, we naturally have 10 samples
            # So we interpolate to get 11 samples
            t_samples = np.linspace(t_start, t_end, 11)  # 11 samples including boundaries
            
            interpolated_imu = np.zeros((11, 6))
            for j in range(6):
                f = interpolate.interp1d(self.imu_timestamps, self.imu_data[:, j], 
                                       kind='linear', fill_value='extrapolate')
                interpolated_imu[:, j] = f(t_samples)
            
            synced_imu.append(interpolated_imu)
        
        # Stack to get [110, 6] array (10 transitions × 11 samples)
        return np.vstack(synced_imu).astype(np.float32)
        
    def _get_relative_poses(self, start_idx):
        """Get relative poses between consecutive frames."""
        if not self.has_gt:
            return torch.zeros(self.sequence_length - 1, 7)
        
        img_times = self.image_timestamps_sec[start_idx:start_idx + self.sequence_length]
        
        # Interpolate ground truth to image timestamps
        positions = np.zeros((self.sequence_length, 3))
        quaternions = np.zeros((self.sequence_length, 4))
        
        for i in range(self.sequence_length):
            t = img_times[i]
            
            # Find nearest GT timestamps
            idx = np.searchsorted(self.gt_timestamps, t)
            if idx == 0:
                positions[i] = self.gt_positions[0]
                quaternions[i] = self.gt_quaternions[0]
            elif idx >= len(self.gt_timestamps):
                positions[i] = self.gt_positions[-1]
                quaternions[i] = self.gt_quaternions[-1]
            else:
                # Linear interpolation for position
                t0, t1 = self.gt_timestamps[idx-1], self.gt_timestamps[idx]
                alpha = (t - t0) / (t1 - t0)
                positions[i] = (1 - alpha) * self.gt_positions[idx-1] + alpha * self.gt_positions[idx]
                
                # SLERP for quaternion
                q0 = R.from_quat(self.gt_quaternions[idx-1])
                q1 = R.from_quat(self.gt_quaternions[idx])
                q_interp = R.from_quat(q0.as_quat()) * R.from_quat((q0.inv() * q1).as_quat()) ** alpha
                quaternions[i] = q_interp.as_quat()
        
        # Compute relative poses
        relative_poses = []
        for i in range(self.sequence_length - 1):
            # Relative translation
            rel_trans = positions[i+1] - positions[i]
            
            # Relative rotation
            q1 = R.from_quat(quaternions[i])
            q2 = R.from_quat(quaternions[i+1])
            rel_rot = q1.inv() * q2
            rel_quat = rel_rot.as_quat()
            
            # Combine [trans(3), quat(4)]
            rel_pose = np.concatenate([rel_trans, rel_quat])
            relative_poses.append(rel_pose)
        
        return torch.tensor(np.array(relative_poses), dtype=torch.float32)
        
    def _load_and_preprocess_image(self, idx):
        """Load and preprocess a single image."""
        img_path = self.mav0_dir / self.camera / 'data' / self.image_filenames[idx]
        
        # Load image
        img = Image.open(img_path)
        
        # Keep original resolution for 512x512
        if img.size == (1024, 1024):
            # Use 2x2 binning for 1024x1024 images (exact 2x downsampling)
            img_array = np.array(img)
            img_array = img_array.reshape(512, 2, 512, 2).mean(axis=(1, 3))
            img = Image.fromarray(img_array.astype(np.uint8))
        # else: keep 512x512 as is
        
        # Convert to RGB if grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to tensor and normalize
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        return img
        
    def __len__(self):
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        """Get a single sample."""
        start_idx = self.valid_indices[idx]
        
        # Load sequence of images
        images = []
        for i in range(self.sequence_length):
            img = self._load_and_preprocess_image(start_idx + i)
            images.append(img)
        images = torch.stack(images)  # [11, 3, 256, 512]
        
        # Get synchronized IMU data
        imu = self._sync_imu_to_images(start_idx)  # [110, 6]
        imu = torch.from_numpy(imu)
        
        # Get relative poses
        gt_poses = self._get_relative_poses(start_idx)  # [10, 7]
        
        return {
            'images': images,
            'imu': imu,
            'gt_poses': gt_poses,
            'sequence_name': f"sample_{start_idx:06d}"
        }


class VIFTFromScratch(torch.nn.Module):
    """VIFT model wrapper for evaluation."""
    
    def __init__(self, encoder_type='flownet', transformer_layers=8, transformer_heads=16, 
                 transformer_dim_feedforward=4096, transformer_dropout=0.1):
        super().__init__()
        
        # Model configuration (must match training)
        class Config:
            # Sequence parameters
            seq_len = 11
            
            # Image parameters
            img_w = 512
            img_h = 512  # Updated to support square images
            
            # Feature dimensions
            v_f_len = 512  # Visual feature dimension
            i_f_len = 256  # IMU feature dimension
            
            # IMU encoder parameters
            imu_dropout = 0.2
            
            # Transformer parameters
            embedding_dim = 768  # v_f_len + i_f_len
            num_layers = transformer_layers
            nhead = transformer_heads
            dim_feedforward = transformer_dim_feedforward
            dropout = transformer_dropout
            
            # For compatibility
            rnn_hidden_size = 512
            rnn_dropout_between = 0.1
            rnn_dropout_out = 0.1
            fuse_method = 'cat'
        
        self.config = Config()
        # Add encoder type after creating config
        self.config.encoder_type = encoder_type
        self.backbone = TransformerVIO(self.config)
        
        # Replace output layer for quaternion output
        hidden_dim = self.config.embedding_dim
        self.pose_predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim // 2, 7)  # 3 trans + 4 quat
        )
    
    def forward(self, batch):
        """Forward pass."""
        images = batch['images']  # [B, 11, 3, 256, 512]
        imu = batch['imu']        # [B, 110, 6]
        
        # Get features from backbone
        fv, fi = self.backbone.Feature_net(images, imu)
        
        # Concatenate features
        combined_features = torch.cat([fv, fi], dim=-1)  # [B, 10, 768]
        
        # Apply pose predictor
        batch_size, seq_len, feat_dim = combined_features.shape
        combined_flat = combined_features.reshape(-1, feat_dim)
        poses_flat = self.pose_predictor(combined_flat)
        poses = poses_flat.reshape(batch_size, seq_len, 7)
        
        # Normalize quaternion part
        trans = poses[:, :, :3]
        quat = poses[:, :, 3:]
        quat = torch.nn.functional.normalize(quat, p=2, dim=-1)
        
        return {
            'poses': torch.cat([trans, quat], dim=-1)
        }


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    # q = [qx, qy, qz, qw]
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    
    return R


def integrate_poses(relative_poses):
    """Integrate relative poses to get absolute trajectory."""
    positions = [np.zeros(3)]
    rotations = [np.eye(3)]
    quaternions = [np.array([0, 0, 0, 1])]  # Identity quaternion
    
    for rel_pose in relative_poses:
        # Extract translation and rotation
        trans = rel_pose[:3]
        quat = rel_pose[3:]
        
        # Normalize quaternion
        quat = quat / np.linalg.norm(quat)
        
        # Convert quaternion to rotation matrix
        R_rel = quaternion_to_rotation_matrix(quat)
        
        # Update absolute pose
        R_abs = rotations[-1] @ R_rel
        t_abs = positions[-1] + rotations[-1] @ trans
        
        # Convert absolute rotation back to quaternion
        r_abs = R.from_matrix(R_abs)
        q_abs = r_abs.as_quat()  # Returns [qx, qy, qz, qw]
        
        positions.append(t_abs)
        rotations.append(R_abs)
        quaternions.append(q_abs)
    
    return np.array(positions), np.array(quaternions)


def save_trajectory_csv(pred_positions, pred_rotations, gt_positions, gt_rotations, 
                       sequence_name, output_dir):
    """Save trajectory data to CSV files."""
    import pandas as pd
    
    # Save ground truth CSV
    gt_data = []
    for i in range(len(gt_positions)):
        row = {
            'frame': i,
            'x': gt_positions[i, 0],
            'y': gt_positions[i, 1],
            'z': gt_positions[i, 2]
        }
        if i < len(gt_rotations):
            # Store quaternions
            row.update({
                'qx': gt_rotations[i][0],
                'qy': gt_rotations[i][1],
                'qz': gt_rotations[i][2],
                'qw': gt_rotations[i][3]
            })
        gt_data.append(row)
    
    gt_df = pd.DataFrame(gt_data)
    gt_csv_path = output_dir / f'trajectory_{sequence_name}_gt.csv'
    gt_df.to_csv(gt_csv_path, index=False)
    print(f"Saved ground truth to {gt_csv_path}")
    
    # Save prediction CSV with errors
    pred_data = []
    for i in range(len(pred_positions)):
        row = {
            'frame': i,
            'x': pred_positions[i, 0],
            'y': pred_positions[i, 1],
            'z': pred_positions[i, 2]
        }
        
        if i < len(pred_rotations):
            # Store quaternions
            row.update({
                'qx': pred_rotations[i][0],
                'qy': pred_rotations[i][1],
                'qz': pred_rotations[i][2],
                'qw': pred_rotations[i][3]
            })
        
        # Add errors compared to ground truth
        if i < len(gt_positions):
            row['trans_error'] = np.linalg.norm(pred_positions[i] - gt_positions[i])
            
            if i < len(gt_rotations) and i < len(pred_rotations):
                # Compute rotation error
                gt_rot = R.from_quat(gt_rotations[i])
                pred_rot = R.from_quat(pred_rotations[i])
                rel_rot = gt_rot.inv() * pred_rot
                rot_error = np.abs(rel_rot.magnitude() * 180 / np.pi)
                row['rot_error_deg'] = rot_error
        
        pred_data.append(row)
    
    pred_df = pd.DataFrame(pred_data)
    pred_csv_path = output_dir / f'trajectory_{sequence_name}_pred.csv'
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"Saved predictions to {pred_csv_path}")


def evaluate_sequence(model, dataset, device, sequence_name):
    """Evaluate model on a single sequence."""
    model.eval()
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=4
    )
    
    all_predictions = []
    all_ground_truth = []
    
    print(f"\nEvaluating sequence: {sequence_name}")
    print(f"Number of samples: {len(dataset)}")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Processing {sequence_name}"):
            # Move to device
            batch_gpu = {
                'images': batch['images'].to(device),
                'imu': batch['imu'].to(device).float()
            }
            
            # Get predictions
            predictions = model(batch_gpu)
            pred_poses = predictions['poses'].cpu().numpy()[0]  # [10, 7]
            
            # Get ground truth
            gt_poses = batch['gt_poses'].numpy()[0]  # [10, 7]
            
            all_predictions.append(pred_poses)
            all_ground_truth.append(gt_poses)
    
    # Stack all predictions
    all_predictions = np.vstack(all_predictions)  # [N*10, 7]
    all_ground_truth = np.vstack(all_ground_truth)  # [N*10, 7]
    
    # Integrate to get full trajectories
    pred_positions, pred_rotations = integrate_poses(all_predictions)
    gt_positions, gt_rotations = integrate_poses(all_ground_truth)
    
    # Compute metrics
    # Translation error
    trans_errors = np.linalg.norm(all_predictions[:, :3] - all_ground_truth[:, :3], axis=1)
    mean_trans_error = np.mean(trans_errors)
    
    # Rotation error (geodesic distance)
    rot_errors = []
    for i in range(len(all_predictions)):
        q_pred = all_predictions[i, 3:]
        q_gt = all_ground_truth[i, 3:]
        
        # Normalize
        q_pred = q_pred / np.linalg.norm(q_pred)
        q_gt = q_gt / np.linalg.norm(q_gt)
        
        # Compute angle
        dot = np.clip(np.abs(np.dot(q_pred, q_gt)), -1.0, 1.0)
        angle = 2 * np.arccos(dot)
        rot_errors.append(np.degrees(angle))
    
    mean_rot_error = np.mean(rot_errors)
    
    # Compute ATE on integrated trajectory
    # compute_ate returns (ate, aligned_trajectory, (R, t, s))
    ate_raw, aligned_pred = compute_ate(pred_positions, gt_positions)
    
    # Align trajectories using Umeyama
    pred_aligned, R_align, t_align, s_align = align_trajectory(
        pred_positions[1:], 
        gt_positions[:len(pred_positions)-1],
        with_scale=True
    )
    ate_aligned, _ = compute_ate(pred_aligned, gt_positions[:len(pred_aligned)])
    
    metrics = {
        'sequence': sequence_name,
        'num_samples': len(dataset),
        'mean_trans_error': mean_trans_error,
        'mean_rot_error': mean_rot_error,
        'ate_raw': ate_raw,
        'ate_aligned': ate_aligned,
        'scale': s_align
    }
    
    return metrics, pred_positions, gt_positions, pred_aligned, pred_rotations, gt_rotations


def create_interactive_3d_plot(pred_positions, gt_positions, pred_aligned, sequence_name, output_path):
    """Create interactive 3D trajectory plot using Plotly"""
    if not PLOTLY_AVAILABLE:
        return
    
    # Convert to centimeters
    pred_positions_cm = pred_positions * 100
    gt_positions_cm = gt_positions * 100
    pred_aligned_cm = pred_aligned * 100
    
    # Create figure
    fig = go.Figure()
    
    # Add ground truth trajectory
    fig.add_trace(go.Scatter3d(
        x=gt_positions_cm[:, 0],
        y=gt_positions_cm[:, 1],
        z=gt_positions_cm[:, 2],
        mode='lines',
        name='Ground Truth',
        line=dict(color='green', width=4),
        hovertemplate='GT<br>X: %{x:.2f}cm<br>Y: %{y:.2f}cm<br>Z: %{z:.2f}cm<extra></extra>'
    ))
    
    # Add predicted trajectory
    fig.add_trace(go.Scatter3d(
        x=pred_positions_cm[:, 0],
        y=pred_positions_cm[:, 1],
        z=pred_positions_cm[:, 2],
        mode='lines',
        name='Prediction (Raw)',
        line=dict(color='blue', width=4),
        hovertemplate='Pred<br>X: %{x:.2f}cm<br>Y: %{y:.2f}cm<br>Z: %{z:.2f}cm<extra></extra>'
    ))
    
    # Add aligned predicted trajectory
    fig.add_trace(go.Scatter3d(
        x=pred_aligned_cm[:, 0],
        y=pred_aligned_cm[:, 1],
        z=pred_aligned_cm[:, 2],
        mode='lines',
        name='Prediction (Aligned)',
        line=dict(color='red', width=4, dash='dash'),
        hovertemplate='Aligned<br>X: %{x:.2f}cm<br>Y: %{y:.2f}cm<br>Z: %{z:.2f}cm<extra></extra>'
    ))
    
    # Add start and end markers
    fig.add_trace(go.Scatter3d(
        x=[gt_positions_cm[0, 0]],
        y=[gt_positions_cm[0, 1]],
        z=[gt_positions_cm[0, 2]],
        mode='markers',
        name='Start',
        marker=dict(color='green', size=10, symbol='circle'),
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[gt_positions_cm[-1, 0]],
        y=[gt_positions_cm[-1, 1]],
        z=[gt_positions_cm[-1, 2]],
        mode='markers',
        name='End',
        marker=dict(color='red', size=10, symbol='x'),
        showlegend=True
    ))
    
    # Calculate path lengths and errors
    gt_length = np.sum(np.linalg.norm(np.diff(gt_positions, axis=0), axis=1)) * 100
    pred_length = np.sum(np.linalg.norm(np.diff(pred_positions, axis=0), axis=1)) * 100
    ape_mean = np.mean(np.linalg.norm(pred_positions - gt_positions[:len(pred_positions)], axis=1)) * 100
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'TUM VI - {sequence_name} - Interactive 3D Trajectory<br>' +
                 f'GT Length: {gt_length:.1f}cm, Pred Length: {pred_length:.1f}cm, APE: {ape_mean:.2f}cm',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='X (cm)',
            yaxis_title='Y (cm)',
            zaxis_title='Z (cm)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1200,
        height=800,
        hovermode='closest'
    )
    
    # Save HTML
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"Saved interactive 3D plot to {output_path}")


def plot_trajectory_3d_static(pred_positions, gt_positions, pred_aligned, sequence_name, output_path):
    """Create static 3D trajectory plot with matplotlib"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert to centimeters for display
    pred_positions_cm = pred_positions * 100
    gt_positions_cm = gt_positions * 100
    pred_aligned_cm = pred_aligned * 100
    
    # Plot trajectories
    ax.plot(gt_positions_cm[:, 0], gt_positions_cm[:, 1], gt_positions_cm[:, 2], 
            'g-', linewidth=2, label='Ground Truth')
    ax.plot(pred_positions_cm[:, 0], pred_positions_cm[:, 1], pred_positions_cm[:, 2], 
            'b-', linewidth=2, label='Prediction (Raw)', alpha=0.8)
    ax.plot(pred_aligned_cm[:, 0], pred_aligned_cm[:, 1], pred_aligned_cm[:, 2], 
            'r--', linewidth=2, label='Prediction (Aligned)', alpha=0.8)
    
    # Mark start and end
    ax.scatter(*gt_positions_cm[0], color='green', s=100, marker='o', label='Start')
    ax.scatter(*gt_positions_cm[-1], color='red', s=100, marker='x', label='End')
    
    # Calculate path lengths
    gt_length = np.sum(np.linalg.norm(np.diff(gt_positions, axis=0), axis=1)) * 100
    pred_length = np.sum(np.linalg.norm(np.diff(pred_positions, axis=0), axis=1)) * 100
    
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    
    # Set equal aspect ratio for all axes
    try:
        ax.set_box_aspect([1, 1, 1])  # Available in matplotlib >= 3.4
    except AttributeError:
        # Fallback for older matplotlib versions
        pass
    
    ax.set_title(f'TUM VI - {sequence_name} - 3D Trajectory\nGT Length: {gt_length:.1f}cm, Pred Length: {pred_length:.1f}cm')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved static 3D plot to {output_path}")


def plot_trajectory(pred_positions, gt_positions, pred_aligned, sequence_name, output_dir):
    """Plot 3D trajectories."""
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Raw trajectories
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 
             'g-', label='Ground Truth', linewidth=2)
    ax1.plot(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2], 
             'b-', label='Prediction', linewidth=2)
    ax1.set_title(f'{sequence_name} - Raw Trajectories')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()
    ax1.axis('equal')
    
    # Plot 2: Aligned trajectories
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot(gt_positions[:len(pred_aligned), 0], 
             gt_positions[:len(pred_aligned), 1], 
             gt_positions[:len(pred_aligned), 2], 
             'g-', label='Ground Truth', linewidth=2)
    ax2.plot(pred_aligned[:, 0], pred_aligned[:, 1], pred_aligned[:, 2], 
             'r-', label='Aligned Prediction', linewidth=2)
    ax2.set_title(f'{sequence_name} - Aligned Trajectories')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.legend()
    ax2.axis('equal')
    
    # Plot 3: Top-down view
    ax3 = fig.add_subplot(133)
    ax3.plot(gt_positions[:, 0], gt_positions[:, 1], 
             'g-', label='Ground Truth', linewidth=2)
    ax3.plot(pred_positions[:, 0], pred_positions[:, 1], 
             'b-', label='Prediction', linewidth=2)
    ax3.plot(pred_aligned[:, 0], pred_aligned[:, 1], 
             'r--', label='Aligned', linewidth=2)
    ax3.set_title(f'{sequence_name} - Top-down View')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.legend()
    ax3.axis('equal')
    ax3.grid(True)
    
    plt.tight_layout()
    output_path = output_dir / f"{sequence_name}_trajectory.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved trajectory plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate VIFT on TUM VI dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--sequence-dir', type=str, required=True,
                        help='Path to extracted TUM VI sequence (e.g., /path/to/dataset-room1_512_16)')
    parser.add_argument('--output-dir', type=str, default='evaluation_tumvi',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine sequence name from directory
    seq_dir_name = Path(args.sequence_dir).name
    if 'room' in seq_dir_name:
        # Extract room number from dataset-room1_512_16 format
        sequence_name = seq_dir_name.split('-')[1].split('_')[0] if '-' in seq_dir_name else seq_dir_name
    else:
        sequence_name = seq_dir_name
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Extract model configuration from checkpoint
    # Default to older model configuration for backward compatibility
    encoder_type = 'cnn'  # Default to CNN for old checkpoints
    transformer_layers = 4
    transformer_heads = 8
    transformer_dim_feedforward = 2048
    transformer_dropout = 0.1
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        if hasattr(config, 'encoder_type'):
            encoder_type = config.encoder_type
        if hasattr(config, 'num_layers'):
            transformer_layers = config.num_layers
        if hasattr(config, 'nhead'):
            transformer_heads = config.nhead
        if hasattr(config, 'dim_feedforward'):
            transformer_dim_feedforward = config.dim_feedforward
        if hasattr(config, 'dropout'):
            transformer_dropout = config.dropout
    
    model = VIFTFromScratch(
        encoder_type=encoder_type,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        transformer_dim_feedforward=transformer_dim_feedforward,
        transformer_dropout=transformer_dropout
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from: {args.checkpoint}")
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = TUMVIDatasetDirect(
        sequence_dir=args.sequence_dir,
        sequence_length=11,
        stride=10  # Use larger stride for evaluation
    )
    
    # Evaluate
    metrics, pred_pos, gt_pos, pred_aligned, pred_rot, gt_rot = evaluate_sequence(
        model, dataset, device, sequence_name
    )
    
    # Print metrics
    print(f"\n{sequence_name} Results:")
    print(f"  Samples evaluated: {metrics['num_samples']}")
    print(f"  Mean translation error: {metrics['mean_trans_error']*100:.2f} cm")
    print(f"  Mean rotation error: {metrics['mean_rot_error']:.2f}°")
    print(f"  ATE (raw): {metrics['ate_raw']:.3f} m")
    print(f"  ATE (aligned): {metrics['ate_aligned']:.3f} m")
    print(f"  Scale factor: {metrics['scale']:.3f}")
    
    # Plot trajectory
    plot_trajectory(pred_pos, gt_pos, pred_aligned, sequence_name, output_dir)
    
    # Create static 3D plot
    plot_trajectory_3d_static(pred_pos, gt_pos, pred_aligned, sequence_name, 
                              output_dir / f"{sequence_name}_trajectory_3d.png")
    
    # Create interactive 3D plot
    create_interactive_3d_plot(pred_pos, gt_pos, pred_aligned, sequence_name,
                               output_dir / f"{sequence_name}_trajectory_3d_interactive.html")
    
    # Save trajectory CSV files
    save_trajectory_csv(pred_pos, pred_rot, gt_pos, gt_rot, sequence_name, output_dir)
    
    # Save summary
    summary_path = output_dir / f"{sequence_name}_evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"TUM VI Evaluation Results - {sequence_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Sequence: {sequence_name}\n")
        f.write(f"  Samples: {metrics['num_samples']}\n")
        f.write(f"  Translation error: {metrics['mean_trans_error']*100:.2f} cm\n")
        f.write(f"  Rotation error: {metrics['mean_rot_error']:.2f}°\n")
        f.write(f"  ATE (raw): {metrics['ate_raw']:.3f} m\n")
        f.write(f"  ATE (aligned): {metrics['ate_aligned']:.3f} m\n")
        f.write(f"  Scale: {metrics['scale']:.3f}\n")
    
    print(f"\nSaved evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()