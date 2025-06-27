"""
Hybrid VIO-Transformer Architecture
Combines geometric VIO with learned transformer corrections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.transform import Rotation as R

from .msckf_state import MSCKFState
from .imu_preintegration import IMUPreintegration, IMUPreintegrationBuffer
from .feature_tracker_msckf import MSCKFFeatureTracker
from .msckf_update import MSCKFUpdate
from .zupt_detector import ZUPTDetector, ZUPTUpdater
from .mini_bundle_adjustment import MiniBA
from .learned_imu_bias import LearnedIMUBias, BiasedIMUPreintegration
from .adaptive_noise_model import AdaptiveNoiseModel, AdaptiveMSCKFState


class UncertaintyEstimator(nn.Module):
    """
    Estimates uncertainty for transformer predictions.
    """
    
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 6),  # 3 for translation, 3 for rotation
            nn.Softplus()  # Ensure positive uncertainty
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Estimate uncertainty from features.
        
        Args:
            features: [B, T, F] feature tensor
            
        Returns:
            uncertainty: [B, T, 6] uncertainty estimates
        """
        B, T, F = features.shape
        features_flat = features.view(B * T, F)
        uncertainty_flat = self.net(features_flat)
        return uncertainty_flat.view(B, T, 6)


class VIOTransformerHybrid(nn.Module):
    """
    Hybrid architecture combining VIO and transformer.
    
    Architecture:
    1. VIO backend provides geometric base estimate
    2. Transformer refines with learned corrections
    3. Uncertainty-weighted fusion combines both
    
    Training stages:
    - Stage 1: VIO only (no learning)
    - Stage 2: Frozen VIO + transformer training
    - Stage 3: Joint fine-tuning
    """
    
    def __init__(self, 
                 visual_encoder: nn.Module,
                 imu_encoder: nn.Module,
                 transformer: nn.Module,
                 camera_matrix: np.ndarray,
                 window_size: int = 10,
                 use_ba: bool = True,
                 use_zupt: bool = True):
        """
        Initialize hybrid VIO-transformer.
        
        Args:
            visual_encoder: Visual feature encoder (e.g., SEA-RAFT)
            imu_encoder: IMU feature encoder
            transformer: Pose estimation transformer
            camera_matrix: Camera intrinsic matrix
            window_size: MSCKF sliding window size
            use_ba: Enable mini bundle adjustment
            use_zupt: Enable ZUPT detection
        """
        super().__init__()
        
        # Encoders and transformer
        self.visual_encoder = visual_encoder
        self.imu_encoder = imu_encoder
        self.transformer = transformer
        
        # Enhanced VIO components with critical improvements
        # 1. Learned IMU bias correction
        self.imu_bias_corrector = LearnedIMUBias(
            hidden_dim=16,
            window_size=10,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 2. Adaptive noise model
        self.noise_model = AdaptiveNoiseModel(
            imu_window=20,
            hidden_dim=64,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 3. Base VIO components
        base_msckf_state = MSCKFState(window_size=window_size)
        self.msckf_state = AdaptiveMSCKFState(base_msckf_state, self.noise_model)
        
        # Wrap IMU buffer with bias correction
        base_imu_buffer = IMUPreintegrationBuffer(window_size=window_size)
        self.imu_buffer = base_imu_buffer  # We'll wrap individual preintegrators
        
        self.feature_tracker = MSCKFFeatureTracker(max_features=150)
        self.msckf_update = MSCKFUpdate(camera_matrix)  # Now with Mahalanobis gating
        
        if use_zupt:
            self.zupt_detector = ZUPTDetector()
            self.zupt_updater = ZUPTUpdater()
        else:
            self.zupt_detector = None
            
        if use_ba:
            self.mini_ba = MiniBA(device='cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.mini_ba = None
            
        # Uncertainty estimation
        self.uncertainty_estimator = UncertaintyEstimator(
            feature_dim=transformer.embedding_dim
        )
        
        # Fusion parameters
        self.vio_weight = nn.Parameter(torch.tensor(0.7))
        self.transformer_weight = nn.Parameter(torch.tensor(0.3))
        
        # Camera parameters
        self.K = torch.tensor(camera_matrix, dtype=torch.float32)
        
        # Training stage
        self.training_stage = 1  # 1: VIO only, 2: Frozen VIO, 3: Joint
        
        # Initialize bias estimate
        self.current_imu_bias = np.zeros(6)
        
    def set_training_stage(self, stage: int):
        """Set training stage for curriculum learning."""
        self.training_stage = stage
        
        if stage == 1:
            # VIO only - freeze all learning components
            for param in self.parameters():
                param.requires_grad = False
        elif stage == 2:
            # Frozen VIO + transformer
            for param in self.parameters():
                param.requires_grad = True
            # Keep VIO components frozen
            self.vio_weight.requires_grad = False
            self.transformer_weight.requires_grad = False
        else:  # stage 3
            # Joint fine-tuning
            for param in self.parameters():
                param.requires_grad = True
                
    def process_imu_measurements(self, imu_data: List[Tuple[np.ndarray, np.ndarray, float]]):
        """
        Process high-rate IMU measurements with bias correction.
        
        Args:
            imu_data: List of (acc, gyro, dt) tuples
        """
        for acc, gyro, dt in imu_data:
            # Add to bias corrector buffer
            self.imu_bias_corrector.add_imu_measurement(acc, gyro)
            
            # Add to noise model buffer
            self.noise_model.add_imu_measurement(acc, gyro)
            
            # Get bias correction
            current_bias = getattr(self, 'current_imu_bias', np.zeros(6))
            bias_correction, _ = self.imu_bias_corrector.predict_bias_correction(
                current_bias, return_numpy=True
            )
            
            # Update bias estimate
            self.current_imu_bias = current_bias + bias_correction
            
            # Apply bias correction
            acc_corrected = acc - self.current_imu_bias[:3]
            gyro_corrected = gyro - self.current_imu_bias[3:]
            
            # Add corrected measurement to pre-integrator
            self.imu_buffer.add_measurement(acc_corrected, gyro_corrected, dt)
            
            # ZUPT detection
            if self.zupt_detector is not None:
                detection = self.zupt_detector.add_measurement(acc, gyro, dt)
                if detection is not None and detection.is_stationary:
                    self.zupt_updater.apply_zupt(detection, self.msckf_state.base)
                    
    def vio_update(self, 
                  image: torch.Tensor,
                  timestamp: float,
                  frame_id: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Perform VIO update with new image.
        
        Args:
            image: Current image [B, C, H, W]
            timestamp: Current timestamp
            frame_id: Optional frame ID
            
        Returns:
            VIO estimates and uncertainties
        """
        # Get completed IMU pre-integration
        imu_preint = self.imu_buffer.new_keyframe(frame_id)
        
        if imu_preint.delta_t > 0:
            # Predict new state
            R_new, p_new, v_new = imu_preint.predict(
                self.msckf_state.imu_state.rotation,
                self.msckf_state.imu_state.position,
                self.msckf_state.imu_state.velocity
            )
            
            # Simple state transition
            Phi = np.eye(15)  # TODO: Compute proper transition matrix
            
            # Propagate state with adaptive noise
            self.msckf_state.propagate_imu(R_new, p_new, v_new, Phi, timestamp)
            
        # Add camera state
        self.msckf_state.augment_camera_state(timestamp)
        
        # Extract visual features
        if hasattr(self.visual_encoder, 'encode_image'):
            # For SEA-RAFT encoder
            visual_features = self.visual_encoder.encode_image(image, [frame_id])
            
            # Track features if we have correlation volume
            if hasattr(self.visual_encoder, 'get_correlation_volume'):
                corr_volume = self.visual_encoder.get_correlation_volume()
                tracked_features = self.feature_tracker.track_features(
                    corr_volume, timestamp, image.shape[-2:]
                )
                
                # MSCKF update with tracked features
                ready_features = self.feature_tracker.get_tracks_for_update()
                if len(ready_features) > 3:
                    # Update visual metrics for adaptive noise
                    mean_confidence = np.mean([f.confidence for f in tracked_features]) if tracked_features else 0.0
                    self.msckf_state.update_visual_metrics(len(ready_features), mean_confidence)
                    
                    # MSCKF update with Mahalanobis gating
                    dx, chi2 = self.msckf_update.update(ready_features, self.msckf_state.base)
                    
                    # Remove used features
                    used_ids = [f.id for f in ready_features]
                    self.feature_tracker.remove_features(used_ids)
                    
        # Mini bundle adjustment
        if self.mini_ba is not None and len(self.msckf_state.camera_states) >= 3:
            # Create BA problem
            ba_features = self.feature_tracker.get_tracks_for_update(min_length=2)
            if len(ba_features) > 5:
                frames, points = self.mini_ba.create_ba_problem(
                    self.msckf_state, ba_features
                )
                
                # Select subset and optimize
                sel_frames, sel_points = self.mini_ba.select_frames_and_points(
                    frames, points
                )
                
                if len(sel_frames) > 2 and len(sel_points) > 3:
                    opt_poses, opt_points, cost = self.mini_ba.optimize(
                        sel_frames, sel_points
                    )
                    
                    # Update state with optimized poses
                    # (simplified - in practice need careful update)
                    
        # Convert VIO state to torch
        vio_state = self.msckf_state.to_torch(image.device)
        
        # Extract pose estimate
        vio_rotation = vio_state['imu_rotation']  # [3, 3]
        vio_translation = vio_state['imu_position']  # [3]
        
        # Estimate uncertainty from covariance
        cov_diag = torch.diag(vio_state['covariance'])
        vio_uncertainty = torch.cat([
            cov_diag[0:3],  # Position uncertainty
            cov_diag[6:9]   # Rotation uncertainty
        ])
        
        return {
            'rotation': vio_rotation,
            'translation': vio_translation,
            'uncertainty': vio_uncertainty,
            'covariance': vio_state['covariance']
        }
        
    def forward(self, 
               images: torch.Tensor,
               imu_data: torch.Tensor,
               timestamps: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass combining VIO and transformer.
        
        Args:
            images: [B, T, C, H, W] image sequence
            imu_data: [B, T, K, 6] IMU measurements (K samples per frame)
            timestamps: [B, T] timestamps
            
        Returns:
            Dictionary with pose estimates and uncertainties
        """
        B, T, C, H, W = images.shape
        
        # Stage 1: VIO estimates
        vio_poses = []
        vio_uncertainties = []
        
        for t in range(T):
            # Process IMU up to frame t
            if t > 0:
                imu_slice = imu_data[:, t-1:t].cpu().numpy()  # IMU between frames
                for b in range(B):
                    imu_list = [(imu_slice[b, 0, k, :3], 
                                imu_slice[b, 0, k, 3:6], 
                                0.001) for k in range(imu_slice.shape[2])]
                    self.process_imu_measurements(imu_list)
                    
            # VIO update
            vio_result = self.vio_update(
                images[:, t], 
                float(t) if timestamps is None else timestamps[0, t].item()
            )
            
            vio_poses.append(torch.cat([
                vio_result['translation'],
                self.matrix_to_quaternion(vio_result['rotation'])
            ]))
            vio_uncertainties.append(vio_result['uncertainty'])
            
        vio_poses = torch.stack(vio_poses, dim=1)  # [B, T, 7]
        vio_uncertainties = torch.stack(vio_uncertainties, dim=1)  # [B, T, 6]
        
        if self.training_stage == 1:
            # VIO only - return relative poses between consecutive frames
            # Convert absolute poses to relative
            relative_poses = []
            for t in range(1, T):
                # Simple relative pose (this is a placeholder for VIO-only mode)
                # In practice, VIO computes these from IMU integration
                trans_rel = vio_poses[:, t, :3] - vio_poses[:, t-1, :3]
                quat_rel = vio_poses[:, t, 3:]  # Simplified - should be relative rotation
                rel_pose = torch.cat([trans_rel, quat_rel], dim=-1)
                relative_poses.append(rel_pose)
            
            relative_poses = torch.stack(relative_poses, dim=1)  # [B, T-1, 7]
            
            return {
                'poses': relative_poses,
                'uncertainties': vio_uncertainties[:, 1:, :],  # Skip first frame
                'vio_poses': vio_poses,
                'transformer_poses': None,
                'fusion_weights': None
            }
            
        # Stage 2/3: Transformer refinement
        # Extract features
        visual_features = self.visual_encoder(images.view(B*T, C, H, W))
        visual_features = visual_features.view(B, T, -1)
        
        imu_features = self.imu_encoder(imu_data)
        
        # Transformer prediction
        transformer_out = self.transformer(visual_features, imu_features)
        transformer_poses = transformer_out['poses']  # [B, T, 7]
        
        # Uncertainty estimation
        combined_features = torch.cat([visual_features, imu_features], dim=-1)
        transformer_uncertainty = self.uncertainty_estimator(combined_features)
        
        # Stage 3: Uncertainty-weighted fusion
        if self.training_stage >= 3:
            # Compute fusion weights based on uncertainties
            vio_confidence = 1.0 / (1.0 + vio_uncertainties)
            trans_confidence = 1.0 / (1.0 + transformer_uncertainty)
            
            # Normalize weights
            total_confidence = vio_confidence + trans_confidence
            vio_weight = vio_confidence / total_confidence
            trans_weight = trans_confidence / total_confidence
            
            # Weighted fusion
            fused_poses = vio_weight[..., :3] * vio_poses[..., :3] + \
                         trans_weight[..., :3] * transformer_poses[..., :3]
            
            # Quaternion fusion (SLERP)
            fused_quats = self.quaternion_slerp(
                vio_poses[..., 3:], 
                transformer_poses[..., 3:],
                trans_weight[..., 3:].mean(dim=-1, keepdim=True)
            )
            
            fused_poses = torch.cat([fused_poses, fused_quats], dim=-1)
            
            return {
                'poses': fused_poses,
                'uncertainties': torch.minimum(vio_uncertainties, transformer_uncertainty),
                'vio_poses': vio_poses,
                'transformer_poses': transformer_poses,
                'fusion_weights': torch.stack([vio_weight, trans_weight], dim=-1)
            }
        else:
            # Stage 2: Transformer only
            return {
                'poses': transformer_poses,
                'uncertainties': transformer_uncertainty,
                'vio_poses': vio_poses,
                'transformer_poses': transformer_poses,
                'fusion_weights': None
            }
            
    def matrix_to_quaternion(self, matrix: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to quaternion."""
        if matrix.dim() == 2:
            matrix = matrix.unsqueeze(0)
            
        # Based on "Converting a Rotation Matrix to a Quaternion" by Mike Day
        m00, m01, m02 = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
        m10, m11, m12 = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
        m20, m21, m22 = matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2]
        
        trace = m00 + m11 + m22
        
        if trace > 0:
            s = 0.5 / torch.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m21 - m12) * s
            y = (m02 - m20) * s
            z = (m10 - m01) * s
        elif m00 > m11 and m00 > m22:
            s = 2.0 * torch.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * torch.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * torch.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
            
        return torch.stack([x, y, z, w], dim=-1).squeeze(0)
        
    def quaternion_slerp(self, q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between quaternions."""
        # Normalize
        q1 = F.normalize(q1, dim=-1)
        q2 = F.normalize(q2, dim=-1)
        
        # Compute angle
        dot = (q1 * q2).sum(dim=-1, keepdim=True)
        
        # If negative dot, negate one quaternion
        mask = dot < 0
        q2 = torch.where(mask.unsqueeze(-1), -q2, q2)
        dot = torch.where(mask, -dot, dot)
        
        # Clamp
        dot = torch.clamp(dot, -1.0, 1.0)
        
        # Angle
        theta = torch.acos(dot)
        
        # SLERP
        sin_theta = torch.sin(theta)
        
        # Linear interpolation for small angles
        linear_mask = sin_theta.abs() < 1e-6
        
        s1 = torch.where(linear_mask, 
                        1.0 - t,
                        torch.sin((1.0 - t) * theta) / sin_theta)
        s2 = torch.where(linear_mask,
                        t,
                        torch.sin(t * theta) / sin_theta)
                        
        return s1 * q1 + s2 * q2