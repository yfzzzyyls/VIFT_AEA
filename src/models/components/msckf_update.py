"""
MSCKF Measurement Update Module
Implements sparse QR decomposition for null-space projection.
Based on: "A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation"
"""

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from .msckf_state import MSCKFState
from .feature_tracker_msckf import Feature


@dataclass
class MSCKFMeasurement:
    """Single feature measurement for MSCKF update."""
    feature_id: int
    camera_timestamps: List[float]
    pixel_observations: List[np.ndarray]  # List of [u, v] observations
    feature_position: Optional[np.ndarray] = None  # Triangulated 3D position


class MSCKFUpdate:
    """
    MSCKF measurement update using null-space projection.
    
    Key features:
    - Sparse QR decomposition for efficiency
    - Null-space projection to eliminate feature states
    - Chi-squared gating for outlier rejection
    - Mahalanobis distance for consistency check
    """
    
    def __init__(self,
                 camera_matrix: np.ndarray,
                 chi2_threshold: float = 5.991,  # 95% for 2 DOF
                 min_baseline: float = 0.1,      # Minimum baseline for triangulation
                 max_reprojection_error: float = 2.0,  # Pixels
                 use_sparse_qr: bool = True):
        """
        Initialize MSCKF update module.
        
        Args:
            camera_matrix: Camera intrinsic matrix [3x3]
            chi2_threshold: Chi-squared threshold for measurement gating
            min_baseline: Minimum baseline for triangulation (meters)
            max_reprojection_error: Maximum reprojection error (pixels)
            use_sparse_qr: Use sparse QR for efficiency
        """
        self.K = camera_matrix
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]
        
        self.chi2_threshold = chi2_threshold
        self.min_baseline = min_baseline
        self.max_reprojection_error = max_reprojection_error
        self.use_sparse_qr = use_sparse_qr
        
        # Measurement noise (pixels)
        self.R_pixel = np.eye(2) * (0.5 ** 2)  # 0.5 pixel std
        
    def process_feature_tracks(self, 
                             features: List[Feature],
                             state: MSCKFState) -> List[MSCKFMeasurement]:
        """
        Process feature tracks into measurements.
        
        Args:
            features: List of tracked features
            state: Current MSCKF state
            
        Returns:
            List of measurements ready for update
        """
        measurements = []
        
        for feature in features:
            # Get observations in camera states
            obs_in_window = []
            timestamps_in_window = []
            
            for timestamp, pixel_obs in feature.observations.items():
                if timestamp in state.camera_states:
                    obs_in_window.append(pixel_obs)
                    timestamps_in_window.append(timestamp)
                    
            # Need at least 2 observations for triangulation
            if len(obs_in_window) >= 2:
                measurement = MSCKFMeasurement(
                    feature_id=feature.id,
                    camera_timestamps=timestamps_in_window,
                    pixel_observations=obs_in_window
                )
                
                # Triangulate 3D position
                success, position_3d = self._triangulate_feature(
                    measurement, state
                )
                
                if success:
                    measurement.feature_position = position_3d
                    measurements.append(measurement)
                    
        return measurements
        
    def _triangulate_feature(self, 
                           measurement: MSCKFMeasurement,
                           state: MSCKFState) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Triangulate 3D feature position from multiple observations.
        
        Uses linear triangulation with SVD.
        
        Returns:
            (success, position_3d)
        """
        if len(measurement.camera_timestamps) < 2:
            return False, None
            
        # Build linear system A * X = 0
        A_list = []
        
        for i, timestamp in enumerate(measurement.camera_timestamps):
            if timestamp not in state.camera_states:
                continue
                
            cam_state = state.camera_states[timestamp]
            R_cw = cam_state.rotation.T  # Camera to world rotation
            t_cw = -R_cw @ cam_state.position  # Camera to world translation
            
            # Projection matrix P = K[R|t]
            P = self.K @ np.hstack([R_cw, t_cw.reshape(-1, 1)])
            
            # Pixel observation
            u, v = measurement.pixel_observations[i]
            
            # Build equations: x * P[2,:] - P[0,:] = 0, y * P[2,:] - P[1,:] = 0
            A_list.append(u * P[2, :] - P[0, :])
            A_list.append(v * P[2, :] - P[1, :])
            
        if len(A_list) < 4:
            return False, None
            
        A = np.vstack(A_list)
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X_homo = Vt[-1, :]  # Last row of V (smallest singular value)
        
        # Convert from homogeneous
        if abs(X_homo[3]) < 1e-10:
            return False, None
            
        position_3d = X_homo[:3] / X_homo[3]
        
        # Sanity checks
        if np.linalg.norm(position_3d) > 100.0:  # Too far
            return False, None
            
        # Check baseline
        cam_positions = []
        for timestamp in measurement.camera_timestamps[:2]:
            if timestamp in state.camera_states:
                cam_positions.append(state.camera_states[timestamp].position)
                
        if len(cam_positions) >= 2:
            baseline = np.linalg.norm(cam_positions[1] - cam_positions[0])
            if baseline < self.min_baseline:
                return False, None
                
        # Check reprojection error
        avg_error = self._compute_reprojection_error(
            position_3d, measurement, state
        )
        
        if avg_error > self.max_reprojection_error:
            return False, None
            
        return True, position_3d
        
    def _compute_reprojection_error(self,
                                   position_3d: np.ndarray,
                                   measurement: MSCKFMeasurement,
                                   state: MSCKFState) -> float:
        """
        Compute average reprojection error.
        
        Returns:
            Average reprojection error in pixels
        """
        errors = []
        
        for i, timestamp in enumerate(measurement.camera_timestamps):
            if timestamp not in state.camera_states:
                continue
                
            cam_state = state.camera_states[timestamp]
            
            # Transform to camera frame
            p_c = cam_state.rotation.T @ (position_3d - cam_state.position)
            
            if p_c[2] <= 0.1:  # Behind camera
                return float('inf')
                
            # Project
            u_pred = self.fx * p_c[0] / p_c[2] + self.cx
            v_pred = self.fy * p_c[1] / p_c[2] + self.cy
            
            # Compare with observation
            u_obs, v_obs = measurement.pixel_observations[i]
            error = np.sqrt((u_pred - u_obs)**2 + (v_pred - v_obs)**2)
            errors.append(error)
            
        return np.mean(errors) if errors else float('inf')
        
    def compute_residual_and_jacobian(self,
                                    measurements: List[MSCKFMeasurement],
                                    state: MSCKFState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute stacked residual vector and Jacobian matrices.
        
        Args:
            measurements: List of feature measurements
            state: Current MSCKF state
            
        Returns:
            r: Residual vector
            H_x: Jacobian w.r.t. state
            H_f: Jacobian w.r.t. features
        """
        residuals = []
        H_x_blocks = []
        H_f_blocks = []
        
        for j, meas in enumerate(measurements):
            if meas.feature_position is None:
                continue
                
            # Residual and Jacobians for this feature
            r_j = []
            H_x_j = []
            
            for i, timestamp in enumerate(meas.camera_timestamps):
                if timestamp not in state.camera_states:
                    continue
                    
                cam_state = state.camera_states[timestamp]
                cam_idx_start, cam_idx_end = state.get_cam_state_indices(timestamp)
                
                # Transform feature to camera frame
                p_w = meas.feature_position
                R_cw = cam_state.rotation.T
                t_cw = cam_state.position
                p_c = R_cw @ (p_w - t_cw)
                
                if p_c[2] <= 0.1:  # Behind camera
                    continue
                    
                # Predicted pixel coordinates
                u_pred = self.fx * p_c[0] / p_c[2] + self.cx
                v_pred = self.fy * p_c[1] / p_c[2] + self.cy
                
                # Observed pixel coordinates
                u_obs, v_obs = meas.pixel_observations[i]
                
                # Residual
                r_i = np.array([u_obs - u_pred, v_obs - v_pred])
                r_j.append(r_i)
                
                # Jacobian of projection w.r.t. point in camera frame
                x, y, z = p_c
                J_proj = np.array([
                    [self.fx/z, 0, -self.fx*x/z**2],
                    [0, self.fy/z, -self.fy*y/z**2]
                ])
                
                # Jacobian w.r.t. camera pose
                H_pos = J_proj @ R_cw
                H_rot = J_proj @ self._skew_symmetric(p_c) @ R_cw
                
                # Fill state Jacobian
                H_x_i = np.zeros((2, state.state_dim))
                H_x_i[:, cam_idx_start:cam_idx_start+3] = H_pos
                H_x_i[:, cam_idx_start+3:cam_idx_end] = H_rot
                H_x_j.append(H_x_i)
                
            if r_j:
                # Stack for this feature
                residuals.extend(r_j)
                H_x_blocks.extend(H_x_j)
                
                # Feature Jacobian
                n_obs = len(r_j)
                H_f_j = []
                for i in range(n_obs):
                    # Each observation depends on the 3D feature position
                    cam_state = state.camera_states[meas.camera_timestamps[i]]
                    R_cw = cam_state.rotation.T
                    p_c = R_cw @ (meas.feature_position - cam_state.position)
                    
                    # Jacobian of residual w.r.t. feature position
                    x, y, z = p_c
                    J_proj = np.array([
                        [self.fx/z, 0, -self.fx*x/z**2],
                        [0, self.fy/z, -self.fy*y/z**2]
                    ])
                    H_f_i = -J_proj @ R_cw  # Negative because r = obs - pred
                    H_f_j.append(H_f_i)
                    
                H_f_blocks.append(np.vstack(H_f_j))
                
        if not residuals:
            return np.array([]), np.array([]), np.array([])
            
        # Stack all residuals and Jacobians
        r = np.hstack(residuals)
        H_x = np.vstack(H_x_blocks)
        
        # Create block diagonal H_f
        if H_f_blocks:
            H_f = sparse.block_diag(H_f_blocks)
            if not self.use_sparse_qr:
                H_f = H_f.toarray()
        else:
            H_f = np.array([])
            
        return r, H_x, H_f
        
    def nullspace_project(self, 
                         r: np.ndarray,
                         H_x: np.ndarray, 
                         H_f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project residual and Jacobian to null space of H_f.
        
        Uses QR decomposition: H_f = Q * R
        Then: N = I - Q * Q.T (null space projector)
        
        Args:
            r: Residual vector
            H_x: Jacobian w.r.t. state
            H_f: Jacobian w.r.t. features
            
        Returns:
            r_o: Projected residual
            H_o: Projected state Jacobian
        """
        if H_f.size == 0:
            return r, H_x
            
        m, n = H_f.shape if hasattr(H_f, 'shape') else (H_f.shape[0], H_f.shape[1])
        
        if self.use_sparse_qr and sparse.issparse(H_f):
            # Sparse QR decomposition
            # Note: scipy's sparse QR returns Q as a LinearOperator
            # We'll use economy QR for efficiency
            Q, R = sparse_linalg.qr(H_f, mode='economic')
            
            # Project using Q
            # r_o = r - Q @ (Q.T @ r)
            # H_o = H_x - Q @ (Q.T @ H_x)
            Qt_r = Q.T @ r
            Qt_Hx = Q.T @ H_x
            
            r_o = r - Q @ Qt_r
            H_o = H_x - Q @ Qt_Hx
            
        else:
            # Dense QR decomposition
            if sparse.issparse(H_f):
                H_f = H_f.toarray()
                
            Q, R = np.linalg.qr(H_f, mode='reduced')
            
            # Null space projection
            r_o = r - Q @ (Q.T @ r)
            H_o = H_x - Q @ (Q.T @ H_x)
            
        return r_o, H_o
        
    def compute_kalman_gain(self,
                          H: np.ndarray,
                          P: np.ndarray,
                          R: np.ndarray) -> np.ndarray:
        """
        Compute Kalman gain.
        
        K = P * H.T * (H * P * H.T + R)^-1
        
        Args:
            H: Measurement Jacobian
            P: State covariance
            R: Measurement noise covariance
            
        Returns:
            K: Kalman gain
        """
        # Innovation covariance
        S = H @ P @ H.T + R
        
        # Add small diagonal for numerical stability
        S += np.eye(S.shape[0]) * 1e-9
        
        # Kalman gain
        try:
            K = P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            K = P @ H.T @ np.linalg.pinv(S)
            
        return K
        
    def update(self,
              features: List[Feature],
              state: MSCKFState) -> Tuple[np.ndarray, float]:
        """
        Perform MSCKF measurement update.
        
        Args:
            features: List of features to use for update
            state: Current MSCKF state (will be modified)
            
        Returns:
            state_correction: Correction to state vector
            chi2_statistic: Chi-squared test statistic
        """
        # Process features into measurements
        measurements = self.process_feature_tracks(features, state)
        
        if not measurements:
            return np.zeros(state.state_dim), 0.0
            
        # Compute residual and Jacobians
        r, H_x, H_f = self.compute_residual_and_jacobian(measurements, state)
        
        if r.size == 0:
            return np.zeros(state.state_dim), 0.0
            
        # Null space projection
        r_o, H_o = self.nullspace_project(r, H_x, H_f)
        
        # Measurement noise after projection
        n_measurements = r_o.size
        R_o = np.eye(n_measurements) * (self.R_pixel[0, 0])
        
        # Compute Kalman gain
        K = self.compute_kalman_gain(H_o, state.P, R_o)
        
        # Mahalanobis gating (chi-squared test BEFORE update)
        # Compute innovation covariance
        S = H_o @ state.P @ H_o.T + R_o
        
        # Mahalanobis distance
        try:
            S_inv = np.linalg.inv(S)
            mahalanobis_dist = r_o.T @ S_inv @ r_o
        except:
            mahalanobis_dist = float('inf')
            
        # Chi-squared threshold for given confidence level
        # For 95% confidence: chi2.ppf(0.95, df=n_measurements)
        from scipy.stats import chi2 as chi2_dist
        chi2_threshold = chi2_dist.ppf(0.95, df=n_measurements)
        
        # Gate check
        if mahalanobis_dist > chi2_threshold:
            # Reject this update as outlier
            print(f"MSCKF update rejected: Mahalanobis distance {mahalanobis_dist:.2f} > threshold {chi2_threshold:.2f}")
            return np.zeros(state.state_dim), mahalanobis_dist
        
        # State correction
        dx = K @ r_o
        
        # Update state
        x = state.get_state_vector()
        x_new = x + dx
        state.set_state_vector(x_new)
        
        # Update covariance
        # P = (I - K*H) * P * (I - K*H).T + K*R*K.T (Joseph form)
        I_KH = np.eye(state.state_dim) - K @ H_o
        state.P = I_KH @ state.P @ I_KH.T + K @ R_o @ K.T
        
        # Return the Mahalanobis distance as chi2 statistic
        return dx, mahalanobis_dist
        
    def _skew_symmetric(self, v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix."""
        return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])