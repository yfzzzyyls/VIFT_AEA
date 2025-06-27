"""
MSCKF State Management Module
Implements sliding window state for Multi-State Constraint Kalman Filter.
Based on: "A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation"
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Tuple, Optional
import threading
from dataclasses import dataclass


@dataclass
class IMUState:
    """IMU state at a specific time."""
    timestamp: float
    position: np.ndarray      # [3] in world frame
    velocity: np.ndarray      # [3] in world frame  
    rotation: np.ndarray      # [3,3] rotation matrix (body to world)
    bias_acc: np.ndarray      # [3] accelerometer bias
    bias_gyro: np.ndarray     # [3] gyroscope bias
    
    def to_vector(self) -> np.ndarray:
        """Convert to state vector [position, velocity, rotation_vec, bias_acc, bias_gyro]."""
        rot_vec = R.from_matrix(self.rotation).as_rotvec()
        return np.concatenate([
            self.position, self.velocity, rot_vec, 
            self.bias_acc, self.bias_gyro
        ])
        
    @staticmethod
    def from_vector(x: np.ndarray, timestamp: float) -> 'IMUState':
        """Create from state vector."""
        return IMUState(
            timestamp=timestamp,
            position=x[0:3],
            velocity=x[3:6],
            rotation=R.from_rotvec(x[6:9]).as_matrix(),
            bias_acc=x[9:12],
            bias_gyro=x[12:15]
        )


class MSCKFState:
    """
    Fixed-size sliding window state for MSCKF.
    
    State vector layout:
    - IMU state: [position(3), velocity(3), rotation(3), bias_acc(3), bias_gyro(3)] = 15
    - Camera states: N × [position(3), rotation(3)] = N × 6
    
    Total state dimension: 15 + N × 6
    
    Adapted from OpenVINS StateHelper architecture.
    """
    
    def __init__(self, window_size: int = 10, 
                 noise_acc: float = 0.0003924,
                 noise_gyro: float = 0.000205689,
                 noise_bias_acc: float = 0.004905,
                 noise_bias_gyro: float = 0.000001454):
        """
        Initialize MSCKF state manager.
        
        Args:
            window_size: Number of camera poses in sliding window
            noise_*: IMU noise parameters for covariance initialization
        """
        self.window_size = window_size
        
        # State dimensions
        self.imu_state_dim = 15  # pos(3) + vel(3) + rot(3) + biases(6)
        self.cam_state_dim = 6   # pos(3) + rot(3)
        self.state_dim = self.imu_state_dim + window_size * self.cam_state_dim
        
        # Current IMU state
        self.imu_state = IMUState(
            timestamp=0.0,
            position=np.zeros(3),
            velocity=np.zeros(3),
            rotation=np.eye(3),
            bias_acc=np.zeros(3),
            bias_gyro=np.zeros(3)
        )
        
        # Camera states in sliding window
        self.camera_states: Dict[float, IMUState] = {}  # timestamp -> state
        self.camera_timestamps: List[float] = []  # Ordered list
        
        # State covariance (kept in compact form)
        self.P = np.eye(self.state_dim) * 1e-3
        
        # Initialize IMU covariance
        self.P[0:3, 0:3] *= 1e-2    # Position
        self.P[3:6, 3:6] *= 1e-2    # Velocity
        self.P[6:9, 6:9] *= 1e-3    # Rotation
        self.P[9:12, 9:12] *= (noise_bias_acc ** 2)    # Accel bias
        self.P[12:15, 12:15] *= (noise_bias_gyro ** 2) # Gyro bias
        
        # Noise parameters
        self.Q_imu = np.zeros((12, 12))
        self.Q_imu[0:3, 0:3] = np.eye(3) * (noise_acc ** 2)      # Accel noise
        self.Q_imu[3:6, 3:6] = np.eye(3) * (noise_gyro ** 2)     # Gyro noise
        self.Q_imu[6:9, 6:9] = np.eye(3) * (noise_bias_acc ** 2) # Accel bias walk
        self.Q_imu[9:12, 9:12] = np.eye(3) * (noise_bias_gyro ** 2) # Gyro bias walk
        
        # First-Estimate Jacobian storage for consistency
        self.H_fej: Dict[float, np.ndarray] = {}
        
        # Thread safety
        self.lock = threading.Lock()
        
    def get_state_vector(self) -> np.ndarray:
        """Get full state vector [imu_state, camera_states]."""
        with self.lock:
            x = np.zeros(self.state_dim)
            
            # IMU state
            x[0:self.imu_state_dim] = self.imu_state.to_vector()
            
            # Camera states
            for i, timestamp in enumerate(self.camera_timestamps):
                start_idx = self.imu_state_dim + i * self.cam_state_dim
                cam_state = self.camera_states[timestamp]
                rot_vec = R.from_matrix(cam_state.rotation).as_rotvec()
                x[start_idx:start_idx+3] = cam_state.position
                x[start_idx+3:start_idx+6] = rot_vec
                
            return x
            
    def set_state_vector(self, x: np.ndarray):
        """Update state from vector."""
        with self.lock:
            # Update IMU state
            self.imu_state = IMUState.from_vector(x[0:self.imu_state_dim], 
                                                 self.imu_state.timestamp)
            
            # Update camera states
            for i, timestamp in enumerate(self.camera_timestamps):
                start_idx = self.imu_state_dim + i * self.cam_state_dim
                pos = x[start_idx:start_idx+3]
                rot = R.from_rotvec(x[start_idx+3:start_idx+6]).as_matrix()
                
                self.camera_states[timestamp].position = pos
                self.camera_states[timestamp].rotation = rot
                
    def propagate_imu(self, R_new: np.ndarray, p_new: np.ndarray, 
                     v_new: np.ndarray, Phi: np.ndarray, Q: np.ndarray,
                     timestamp: float):
        """
        Propagate IMU state and covariance.
        
        Args:
            R_new: New rotation matrix
            p_new: New position
            v_new: New velocity
            Phi: State transition matrix (15x15)
            Q: Process noise covariance (15x15)
            timestamp: New timestamp
        """
        with self.lock:
            # Update IMU state
            self.imu_state.rotation = R_new
            self.imu_state.position = p_new
            self.imu_state.velocity = v_new
            self.imu_state.timestamp = timestamp
            
            # Propagate covariance
            # P_new = Phi @ P @ Phi.T + Q
            P_imu = self.P[0:15, 0:15]
            P_imu_new = Phi @ P_imu @ Phi.T + Q
            
            # Update IMU-IMU block
            self.P[0:15, 0:15] = P_imu_new
            
            # Update IMU-Camera blocks
            for i in range(len(self.camera_timestamps)):
                cam_idx = 15 + i * 6
                self.P[0:15, cam_idx:cam_idx+6] = Phi @ self.P[0:15, cam_idx:cam_idx+6]
                self.P[cam_idx:cam_idx+6, 0:15] = self.P[cam_idx:cam_idx+6, 0:15] @ Phi.T
                
    def augment_camera_state(self, timestamp: float):
        """
        Add current IMU pose to sliding window as new camera state.
        
        Args:
            timestamp: Timestamp for new camera state
        """
        with self.lock:
            # Clone current IMU state
            new_cam_state = IMUState(
                timestamp=timestamp,
                position=self.imu_state.position.copy(),
                velocity=self.imu_state.velocity.copy(),
                rotation=self.imu_state.rotation.copy(),
                bias_acc=self.imu_state.bias_acc.copy(),
                bias_gyro=self.imu_state.bias_gyro.copy()
            )
            
            # Check if we need to marginalize oldest state
            if len(self.camera_states) >= self.window_size:
                self.marginalize_oldest()
                
            # Add to window
            self.camera_states[timestamp] = new_cam_state
            self.camera_timestamps.append(timestamp)
            
            # Augment covariance
            self._augment_covariance()
            
    def _augment_covariance(self):
        """
        Augment covariance matrix for new camera state.
        Uses error state formulation.
        """
        # Jacobian of new camera state w.r.t IMU state
        J = np.zeros((6, 15))
        J[0:3, 0:3] = np.eye(3)  # Position
        J[3:6, 6:9] = np.eye(3)  # Rotation
        
        # Get current covariance size
        n = 15 + (len(self.camera_timestamps) - 1) * 6
        
        # Safety check
        if n > self.P.shape[0]:
            return
            
        # Create augmented covariance
        P_aug = np.zeros((n + 6, n + 6))
        P_aug[0:n, 0:n] = self.P[0:n, 0:n]
        
        # New camera covariance
        P_aug[n:n+6, 0:15] = J @ self.P[0:15, 0:15]
        P_aug[0:15, n:n+6] = P_aug[n:n+6, 0:15].T
        P_aug[n:n+6, n:n+6] = J @ self.P[0:15, 0:15] @ J.T
        
        # Cross covariances with other cameras
        for i in range(len(self.camera_timestamps) - 1):
            cam_idx = 15 + i * 6
            if cam_idx + 6 <= n:  # Safety check
                P_aug[n:n+6, cam_idx:cam_idx+6] = J @ self.P[0:15, cam_idx:cam_idx+6]
                P_aug[cam_idx:cam_idx+6, n:n+6] = P_aug[n:n+6, cam_idx:cam_idx+6].T
            
        self.P = P_aug
        
    def marginalize_oldest(self):
        """
        Marginalize oldest camera state using Schur complement.
        Ensures consistent marginalization via First-Estimate Jacobian.
        """
        if not self.camera_timestamps:
            return
            
        oldest_timestamp = self.camera_timestamps[0]
        
        # Get indices
        marg_idx = 15  # First camera state
        keep_idx = list(range(15)) + list(range(21, self.P.shape[0]))
        
        # Schur complement
        P_mm = self.P[marg_idx:marg_idx+6, marg_idx:marg_idx+6]
        P_mr = self.P[marg_idx:marg_idx+6, keep_idx]
        P_rm = self.P[keep_idx, marg_idx:marg_idx+6]
        P_rr = self.P[np.ix_(keep_idx, keep_idx)]
        
        # Check if invertible
        if np.linalg.cond(P_mm) < 1e12:
            P_mm_inv = np.linalg.inv(P_mm + np.eye(6) * 1e-9)
            P_marginalized = P_rr - P_rm @ P_mm_inv @ P_mr
        else:
            # Fall back to simple removal
            P_marginalized = P_rr
            
        # Update covariance
        new_size = len(keep_idx)
        self.P = np.zeros((new_size, new_size))
        self.P[:, :] = P_marginalized
        
        # Remove from state
        del self.camera_states[oldest_timestamp]
        self.camera_timestamps.pop(0)
        
    def get_cam_state_indices(self, timestamp: float) -> Tuple[int, int]:
        """
        Get state vector indices for camera state at given timestamp.
        
        Returns:
            (start_idx, end_idx) for camera state in state vector
        """
        with self.lock:
            if timestamp not in self.camera_states:
                raise ValueError(f"Timestamp {timestamp} not in sliding window")
                
            cam_idx = self.camera_timestamps.index(timestamp)
            start_idx = self.imu_state_dim + cam_idx * self.cam_state_dim
            end_idx = start_idx + self.cam_state_dim
            
            return start_idx, end_idx
            
    def get_feature_jacobian(self, cam_timestamps: List[float], 
                           feature_pos: np.ndarray) -> np.ndarray:
        """
        Compute measurement Jacobian for a feature observed from multiple cameras.
        
        Args:
            cam_timestamps: List of camera timestamps that observed the feature
            feature_pos: 3D feature position in world frame
            
        Returns:
            Jacobian matrix H of size [2*M, state_dim] where M = len(cam_timestamps)
        """
        with self.lock:
            M = len(cam_timestamps)
            H = np.zeros((2 * M, self.state_dim))
            
            for i, timestamp in enumerate(cam_timestamps):
                if timestamp not in self.camera_states:
                    continue
                    
                cam_state = self.camera_states[timestamp]
                start_idx, end_idx = self.get_cam_state_indices(timestamp)
                
                # Transform feature to camera frame
                p_c = cam_state.rotation.T @ (feature_pos - cam_state.position)
                
                # Check if behind camera
                if p_c[2] <= 0.1:  # 10cm minimum depth
                    continue
                    
                # Projection Jacobian
                x, y, z = p_c
                H_proj = np.array([
                    [1/z, 0, -x/z**2],
                    [0, 1/z, -y/z**2]
                ])
                
                # Jacobian w.r.t camera pose
                H_pos = -H_proj @ cam_state.rotation.T
                H_rot = H_proj @ self._skew_symmetric(p_c)
                
                # Fill in Jacobian
                row_start = 2 * i
                H[row_start:row_start+2, start_idx:start_idx+3] = H_pos
                H[row_start:row_start+2, start_idx+3:end_idx] = H_rot
                
            return H
            
    def _skew_symmetric(self, v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix."""
        return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
                        
    def clone_state(self) -> 'MSCKFState':
        """Create a deep copy of current state."""
        with self.lock:
            new_state = MSCKFState(self.window_size)
            
            # Copy IMU state
            new_state.imu_state = IMUState(
                timestamp=self.imu_state.timestamp,
                position=self.imu_state.position.copy(),
                velocity=self.imu_state.velocity.copy(),
                rotation=self.imu_state.rotation.copy(),
                bias_acc=self.imu_state.bias_acc.copy(),
                bias_gyro=self.imu_state.bias_gyro.copy()
            )
            
            # Copy camera states
            for ts, cam_state in self.camera_states.items():
                new_state.camera_states[ts] = IMUState(
                    timestamp=cam_state.timestamp,
                    position=cam_state.position.copy(),
                    velocity=cam_state.velocity.copy(),
                    rotation=cam_state.rotation.copy(),
                    bias_acc=cam_state.bias_acc.copy(),
                    bias_gyro=cam_state.bias_gyro.copy()
                )
            new_state.camera_timestamps = self.camera_timestamps.copy()
            
            # Copy covariance
            new_state.P = self.P.copy()
            
            return new_state
            
    def to_torch(self, device='cuda') -> Dict[str, torch.Tensor]:
        """Convert state to PyTorch tensors."""
        with self.lock:
            return {
                'state_vector': torch.from_numpy(self.get_state_vector()).float().to(device),
                'covariance': torch.from_numpy(self.P).float().to(device),
                'imu_position': torch.from_numpy(self.imu_state.position).float().to(device),
                'imu_velocity': torch.from_numpy(self.imu_state.velocity).float().to(device),
                'imu_rotation': torch.from_numpy(self.imu_state.rotation).float().to(device),
                'imu_bias_acc': torch.from_numpy(self.imu_state.bias_acc).float().to(device),
                'imu_bias_gyro': torch.from_numpy(self.imu_state.bias_gyro).float().to(device),
                'window_size': torch.tensor(len(self.camera_states)).to(device),
            }