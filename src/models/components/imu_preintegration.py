"""
Pure Python IMU Pre-integration Module
Implements Forster et al.'s on-manifold pre-integration without GTSAM dependency.
Based on: "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry"
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from typing import Optional, Tuple, List, Dict


class IMUPreintegration:
    """
    Pure Python implementation of IMU pre-integration on SE(3) manifold.
    
    Key features:
    - On-manifold integration (SO(3) × R³)
    - Analytical bias correction via first-order approximation
    - Covariance propagation
    - PyTorch compatibility
    """
    
    def __init__(self, 
                 accel_noise_sigma: float = 0.0003924,      # m/s²/√Hz (from Aria datasheet)
                 gyro_noise_sigma: float = 0.000205689,     # rad/s/√Hz  
                 accel_bias_rw_sigma: float = 0.004905,     # m/s³/√Hz
                 gyro_bias_rw_sigma: float = 0.000001454,   # rad/s²/√Hz
                 gravity: np.ndarray = np.array([0, 0, -9.81])):  # m/s²
        """
        Initialize IMU pre-integration parameters.
        """
        # Noise parameters (continuous-time)
        self.sigma_a = accel_noise_sigma
        self.sigma_g = gyro_noise_sigma
        self.sigma_ba = accel_bias_rw_sigma
        self.sigma_bg = gyro_bias_rw_sigma
        
        # Gravity vector
        self.gravity = gravity
        
        # Initialize state
        self.reset()
        
    def reset(self, bias_acc: Optional[np.ndarray] = None, 
              bias_gyro: Optional[np.ndarray] = None):
        """
        Reset pre-integration with optional initial bias.
        """
        # Initial bias
        self.bias_acc_init = bias_acc if bias_acc is not None else np.zeros(3)
        self.bias_gyro_init = bias_gyro if bias_gyro is not None else np.zeros(3)
        
        # Pre-integrated values
        self.delta_R = np.eye(3)  # Rotation increment
        self.delta_v = np.zeros(3)  # Velocity increment
        self.delta_p = np.zeros(3)  # Position increment
        
        # Jacobians w.r.t. bias (for first-order correction)
        self.J_R_bg = np.zeros((3, 3))  # Rotation w.r.t. gyro bias
        self.J_v_ba = np.zeros((3, 3))  # Velocity w.r.t. accel bias
        self.J_v_bg = np.zeros((3, 3))  # Velocity w.r.t. gyro bias
        self.J_p_ba = np.zeros((3, 3))  # Position w.r.t. accel bias
        self.J_p_bg = np.zeros((3, 3))  # Position w.r.t. gyro bias
        
        # Covariance matrix (9x9: rotation, position, velocity)
        self.cov = np.zeros((9, 9))
        
        # Noise covariance (per unit time)
        self.Q = np.zeros((18, 18))  # [gyro_noise, accel_noise, gyro_bias_rw, accel_bias_rw]
        self.Q[0:3, 0:3] = np.eye(3) * (self.sigma_g ** 2)
        self.Q[3:6, 3:6] = np.eye(3) * (self.sigma_a ** 2)
        self.Q[6:9, 6:9] = np.eye(3) * (self.sigma_bg ** 2)
        self.Q[9:12, 9:12] = np.eye(3) * (self.sigma_ba ** 2)
        
        # Integration time
        self.delta_t = 0.0
        
        # Store measurements for debugging
        self.measurements = []
        
    def skew_symmetric(self, v: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix from vector."""
        return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
        
    def exp_SO3(self, phi: np.ndarray) -> np.ndarray:
        """Exponential map for SO(3)."""
        angle = np.linalg.norm(phi)
        if angle < 1e-8:
            return np.eye(3) + self.skew_symmetric(phi)
        
        axis = phi / angle
        K = self.skew_symmetric(axis)
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        
    def log_SO3(self, R: np.ndarray) -> np.ndarray:
        """Logarithm map for SO(3)."""
        trace = np.trace(R)
        if trace >= 3.0 - 1e-8:
            return np.zeros(3)
            
        angle = np.arccos((trace - 1) / 2)
        return angle / (2 * np.sin(angle)) * np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ])
        
    def right_jacobian_SO3(self, phi: np.ndarray) -> np.ndarray:
        """Right Jacobian of SO(3)."""
        angle = np.linalg.norm(phi)
        if angle < 1e-8:
            return np.eye(3) - 0.5 * self.skew_symmetric(phi)
            
        axis = phi / angle
        K = self.skew_symmetric(axis)
        return np.eye(3) - ((1 - np.cos(angle)) / angle) * K + ((angle - np.sin(angle)) / angle) * K @ K
        
    def integrate_measurement(self, acc: np.ndarray, gyro: np.ndarray, dt: float):
        """
        Integrate single IMU measurement using on-manifold integration.
        
        Args:
            acc: Accelerometer measurement [3] in m/s²
            gyro: Gyroscope measurement [3] in rad/s
            dt: Time delta in seconds
        """
        # Remove bias from measurements
        acc_unbiased = acc - self.bias_acc_init
        gyro_unbiased = gyro - self.bias_gyro_init
        
        # Store for debugging
        self.measurements.append((acc, gyro, dt))
        
        # Save previous values
        delta_R_prev = self.delta_R.copy()
        delta_v_prev = self.delta_v.copy()
        delta_p_prev = self.delta_p.copy()
        
        # Rotation update (exact integration)
        omega_dt = gyro_unbiased * dt
        dR = self.exp_SO3(omega_dt)
        self.delta_R = self.delta_R @ dR
        
        # Velocity and position updates (mid-point integration)
        acc_nav = delta_R_prev @ acc_unbiased  # Transform to navigation frame
        self.delta_v += acc_nav * dt
        self.delta_p += delta_v_prev * dt + 0.5 * acc_nav * dt * dt
        
        # Update Jacobians (first-order approximation)
        # See Forster et al. supplementary material for derivations
        
        # Helper matrices
        acc_skew = self.skew_symmetric(acc_unbiased)
        Jr = self.right_jacobian_SO3(omega_dt)
        
        # Jacobian updates
        A = np.eye(3) - self.skew_symmetric(omega_dt)
        B = -self.skew_symmetric(acc_nav) * dt
        C = -0.5 * self.skew_symmetric(acc_nav) * dt * dt
        
        # Update bias Jacobians
        self.J_R_bg = A @ self.J_R_bg - Jr * dt
        self.J_v_ba = self.J_v_ba + delta_R_prev * dt
        self.J_v_bg = self.J_v_bg + B @ self.J_R_bg
        self.J_p_ba = self.J_p_ba + self.J_v_ba * dt + 0.5 * delta_R_prev * dt * dt
        self.J_p_bg = self.J_p_bg + self.J_v_bg * dt + C @ self.J_R_bg
        
        # Covariance propagation
        self._propagate_covariance(A, B, C, delta_R_prev, dt)
        
        # Update total time
        self.delta_t += dt
        
    def _propagate_covariance(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, 
                             R_prev: np.ndarray, dt: float):
        """
        Propagate uncertainty through linearized dynamics.
        """
        # State transition matrix F (9x9)
        F = np.eye(9)
        F[0:3, 0:3] = A
        F[3:6, 0:3] = C
        F[3:6, 6:9] = np.eye(3) * dt
        F[6:9, 0:3] = B
        
        # Noise Jacobian G (9x12)
        G = np.zeros((9, 12))
        G[0:3, 0:3] = -np.eye(3) * dt  # Gyro noise → rotation
        G[6:9, 3:6] = R_prev * dt      # Accel noise → velocity
        G[3:6, 3:6] = 0.5 * R_prev * dt * dt  # Accel noise → position
        
        # Process noise (include both measurement noise and bias random walk)
        Q_d = self.Q[0:12, 0:12] * dt
        
        # Process noise already includes bias random walk in Q matrix
        
        # Propagate covariance
        self.cov = F @ self.cov @ F.T + G @ Q_d @ G.T
        
    def get_deltas(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get pre-integrated deltas.
        
        Returns:
            delta_p: Pre-integrated position change [3]
            delta_v: Pre-integrated velocity change [3]
            delta_R: Pre-integrated rotation matrix [3,3]
        """
        return self.delta_p.copy(), self.delta_v.copy(), self.delta_R.copy()
        
    def predict(self, R_i: np.ndarray, p_i: np.ndarray, v_i: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict next state using pre-integrated measurements.
        
        Args:
            R_i: Rotation matrix at time i [3,3]
            p_i: Position at time i [3]
            v_i: Velocity at time i [3]
            
        Returns:
            R_j: Predicted rotation at time j
            p_j: Predicted position at time j
            v_j: Predicted velocity at time j
        """
        # Apply pre-integrated measurements
        R_j = R_i @ self.delta_R
        v_j = v_i + R_i @ self.delta_v + self.gravity * self.delta_t
        p_j = p_i + v_i * self.delta_t + 0.5 * self.gravity * self.delta_t**2 + R_i @ self.delta_p
        
        return R_j, p_j, v_j
        
    def correct_bias(self, d_bias_acc: np.ndarray, d_bias_gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply first-order bias correction to pre-integrated values.
        
        Args:
            d_bias_acc: Accelerometer bias correction [3]
            d_bias_gyro: Gyroscope bias correction [3]
            
        Returns:
            Corrected (delta_p, delta_v, delta_R)
        """
        # Rotation correction - note the negative sign (we're removing bias effect)
        d_phi = -self.J_R_bg @ d_bias_gyro
        delta_R_corrected = self.delta_R @ self.exp_SO3(d_phi)
        
        # Velocity correction - subtract bias effects
        delta_v_corrected = self.delta_v - self.J_v_ba @ d_bias_acc - self.J_v_bg @ d_bias_gyro
        
        # Position correction - subtract bias effects
        delta_p_corrected = self.delta_p - self.J_p_ba @ d_bias_acc - self.J_p_bg @ d_bias_gyro
        
        return delta_p_corrected, delta_v_corrected, delta_R_corrected
        
    def get_covariance(self) -> np.ndarray:
        """Get 9x9 covariance matrix for (rotation, position, velocity)."""
        return self.cov.copy()
        
    def to_torch(self, device='cuda') -> Dict[str, torch.Tensor]:
        """
        Convert pre-integrated values to PyTorch tensors.
        """
        delta_p, delta_v, delta_R = self.get_deltas()
        
        return {
            'delta_p': torch.from_numpy(delta_p).float().to(device),
            'delta_v': torch.from_numpy(delta_v).float().to(device),
            'delta_R': torch.from_numpy(delta_R).float().to(device),
            'delta_t': torch.tensor(self.delta_t).float().to(device),
            'covariance': torch.from_numpy(self.cov).float().to(device),
            # Bias Jacobians for optimization
            'J_p_ba': torch.from_numpy(self.J_p_ba).float().to(device),
            'J_p_bg': torch.from_numpy(self.J_p_bg).float().to(device),
            'J_v_ba': torch.from_numpy(self.J_v_ba).float().to(device),
            'J_v_bg': torch.from_numpy(self.J_v_bg).float().to(device),
            'J_R_bg': torch.from_numpy(self.J_R_bg).float().to(device),
        }
        
    def get_information_matrix(self) -> np.ndarray:
        """Get information matrix (inverse covariance) for optimization."""
        # Add small diagonal to ensure positive definite
        cov_reg = self.cov + np.eye(9) * 1e-9
        return np.linalg.inv(cov_reg)


class IMUPreintegrationBuffer:
    """
    Manages multiple pre-integration instances for sliding window.
    Thread-safe buffer for real-time operation.
    """
    
    def __init__(self, window_size: int = 10, **kwargs):
        """
        Initialize buffer for sliding window pre-integration.
        
        Args:
            window_size: Number of frames in sliding window
            **kwargs: Parameters passed to IMUPreintegration
        """
        self.window_size = window_size
        self.params = kwargs
        
        # Pre-integration between consecutive frames
        self.preintegrators = [
            IMUPreintegration(**kwargs) 
            for _ in range(window_size)
        ]
        
        # Frame associations
        self.frame_ids = [-1] * window_size
        self.current_frame_id = 0
        
        # Current active index
        self.current_idx = 0
        
        # Thread safety
        import threading
        self.lock = threading.Lock()
        
    def add_measurement(self, acc: np.ndarray, gyro: np.ndarray, dt: float):
        """
        Add measurement to current pre-integrator.
        Thread-safe.
        """
        with self.lock:
            if self.current_idx < len(self.preintegrators):
                self.preintegrators[self.current_idx].integrate_measurement(acc, gyro, dt)
                
    def new_keyframe(self, frame_id: Optional[int] = None):
        """
        Signal new keyframe, move to next pre-integrator.
        Returns the completed pre-integration.
        """
        with self.lock:
            # Get completed pre-integration
            completed = self.preintegrators[self.current_idx]
            
            # Move to next slot
            self.current_idx = (self.current_idx + 1) % self.window_size
            
            # Reset the pre-integrator for new measurements
            self.preintegrators[self.current_idx].reset()
            
            # Update frame ID
            if frame_id is None:
                frame_id = self.current_frame_id
                self.current_frame_id += 1
            self.frame_ids[self.current_idx] = frame_id
            
            return completed
            
    def get_current(self) -> IMUPreintegration:
        """Get current active pre-integrator."""
        with self.lock:
            return self.preintegrators[self.current_idx]
            
    def get_window_factors(self) -> List[Tuple[int, int, IMUPreintegration]]:
        """
        Get all valid pre-integrations in current window.
        
        Returns:
            List of (frame_i, frame_j, preintegration) tuples
        """
        with self.lock:
            factors = []
            
            for i in range(self.window_size):
                if self.frame_ids[i] >= 0 and self.preintegrators[i].delta_t > 0:
                    frame_i = self.frame_ids[i]
                    frame_j = self.frame_ids[(i + 1) % self.window_size]
                    
                    if frame_j > frame_i:  # Valid consecutive frames
                        factors.append((frame_i, frame_j, self.preintegrators[i]))
                        
            return sorted(factors, key=lambda x: x[0])  # Sort by frame_i