"""
IMU Pre-integration Module using GTSAM
Implements Forster et al.'s on-manifold pre-integration for robust VIO.
"""

import numpy as np
import torch
import gtsam
from typing import Optional, Tuple, List


class GTSAMPreintegration:
    """
    Wrapper around GTSAM's battle-tested IMU preintegration.
    Provides Python interface to C++ implementation with proven SE(3) manifold integration.
    
    Key features:
    - On-manifold integration (SO(3) × R³)
    - Analytical bias correction
    - Covariance propagation
    - Constant-time factors for optimization
    """
    
    def __init__(self, 
                 accel_noise_sigma: float = 0.0003924,      # m/s²/√Hz (from Aria datasheet)
                 gyro_noise_sigma: float = 0.000205689,     # rad/s/√Hz  
                 accel_bias_rw_sigma: float = 0.004905,     # m/s³/√Hz
                 gyro_bias_rw_sigma: float = 0.000001454,   # rad/s²/√Hz
                 gravity_magnitude: float = 9.81,           # m/s²
                 use_2nd_order_coriolis: bool = False):
        """
        Initialize IMU pre-integration parameters.
        
        Default values based on Aria IMU characteristics.
        """
        # Create GTSAM IMU parameters - use Combined params for bias support
        self.params = gtsam.PreintegrationCombinedParams.MakeSharedU(gravity_magnitude)
        
        # Set noise model (continuous-time)
        self.params.setAccelerometerCovariance(np.eye(3) * (accel_noise_sigma ** 2))
        self.params.setGyroscopeCovariance(np.eye(3) * (gyro_noise_sigma ** 2))
        self.params.setIntegrationCovariance(np.eye(3) * 1e-8)  # Numerical integration uncertainty
        
        # Bias random walk
        self.params.setBiasAccCovariance(np.eye(3) * (accel_bias_rw_sigma ** 2))
        self.params.setBiasOmegaCovariance(np.eye(3) * (gyro_bias_rw_sigma ** 2))
        
        # Coriolis settings
        self.params.setUse2ndOrderCoriolis(use_2nd_order_coriolis)
        
        # Store gravity vector
        self.gravity = np.array([0, 0, -gravity_magnitude])
        
        # Initialize pre-integration measurement
        self.reset()
        
    def reset(self, bias_acc: Optional[np.ndarray] = None, 
              bias_gyro: Optional[np.ndarray] = None):
        """
        Reset pre-integration with optional initial bias.
        
        Args:
            bias_acc: Initial accelerometer bias [3]
            bias_gyro: Initial gyroscope bias [3]
        """
        if bias_acc is None:
            bias_acc = np.zeros(3)
        if bias_gyro is None:
            bias_gyro = np.zeros(3)
            
        # Create initial bias
        self.bias = gtsam.imuBias.ConstantBias(bias_acc, bias_gyro)
        
        # Create new pre-integrated measurement (Combined for bias support)
        self.pim = gtsam.PreintegratedCombinedMeasurements(self.params, self.bias)
        
        # Track total integration time
        self.delta_t = 0.0
        
    def integrate_measurement(self, acc: np.ndarray, gyro: np.ndarray, dt: float):
        """
        Integrate single IMU measurement.
        
        Args:
            acc: Accelerometer measurement [3] in m/s²
            gyro: Gyroscope measurement [3] in rad/s
            dt: Time delta in seconds
        """
        self.pim.integrateMeasurement(acc, gyro, dt)
        self.delta_t += dt
        
    def integrate_batch(self, measurements: List[Tuple[np.ndarray, np.ndarray, float]]):
        """
        Integrate batch of IMU measurements.
        
        Args:
            measurements: List of (acc, gyro, dt) tuples
        """
        for acc, gyro, dt in measurements:
            self.integrate_measurement(acc, gyro, dt)
            
    def predict(self, prev_pose: gtsam.Pose3, prev_vel: np.ndarray) -> Tuple[gtsam.Pose3, np.ndarray]:
        """
        Predict next state using pre-integrated measurements.
        
        Args:
            prev_pose: Previous pose (rotation + translation)
            prev_vel: Previous velocity [3] in world frame
            
        Returns:
            predicted_pose: Predicted pose after integration
            predicted_vel: Predicted velocity after integration
        """
        nav_state = gtsam.NavState(prev_pose, prev_vel)
        predicted_state = self.pim.predict(nav_state, self.bias)
        
        return predicted_state.pose(), predicted_state.velocity()
        
    def get_deltas(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get pre-integrated deltas (position, velocity, rotation).
        
        Returns:
            delta_p: Pre-integrated position change [3]
            delta_v: Pre-integrated velocity change [3]
            delta_R: Pre-integrated rotation matrix [3,3]
        """
        delta_p = self.pim.deltaPij()
        delta_v = self.pim.deltaVij()
        delta_R = self.pim.deltaRij().matrix()
        
        return delta_p, delta_v, delta_R
        
    def get_covariance(self) -> np.ndarray:
        """
        Get pre-integration covariance matrix.
        
        Returns:
            cov: 9x9 covariance matrix for (rotation, position, velocity)
        """
        return self.pim.preintMeasCov()
        
    def correct_bias(self, bias_acc_correction: np.ndarray, 
                    bias_gyro_correction: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply bias correction to pre-integrated values.
        Uses first-order approximation for efficiency.
        
        Args:
            bias_acc_correction: Accelerometer bias correction [3]
            bias_gyro_correction: Gyroscope bias correction [3]
            
        Returns:
            Corrected (delta_p, delta_v, delta_R)
        """
        # Get Jacobians
        H_bias_acc = self.pim.H_biasAcc()  # [9x3]
        H_bias_omega = self.pim.H_biasOmega()  # [9x3]
        
        # Compute corrections
        bias_correction = np.concatenate([bias_acc_correction, bias_gyro_correction])
        H_bias = np.hstack([H_bias_acc, H_bias_omega])  # [9x6]
        
        # Apply first-order correction
        delta_xi = H_bias @ bias_correction  # [9x1]
        
        # Extract corrected values
        delta_R_corrected = self.pim.deltaRij().retract(delta_xi[0:3])
        delta_p_corrected = self.pim.deltaPij() + delta_xi[3:6]
        delta_v_corrected = self.pim.deltaVij() + delta_xi[6:9]
        
        return delta_p_corrected, delta_v_corrected, delta_R_corrected.matrix()
        
    def to_torch(self, device='cuda') -> dict:
        """
        Convert pre-integrated values to PyTorch tensors.
        
        Returns:
            Dictionary with torch tensors on specified device
        """
        delta_p, delta_v, delta_R = self.get_deltas()
        
        return {
            'delta_p': torch.from_numpy(delta_p).float().to(device),
            'delta_v': torch.from_numpy(delta_v).float().to(device),
            'delta_R': torch.from_numpy(delta_R).float().to(device),
            'delta_t': torch.tensor(self.delta_t).float().to(device),
            'covariance': torch.from_numpy(self.get_covariance()).float().to(device),
        }
        
    def create_factor(self, pose_i_key: int, vel_i_key: int, 
                     pose_j_key: int, vel_j_key: int, 
                     bias_key: int) -> gtsam.ImuFactor:
        """
        Create GTSAM IMU factor for optimization.
        
        Args:
            pose_i_key: Key for pose at time i
            vel_i_key: Key for velocity at time i
            pose_j_key: Key for pose at time j
            vel_j_key: Key for velocity at time j
            bias_key: Key for IMU bias
            
        Returns:
            GTSAM IMU factor
        """
        return gtsam.ImuFactor(
            pose_i_key, vel_i_key,
            pose_j_key, vel_j_key,
            bias_key, self.pim
        )


class IMUPreintegrationBuffer:
    """
    Manages multiple pre-integration instances for sliding window.
    """
    
    def __init__(self, window_size: int = 10, **kwargs):
        """
        Initialize buffer for sliding window pre-integration.
        
        Args:
            window_size: Number of frames in sliding window
            **kwargs: Parameters passed to GTSAMPreintegration
        """
        self.window_size = window_size
        self.params = kwargs
        
        # Pre-integration between consecutive frames
        self.preintegrators = [
            GTSAMPreintegration(**kwargs) 
            for _ in range(window_size - 1)
        ]
        
        # Current active index
        self.current_idx = 0
        
    def add_measurement(self, acc: np.ndarray, gyro: np.ndarray, dt: float):
        """
        Add measurement to current pre-integrator.
        """
        if self.current_idx < len(self.preintegrators):
            self.preintegrators[self.current_idx].integrate_measurement(acc, gyro, dt)
            
    def new_keyframe(self):
        """
        Signal new keyframe, move to next pre-integrator.
        """
        self.current_idx = (self.current_idx + 1) % (self.window_size - 1)
        
        # Reset the pre-integrator we'll use next
        if self.current_idx < len(self.preintegrators):
            self.preintegrators[self.current_idx].reset()
            
    def get_current(self) -> GTSAMPreintegration:
        """
        Get current active pre-integrator.
        """
        return self.preintegrators[self.current_idx]
        
    def get_window_factors(self) -> List[GTSAMPreintegration]:
        """
        Get all pre-integrators in current window.
        """
        # Return in temporal order
        factors = []
        for i in range(self.window_size - 1):
            idx = (self.current_idx - i) % (self.window_size - 1)
            if self.preintegrators[idx].delta_t > 0:
                factors.append(self.preintegrators[idx])
        return factors[::-1]  # Reverse to get temporal order