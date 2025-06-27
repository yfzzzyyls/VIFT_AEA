"""
Zero-velocity Update (ZUPT) Detector
Detects stationary periods for AR/VR applications.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import List, Tuple, Optional, Dict
from collections import deque
from dataclasses import dataclass


@dataclass
class ZUPTDetection:
    """ZUPT detection result."""
    timestamp: float
    is_stationary: bool
    confidence: float  # 0-1 confidence score
    acc_variance: float
    gyro_variance: float
    
    
class ZUPTDetector:
    """
    Detects zero-velocity periods using IMU measurements.
    
    Uses multiple signals:
    - Accelerometer magnitude variance
    - Gyroscope magnitude variance  
    - Acceleration vector stability
    - Adaptive thresholds based on noise characteristics
    """
    
    def __init__(self,
                 window_size: int = 50,          # 50ms at 1000Hz
                 acc_threshold: float = 0.5,     # m/s² std dev
                 gyro_threshold: float = 0.05,   # rad/s std dev
                 gravity_magnitude: float = 9.81,
                 confidence_decay: float = 0.95,
                 adaptive_threshold: bool = True):
        """
        Initialize ZUPT detector.
        
        Args:
            window_size: Number of IMU samples for detection window
            acc_threshold: Accelerometer variance threshold
            gyro_threshold: Gyroscope variance threshold  
            gravity_magnitude: Expected gravity magnitude
            confidence_decay: Decay factor for confidence over time
            adaptive_threshold: Enable adaptive threshold adjustment
        """
        self.window_size = window_size
        self.acc_threshold = acc_threshold
        self.gyro_threshold = gyro_threshold
        self.gravity_magnitude = gravity_magnitude
        self.confidence_decay = confidence_decay
        self.adaptive_threshold = adaptive_threshold
        
        # Sliding window buffers
        self.acc_buffer = deque(maxlen=window_size)
        self.gyro_buffer = deque(maxlen=window_size)
        self.timestamp_buffer = deque(maxlen=window_size)
        
        # Adaptive threshold parameters
        self.noise_acc_baseline = acc_threshold
        self.noise_gyro_baseline = gyro_threshold
        self.adaptation_rate = 0.01
        
        # State
        self.last_detection = None
        self.stationary_count = 0
        self.moving_count = 0
        
    def add_measurement(self, 
                       acc: np.ndarray, 
                       gyro: np.ndarray,
                       timestamp: float) -> Optional[ZUPTDetection]:
        """
        Add IMU measurement and check for zero velocity.
        
        Args:
            acc: Accelerometer measurement [3] in m/s²
            gyro: Gyroscope measurement [3] in rad/s
            timestamp: Measurement timestamp
            
        Returns:
            ZUPT detection if window is full, None otherwise
        """
        # Add to buffers
        self.acc_buffer.append(acc)
        self.gyro_buffer.append(gyro)
        self.timestamp_buffer.append(timestamp)
        
        # Need full window for detection
        if len(self.acc_buffer) < self.window_size:
            return None
            
        # Perform detection
        return self._detect_zupt()
        
    def _detect_zupt(self) -> ZUPTDetection:
        """
        Detect zero velocity from buffered measurements.
        
        Returns:
            ZUPT detection result
        """
        # Convert buffers to arrays
        acc_array = np.array(self.acc_buffer)  # [N, 3]
        gyro_array = np.array(self.gyro_buffer)  # [N, 3]
        
        # 1. Accelerometer variance check
        acc_var = np.var(acc_array, axis=0)  # Variance per axis
        acc_std = np.sqrt(np.mean(acc_var))  # RMS std deviation
        
        # 2. Gyroscope variance check  
        gyro_var = np.var(gyro_array, axis=0)
        gyro_std = np.sqrt(np.mean(gyro_var))
        
        # 3. Acceleration magnitude check (should be close to gravity)
        acc_magnitudes = np.linalg.norm(acc_array, axis=1)
        acc_mag_mean = np.mean(acc_magnitudes)
        acc_mag_std = np.std(acc_magnitudes)
        
        # 4. Acceleration direction stability
        acc_directions = acc_array / (acc_magnitudes[:, np.newaxis] + 1e-9)
        mean_direction = np.mean(acc_directions, axis=0)
        direction_deviations = 1 - np.dot(acc_directions, mean_direction)
        direction_stability = 1 - np.mean(direction_deviations)
        
        # Get current thresholds
        acc_thresh = self._get_adaptive_threshold('acc')
        gyro_thresh = self._get_adaptive_threshold('gyro')
        
        # Compute individual scores
        acc_score = np.exp(-acc_std / acc_thresh)
        gyro_score = np.exp(-gyro_std / gyro_thresh)
        
        # Gravity magnitude score (should be close to expected gravity)
        gravity_error = abs(acc_mag_mean - self.gravity_magnitude)
        gravity_score = np.exp(-gravity_error / 0.5)  # 0.5 m/s² tolerance
        
        # Magnitude stability score
        mag_stability_score = np.exp(-acc_mag_std / 0.2)  # 0.2 m/s² tolerance
        
        # Combined confidence
        confidence = (
            0.3 * acc_score +
            0.3 * gyro_score +
            0.2 * gravity_score +
            0.1 * mag_stability_score +
            0.1 * direction_stability
        )
        
        # Binary decision with hysteresis
        is_stationary = self._apply_hysteresis(confidence)
        
        # Update adaptive thresholds if enabled
        if self.adaptive_threshold and is_stationary:
            self._update_noise_baseline(acc_std, gyro_std)
            
        # Create detection result
        detection = ZUPTDetection(
            timestamp=self.timestamp_buffer[-1],
            is_stationary=is_stationary,
            confidence=confidence,
            acc_variance=acc_std,
            gyro_variance=gyro_std
        )
        
        self.last_detection = detection
        return detection
        
    def _apply_hysteresis(self, confidence: float) -> bool:
        """
        Apply hysteresis to avoid rapid switching.
        
        Args:
            confidence: Current confidence score
            
        Returns:
            Stationary decision with hysteresis
        """
        # Thresholds
        high_threshold = 0.7
        low_threshold = 0.3
        
        if self.last_detection is None:
            # First detection
            is_stationary = confidence > high_threshold
        else:
            if self.last_detection.is_stationary:
                # Currently stationary - need low confidence to switch
                is_stationary = confidence > low_threshold
            else:
                # Currently moving - need high confidence to switch
                is_stationary = confidence > high_threshold
                
        # Update state counters
        if is_stationary:
            self.stationary_count += 1
            self.moving_count = 0
        else:
            self.moving_count += 1
            self.stationary_count = 0
            
        return is_stationary
        
    def _get_adaptive_threshold(self, sensor: str) -> float:
        """
        Get adaptive threshold for sensor.
        
        Args:
            sensor: 'acc' or 'gyro'
            
        Returns:
            Current threshold
        """
        if not self.adaptive_threshold:
            return self.acc_threshold if sensor == 'acc' else self.gyro_threshold
            
        if sensor == 'acc':
            return self.noise_acc_baseline
        else:
            return self.noise_gyro_baseline
            
    def _update_noise_baseline(self, acc_std: float, gyro_std: float):
        """
        Update noise baseline during stationary periods.
        
        Args:
            acc_std: Current accelerometer std deviation
            gyro_std: Current gyroscope std deviation
        """
        # Only update after sufficient stationary samples
        if self.stationary_count < 10:
            return
            
        # Exponential moving average update
        self.noise_acc_baseline = (
            (1 - self.adaptation_rate) * self.noise_acc_baseline +
            self.adaptation_rate * (acc_std * 1.5)  # 1.5x safety factor
        )
        
        self.noise_gyro_baseline = (
            (1 - self.adaptation_rate) * self.noise_gyro_baseline +
            self.adaptation_rate * (gyro_std * 1.5)
        )
        
        # Clamp to reasonable ranges
        self.noise_acc_baseline = np.clip(self.noise_acc_baseline, 0.1, 2.0)
        self.noise_gyro_baseline = np.clip(self.noise_gyro_baseline, 0.01, 0.2)
        
    def get_zupt_measurement(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get ZUPT pseudo-measurement for Kalman filter update.
        
        Returns:
            h: Measurement vector (velocity = 0)
            R: Measurement noise covariance
        """
        # Measurement: velocity should be zero
        h = np.zeros(3)
        
        # Measurement noise depends on confidence
        if self.last_detection is not None:
            # Lower confidence = higher noise
            base_noise = 0.01  # 1 cm/s base noise
            noise_scale = 1.0 / (self.last_detection.confidence + 0.1)
            R = np.eye(3) * (base_noise * noise_scale) ** 2
        else:
            # Default high noise
            R = np.eye(3) * 0.1 ** 2
            
        return h, R
        
    def get_state_constraint_jacobian(self, state_dim: int, 
                                    vel_start_idx: int = 3) -> np.ndarray:
        """
        Get Jacobian for ZUPT constraint.
        
        Args:
            state_dim: Total state dimension
            vel_start_idx: Start index of velocity in state vector
            
        Returns:
            H: Jacobian matrix [3 x state_dim]
        """
        H = np.zeros((3, state_dim))
        H[:, vel_start_idx:vel_start_idx+3] = np.eye(3)
        return H
        
    def reset(self):
        """Reset detector state."""
        self.acc_buffer.clear()
        self.gyro_buffer.clear()
        self.timestamp_buffer.clear()
        self.last_detection = None
        self.stationary_count = 0
        self.moving_count = 0
        
    def get_statistics(self) -> dict:
        """Get detector statistics."""
        stats = {
            'buffer_size': len(self.acc_buffer),
            'stationary_count': self.stationary_count,
            'moving_count': self.moving_count,
            'acc_threshold': self._get_adaptive_threshold('acc'),
            'gyro_threshold': self._get_adaptive_threshold('gyro'),
        }
        
        if self.last_detection is not None:
            stats.update({
                'last_confidence': self.last_detection.confidence,
                'last_stationary': self.last_detection.is_stationary,
                'last_acc_std': self.last_detection.acc_variance,
                'last_gyro_std': self.last_detection.gyro_variance,
            })
            
        return stats


class ZUPTConfidenceLSTM(nn.Module):
    """
    LSTM-based confidence predictor for ZUPT detection.
    Learns to predict confidence from IMU sequence patterns.
    """
    
    def __init__(self, 
                 input_dim: int = 12,  # 6 IMU + 6 derived features
                 hidden_dim: int = 32,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Initialize LSTM confidence predictor.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output projection
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Confidence in [0, 1]
        )
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.LayerNorm(12)
        )
        
    def extract_features(self, imu_window: torch.Tensor) -> torch.Tensor:
        """
        Extract features from IMU window.
        
        Args:
            imu_window: [B, T, 6] IMU measurements
            
        Returns:
            features: [B, T, 12] extracted features
        """
        B, T, _ = imu_window.shape
        
        # Basic features
        acc = imu_window[:, :, :3]
        gyro = imu_window[:, :, 3:]
        
        # Magnitude features
        acc_mag = torch.norm(acc, dim=-1, keepdim=True)
        gyro_mag = torch.norm(gyro, dim=-1, keepdim=True)
        
        # Variance features (computed over small windows)
        window_size = min(5, T)
        acc_var = torch.zeros(B, T, 1, device=imu_window.device)
        gyro_var = torch.zeros(B, T, 1, device=imu_window.device)
        
        for i in range(T):
            start = max(0, i - window_size // 2)
            end = min(T, i + window_size // 2 + 1)
            acc_var[:, i] = acc[:, start:end].var(dim=1).mean(dim=-1, keepdim=True)
            gyro_var[:, i] = gyro[:, start:end].var(dim=1).mean(dim=-1, keepdim=True)
            
        # Concatenate all features
        features = torch.cat([
            imu_window,      # Original 6D
            acc_mag,         # 1D
            gyro_mag,        # 1D
            acc_var,         # 1D
            gyro_var,        # 1D
            acc_mag - 9.81,  # Gravity deviation 1D
            torch.diff(gyro_mag, dim=1, prepend=gyro_mag[:, 0:1])  # Gyro change 1D
        ], dim=-1)
        
        # Project to feature space
        features = self.feature_extractor(imu_window)
        
        return features
        
    def forward(self, imu_window: torch.Tensor) -> torch.Tensor:
        """
        Predict ZUPT confidence from IMU window.
        
        Args:
            imu_window: [B, T, 6] IMU measurements
            
        Returns:
            confidence: [B] confidence scores in [0, 1]
        """
        # Extract features
        features = self.extract_features(imu_window)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(features)
        
        # Use last hidden state (combine forward and backward)
        last_hidden = lstm_out[:, -1, :]  # [B, hidden_dim * 2]
        
        # Predict confidence
        confidence = self.confidence_head(last_hidden).squeeze(-1)  # [B]
        
        return confidence
        
    def compute_loss(self, 
                    pred_confidence: torch.Tensor,
                    true_stationary: torch.Tensor,
                    rule_based_confidence: torch.Tensor,
                    alpha: float = 0.7) -> torch.Tensor:
        """
        Compute loss for confidence prediction.
        
        Args:
            pred_confidence: [B] predicted confidence
            true_stationary: [B] ground truth stationary labels (0/1)
            rule_based_confidence: [B] rule-based confidence for regularization
            alpha: Weight for ground truth vs rule-based loss
            
        Returns:
            loss: Scalar loss value
        """
        # Binary cross entropy with ground truth
        bce_loss = nn.functional.binary_cross_entropy(
            pred_confidence, 
            true_stationary.float()
        )
        
        # MSE with rule-based confidence (regularization)
        mse_loss = nn.functional.mse_loss(
            pred_confidence,
            rule_based_confidence
        )
        
        # Combined loss
        loss = alpha * bce_loss + (1 - alpha) * mse_loss
        
        return loss


class ZUPTDetectorWithLSTM(ZUPTDetector):
    """
    Enhanced ZUPT detector with LSTM confidence prediction.
    """
    
    def __init__(self, 
                 *args,
                 use_lstm: bool = True,
                 lstm_model_path: Optional[str] = None,
                 device: str = 'cpu',
                 **kwargs):
        """
        Initialize detector with optional LSTM confidence.
        
        Args:
            use_lstm: Whether to use LSTM confidence
            lstm_model_path: Path to pre-trained LSTM model
            device: Device for LSTM inference
            *args, **kwargs: Arguments for base ZUPTDetector
        """
        super().__init__(*args, **kwargs)
        
        self.use_lstm = use_lstm
        self.device = device
        
        if use_lstm:
            self.lstm_confidence = ZUPTConfidenceLSTM().to(device)
            self.lstm_confidence.eval()
            
            if lstm_model_path and os.path.exists(lstm_model_path):
                self.lstm_confidence.load_state_dict(
                    torch.load(lstm_model_path, map_location=device)
                )
                print(f"Loaded LSTM confidence model from {lstm_model_path}")
                
    def _detect_zupt(self) -> ZUPTDetection:
        """
        Enhanced ZUPT detection with LSTM confidence.
        """
        # Get base detection
        base_detection = super()._detect_zupt()
        
        if not self.use_lstm:
            return base_detection
            
        # Prepare IMU window for LSTM
        imu_window = np.stack([
            np.concatenate([acc, gyro]) 
            for acc, gyro in zip(self.acc_buffer, self.gyro_buffer)
        ])
        imu_tensor = torch.tensor(imu_window, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get LSTM confidence
        with torch.no_grad():
            lstm_confidence = self.lstm_confidence(imu_tensor).item()
            
        # Override confidence only if LSTM is very confident
        if lstm_confidence > 0.9:
            base_detection.confidence = lstm_confidence
            base_detection.is_stationary = True
        elif lstm_confidence < 0.1:
            base_detection.confidence = lstm_confidence
            base_detection.is_stationary = False
        # Otherwise blend with rule-based confidence
        else:
            base_detection.confidence = 0.7 * base_detection.confidence + 0.3 * lstm_confidence
            
        return base_detection


class ZUPTUpdater:
    """
    Applies ZUPT constraints to MSCKF state.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 max_velocity_correction: float = 0.5):  # m/s
        """
        Initialize ZUPT updater.
        
        Args:
            confidence_threshold: Minimum confidence to apply ZUPT
            max_velocity_correction: Maximum velocity correction magnitude
        """
        self.confidence_threshold = confidence_threshold
        self.max_velocity_correction = max_velocity_correction
        
    def apply_zupt(self, 
                  detection: ZUPTDetection,
                  state: 'MSCKFState') -> Tuple[bool, np.ndarray]:
        """
        Apply ZUPT update to state if confident.
        
        Args:
            detection: ZUPT detection result
            state: MSCKF state to update
            
        Returns:
            (applied, velocity_correction)
        """
        # Check confidence
        if not detection.is_stationary:
            return False, np.zeros(3)
            
        if detection.confidence < self.confidence_threshold:
            return False, np.zeros(3)
            
        # Get current velocity
        current_velocity = state.imu_state.velocity
        
        # Compute correction (negative of current velocity)
        velocity_correction = -current_velocity
        
        # Limit correction magnitude
        correction_norm = np.linalg.norm(velocity_correction)
        if correction_norm > self.max_velocity_correction:
            velocity_correction *= self.max_velocity_correction / correction_norm
            
        # Apply correction
        state.imu_state.velocity += velocity_correction
        
        # Update covariance
        # Simple approach: reduce velocity uncertainty
        vel_idx = 3  # Velocity starts at index 3 in state vector
        state.P[vel_idx:vel_idx+3, vel_idx:vel_idx+3] *= (1 - detection.confidence)
        
        return True, velocity_correction