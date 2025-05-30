"""
AR/VR Specific Data Augmentations
Specialized augmentations that simulate realistic head motion patterns
for improved AR/VR visual-inertial odometry training.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import random


class ARVRMotionAugmentation:
    """
    Augmentation class specifically designed for AR/VR head motion patterns.
    Simulates realistic head movements, rotational jitter, and motion blur effects.
    """
    
    def __init__(
        self,
        rotational_jitter_deg: float = 15.0,
        translational_shake_cm: float = 2.0,
        motion_blur_prob: float = 0.3,
        rapid_motion_prob: float = 0.2,
        micro_motion_prob: float = 0.4,
        seed: Optional[int] = None
    ):
        """
        Args:
            rotational_jitter_deg: Maximum rotational jitter in degrees (±15° typical)
            translational_shake_cm: Maximum translational shake in cm (±2cm typical)
            motion_blur_prob: Probability of applying motion blur simulation
            rapid_motion_prob: Probability of simulating rapid head movements
            micro_motion_prob: Probability of adding micro-movements
        """
        self.rotational_jitter = np.deg2rad(rotational_jitter_deg)
        self.translational_shake = translational_shake_cm / 100.0  # Convert to meters
        self.motion_blur_prob = motion_blur_prob
        self.rapid_motion_prob = rapid_motion_prob
        self.micro_motion_prob = micro_motion_prob
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def simulate_head_motion_patterns(self, pose_sequence: torch.Tensor) -> torch.Tensor:
        """
        Simulate realistic AR/VR head motion patterns.
        
        Args:
            pose_sequence: [seq_len, 7] tensor (x,y,z,qx,qy,qz,qw)
        
        Returns:
            Augmented pose sequence with realistic head motion patterns
        """
        seq_len = pose_sequence.shape[0]
        augmented_poses = pose_sequence.clone()
        
        # 1. Add rotational jitter (typical head micro-movements)
        if random.random() < 0.8:  # Apply to most sequences
            rotational_noise = torch.randn(seq_len, 3) * self.rotational_jitter * 0.3
            augmented_poses = self._apply_rotational_noise(augmented_poses, rotational_noise)
        
        # 2. Add translational shake (hand tremor, walking)
        if random.random() < 0.6:
            translational_noise = torch.randn(seq_len, 3) * self.translational_shake * 0.5
            augmented_poses[:, :3] += translational_noise
        
        # 3. Simulate rapid head movements (looking around quickly)
        if random.random() < self.rapid_motion_prob:
            augmented_poses = self._simulate_rapid_motion(augmented_poses)
        
        # 4. Add micro-movements (subtle head adjustments)
        if random.random() < self.micro_motion_prob:
            augmented_poses = self._add_micro_movements(augmented_poses)
        
        return augmented_poses
    
    def _apply_rotational_noise(self, poses: torch.Tensor, rotational_noise: torch.Tensor) -> torch.Tensor:
        """Apply small rotational perturbations to quaternions."""
        for i in range(poses.shape[0]):
            # Convert small angle to quaternion
            angle = torch.norm(rotational_noise[i])
            if angle > 1e-6:
                axis = rotational_noise[i] / angle
                noise_quat = torch.tensor([
                    axis[0] * torch.sin(angle/2),
                    axis[1] * torch.sin(angle/2), 
                    axis[2] * torch.sin(angle/2),
                    torch.cos(angle/2)
                ])
                
                # Multiply quaternions (current * noise)
                current_quat = poses[i, 3:7]
                poses[i, 3:7] = self._multiply_quaternions(current_quat, noise_quat)
        
        return poses
    
    def _multiply_quaternions(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions (q1 * q2)."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        return torch.tensor([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])
    
    def _simulate_rapid_motion(self, poses: torch.Tensor) -> torch.Tensor:
        """Simulate rapid head movements (looking around quickly)."""
        seq_len = poses.shape[0]
        
        # Choose random segment for rapid motion
        start_idx = random.randint(0, max(0, seq_len - 5))
        end_idx = min(start_idx + random.randint(3, 7), seq_len)
        
        # Apply larger rotational changes in this segment
        for i in range(start_idx, end_idx):
            rapid_rotation = torch.randn(3) * self.rotational_jitter * 2.0
            poses = self._apply_rotational_noise(poses[i:i+1], rapid_rotation.unsqueeze(0))
        
        return poses
    
    def _add_micro_movements(self, poses: torch.Tensor) -> torch.Tensor:
        """Add subtle micro-movements throughout the sequence."""
        seq_len = poses.shape[0]
        
        # Very small, high-frequency movements
        micro_rotations = torch.randn(seq_len, 3) * self.rotational_jitter * 0.1
        micro_translations = torch.randn(seq_len, 3) * self.translational_shake * 0.2
        
        poses = self._apply_rotational_noise(poses, micro_rotations)
        poses[:, :3] += micro_translations
        
        return poses


class ScaleAwareAugmentation:
    """
    Scale-aware augmentation that applies different noise levels based on motion magnitude.
    Preserves small motions while adding appropriate noise to larger motions.
    """
    
    def __init__(
        self,
        small_motion_threshold: float = 0.01,  # 1cm or 1 degree
        large_motion_threshold: float = 0.05,  # 5cm or 5 degrees
        small_motion_noise: float = 0.002,     # 2mm noise for small motions
        large_motion_noise: float = 0.01       # 1cm noise for large motions
    ):
        self.small_threshold = small_motion_threshold
        self.large_threshold = large_motion_threshold
        self.small_noise = small_motion_noise
        self.large_noise = large_motion_noise
    
    def __call__(self, pose_sequence: torch.Tensor) -> torch.Tensor:
        """
        Apply scale-aware augmentation based on motion magnitude.
        
        Args:
            pose_sequence: [seq_len, 7] tensor (x,y,z,qx,qy,qz,qw)
        """
        if pose_sequence.shape[0] < 2:
            return pose_sequence
        
        augmented_poses = pose_sequence.clone()
        
        # Calculate motion magnitudes between consecutive frames
        for i in range(1, pose_sequence.shape[0]):
            # Translation magnitude
            trans_motion = torch.norm(pose_sequence[i, :3] - pose_sequence[i-1, :3])
            
            # Rotation magnitude (simplified)
            q1, q2 = pose_sequence[i-1, 3:7], pose_sequence[i, 3:7]
            rot_motion = torch.abs(1 - torch.abs(torch.dot(q1, q2)))  # Quaternion distance
            
            motion_magnitude = max(trans_motion.item(), rot_motion.item())
            
            # Apply appropriate noise level
            if motion_magnitude < self.small_threshold:
                noise_scale = self.small_noise
            elif motion_magnitude > self.large_threshold:
                noise_scale = self.large_noise
            else:
                # Interpolate between small and large noise
                ratio = (motion_magnitude - self.small_threshold) / (self.large_threshold - self.small_threshold)
                noise_scale = self.small_noise + ratio * (self.large_noise - self.small_noise)
            
            # Apply noise
            trans_noise = torch.randn(3) * noise_scale
            rot_noise = torch.randn(3) * noise_scale * 0.1  # Smaller rotation noise
            
            augmented_poses[i, :3] += trans_noise
            # Apply small rotational noise (simplified)
            
        return augmented_poses


class ARVRDataAugmentation:
    """
    Complete AR/VR data augmentation pipeline combining multiple augmentation strategies.
    """
    
    def __init__(
        self,
        enable_motion_patterns: bool = True,
        enable_scale_aware: bool = True,
        enable_temporal_jitter: bool = True,
        augmentation_prob: float = 0.8
    ):
        self.enable_motion_patterns = enable_motion_patterns
        self.enable_scale_aware = enable_scale_aware
        self.enable_temporal_jitter = enable_temporal_jitter
        self.augmentation_prob = augmentation_prob
        
        # Initialize augmentation modules
        if enable_motion_patterns:
            self.motion_aug = ARVRMotionAugmentation()
        
        if enable_scale_aware:
            self.scale_aug = ScaleAwareAugmentation()
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply AR/VR augmentations to a batch of data.
        
        Args:
            batch: Dictionary containing 'features', 'poses', etc.
        
        Returns:
            Augmented batch
        """
        if random.random() > self.augmentation_prob:
            return batch  # Skip augmentation
        
        augmented_batch = batch.copy()
        
        # Apply augmentations to pose sequences
        if 'poses' in batch:
            poses = batch['poses']
            
            # Apply motion pattern augmentations
            if self.enable_motion_patterns and random.random() < 0.7:
                poses = self.motion_aug.simulate_head_motion_patterns(poses)
            
            # Apply scale-aware augmentations
            if self.enable_scale_aware and random.random() < 0.6:
                poses = self.scale_aug(poses)
            
            # Temporal jitter (frame timing variations)
            if self.enable_temporal_jitter and random.random() < 0.3:
                poses = self._apply_temporal_jitter(poses)
            
            augmented_batch['poses'] = poses
        
        return augmented_batch
    
    def _apply_temporal_jitter(self, poses: torch.Tensor) -> torch.Tensor:
        """Apply small temporal variations to simulate frame timing jitter."""
        seq_len = poses.shape[0]
        if seq_len < 3:
            return poses
        
        # Small interpolation between frames to simulate timing variations
        jitter_factor = 0.1  # 10% temporal jitter
        
        for i in range(1, seq_len - 1):
            if random.random() < 0.3:  # Apply to 30% of frames
                alpha = random.uniform(-jitter_factor, jitter_factor)
                # Linear interpolation with neighboring frames
                poses[i] = (1 - abs(alpha)) * poses[i] + abs(alpha) * (poses[i-1] if alpha < 0 else poses[i+1])
        
        return poses


# Helper function to integrate with existing data loading
def create_arvr_augmentation(config: Dict) -> ARVRDataAugmentation:
    """
    Factory function to create AR/VR augmentation from config.
    
    Args:
        config: Configuration dictionary with augmentation parameters
    
    Returns:
        Configured ARVRDataAugmentation instance
    """
    return ARVRDataAugmentation(
        enable_motion_patterns=config.get('enable_motion_patterns', True),
        enable_scale_aware=config.get('enable_scale_aware', True),
        enable_temporal_jitter=config.get('enable_temporal_jitter', True),
        augmentation_prob=config.get('augmentation_prob', 0.8)
    )