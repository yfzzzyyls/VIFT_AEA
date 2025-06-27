#!/usr/bin/env python3
"""
Data augmentation for VIFT training.
Includes photometric and IMU augmentations for better generalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class PhotometricAugmentation(nn.Module):
    """
    Photometric augmentation for visual data.
    Helps SEA-RAFT maintain performance in varying lighting conditions.
    """
    
    def __init__(self, 
                 brightness_range: Tuple[float, float] = (0.9, 1.1),
                 contrast_range: Tuple[float, float] = (0.9, 1.1),
                 saturation_range: Tuple[float, float] = (0.9, 1.1),
                 hue_shift_range: Tuple[float, float] = (-0.05, 0.05),
                 gamma_range: Tuple[float, float] = (0.9, 1.1),
                 noise_std: float = 0.01,
                 p: float = 0.5):
        """
        Initialize photometric augmentation.
        
        Args:
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            saturation_range: Range for saturation adjustment
            hue_shift_range: Range for hue shift
            gamma_range: Range for gamma correction
            noise_std: Standard deviation for Gaussian noise
            p: Probability of applying augmentation
        """
        super().__init__()
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_shift_range = hue_shift_range
        self.gamma_range = gamma_range
        self.noise_std = noise_std
        self.p = p
        
    def adjust_brightness(self, image: torch.Tensor, factor: float) -> torch.Tensor:
        """Adjust brightness by scaling."""
        return torch.clamp(image * factor, 0, 1)
        
    def adjust_contrast(self, image: torch.Tensor, factor: float) -> torch.Tensor:
        """Adjust contrast."""
        mean = image.mean(dim=[-2, -1], keepdim=True)
        return torch.clamp((image - mean) * factor + mean, 0, 1)
        
    def adjust_gamma(self, image: torch.Tensor, gamma: float) -> torch.Tensor:
        """Apply gamma correction."""
        return torch.clamp(image.pow(gamma), 0, 1)
        
    def add_noise(self, image: torch.Tensor, std: float) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(image) * std
        return torch.clamp(image + noise, 0, 1)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply photometric augmentation to images.
        
        Args:
            images: [B, T, 3, H, W] or [B, 3, H, W] tensor of images
            
        Returns:
            Augmented images
        """
        if not self.training or torch.rand(1).item() > self.p:
            return images
            
        # Handle both batched sequences and single images
        original_shape = images.shape
        if len(original_shape) == 5:  # [B, T, 3, H, W]
            B, T = original_shape[:2]
            images = images.reshape(B * T, *original_shape[2:])
        else:
            B, T = images.shape[0], 1
            
        # Sample augmentation parameters
        brightness = torch.empty(1).uniform_(*self.brightness_range).item()
        contrast = torch.empty(1).uniform_(*self.contrast_range).item()
        gamma = torch.empty(1).uniform_(*self.gamma_range).item()
        
        # Apply augmentations
        images = self.adjust_brightness(images, brightness)
        images = self.adjust_contrast(images, contrast)
        images = self.adjust_gamma(images, gamma)
        
        if self.noise_std > 0:
            images = self.add_noise(images, self.noise_std)
            
        # Reshape back
        if len(original_shape) == 5:
            images = images.reshape(B, T, *original_shape[2:])
            
        return images


class IMUAugmentation(nn.Module):
    """
    IMU augmentation with synthetic bias walks and noise.
    Helps bias predictor generalize to various motion patterns.
    """
    
    def __init__(self,
                 bias_walk_std_acc: float = 0.2,      # m/s²
                 bias_walk_std_gyro: float = 0.015,   # rad/s (~150°/s)
                 noise_std_acc: float = 0.1,          # m/s²
                 noise_std_gyro: float = 0.01,        # rad/s
                 bias_correlation_time: float = 100.0, # seconds
                 temperature_coefficient: float = 0.01, # bias/°C
                 p: float = 0.5):
        """
        Initialize IMU augmentation.
        
        Args:
            bias_walk_std_acc: Std dev of accelerometer bias walk
            bias_walk_std_gyro: Std dev of gyroscope bias walk
            noise_std_acc: Std dev of accelerometer noise
            noise_std_gyro: Std dev of gyroscope noise
            bias_correlation_time: Time constant for bias evolution
            temperature_coefficient: Temperature sensitivity
            p: Probability of applying augmentation
        """
        super().__init__()
        self.bias_walk_std_acc = bias_walk_std_acc
        self.bias_walk_std_gyro = bias_walk_std_gyro
        self.noise_std_acc = noise_std_acc
        self.noise_std_gyro = noise_std_gyro
        self.bias_correlation_time = bias_correlation_time
        self.temperature_coefficient = temperature_coefficient
        self.p = p
        
    def generate_bias_walk(self, 
                          length: int, 
                          dt: float,
                          std_acc: float,
                          std_gyro: float,
                          device: torch.device) -> torch.Tensor:
        """
        Generate correlated bias walk using Ornstein-Uhlenbeck process.
        
        Args:
            length: Number of time steps
            dt: Time step size
            std_acc: Accelerometer bias std dev
            std_gyro: Gyroscope bias std dev
            device: Torch device
            
        Returns:
            bias_walk: [length, 6] tensor of bias values
        """
        # OU process parameters
        theta = 1.0 / self.bias_correlation_time
        sigma_acc = std_acc * np.sqrt(2 * theta)
        sigma_gyro = std_gyro * np.sqrt(2 * theta)
        
        # Initialize
        bias = torch.zeros(length, 6, device=device)
        bias[0, :3] = torch.randn(3, device=device) * std_acc
        bias[0, 3:] = torch.randn(3, device=device) * std_gyro
        
        # Generate walk
        for t in range(1, length):
            # OU process: dx = -theta * x * dt + sigma * dW
            drift_acc = -theta * bias[t-1, :3] * dt
            drift_gyro = -theta * bias[t-1, 3:] * dt
            
            diffusion_acc = sigma_acc * np.sqrt(dt) * torch.randn(3, device=device)
            diffusion_gyro = sigma_gyro * np.sqrt(dt) * torch.randn(3, device=device)
            
            bias[t, :3] = bias[t-1, :3] + drift_acc + diffusion_acc
            bias[t, 3:] = bias[t-1, 3:] + drift_gyro + diffusion_gyro
            
        return bias
        
    def add_temperature_bias(self, 
                           imu_data: torch.Tensor,
                           temperature_profile: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add temperature-dependent bias.
        
        Args:
            imu_data: IMU measurements
            temperature_profile: Temperature over time (if None, generate random)
            
        Returns:
            IMU data with temperature bias
        """
        if temperature_profile is None:
            # Generate smooth temperature variation (e.g., device warming up)
            length = imu_data.shape[1] if len(imu_data.shape) > 2 else imu_data.shape[0]
            t = torch.linspace(0, 1, length, device=imu_data.device)
            base_temp = 20.0  # °C
            temp_variation = 5.0 * torch.sin(2 * np.pi * t) + 2.0 * t  # Warming + variation
            temperature_profile = base_temp + temp_variation
            
        # Temperature coefficient affects gyroscope more than accelerometer
        temp_bias = torch.zeros_like(imu_data)
        temp_bias[..., 3:] = self.temperature_coefficient * temperature_profile.unsqueeze(-1)
        temp_bias[..., :3] = 0.1 * self.temperature_coefficient * temperature_profile.unsqueeze(-1)
        
        return imu_data + temp_bias
        
    def forward(self, imu_data: torch.Tensor, dt: float = 0.001) -> torch.Tensor:
        """
        Apply IMU augmentation.
        
        Args:
            imu_data: [B, T, K, 6] or [B, T, 6] IMU measurements
            dt: Time step (default 1ms for 1000Hz)
            
        Returns:
            Augmented IMU data
        """
        if not self.training or torch.rand(1).item() > self.p:
            return imu_data
            
        original_shape = imu_data.shape
        device = imu_data.device
        
        # Handle variable-length format
        if len(original_shape) == 4:  # [B, T, K, 6]
            B, T, K = original_shape[:3]
            # Flatten to [B, T*K, 6] for augmentation
            imu_flat = imu_data.reshape(B, T*K, 6)
        else:  # [B, T, 6]
            B, T = original_shape[:2]
            imu_flat = imu_data
            K = 1
            
        # Generate bias walks for each batch
        bias_walks = []
        for b in range(B):
            bias_walk = self.generate_bias_walk(
                imu_flat.shape[1], 
                dt,
                self.bias_walk_std_acc,
                self.bias_walk_std_gyro,
                device
            )
            bias_walks.append(bias_walk)
        bias_walks = torch.stack(bias_walks)  # [B, T*K, 6]
        
        # Add bias
        imu_augmented = imu_flat + bias_walks
        
        # Add measurement noise
        acc_noise = torch.randn_like(imu_augmented[..., :3]) * self.noise_std_acc
        gyro_noise = torch.randn_like(imu_augmented[..., 3:]) * self.noise_std_gyro
        imu_augmented[..., :3] += acc_noise
        imu_augmented[..., 3:] += gyro_noise
        
        # Add temperature effects
        if torch.rand(1).item() < 0.3:  # 30% chance
            imu_augmented = self.add_temperature_bias(imu_augmented)
            
        # Reshape back
        if len(original_shape) == 4:
            imu_augmented = imu_augmented.reshape(B, T, K, 6)
            
        return imu_augmented


class VIFTAugmentation(nn.Module):
    """
    Combined augmentation for VIFT training.
    """
    
    def __init__(self,
                 enable_photometric: bool = True,
                 enable_imu: bool = True,
                 photometric_kwargs: Optional[Dict] = None,
                 imu_kwargs: Optional[Dict] = None):
        """
        Initialize combined augmentation.
        
        Args:
            enable_photometric: Enable photometric augmentation
            enable_imu: Enable IMU augmentation
            photometric_kwargs: Arguments for PhotometricAugmentation
            imu_kwargs: Arguments for IMUAugmentation
        """
        super().__init__()
        
        self.enable_photometric = enable_photometric
        self.enable_imu = enable_imu
        
        if enable_photometric:
            photometric_kwargs = photometric_kwargs or {}
            self.photometric_aug = PhotometricAugmentation(**photometric_kwargs)
            
        if enable_imu:
            imu_kwargs = imu_kwargs or {}
            self.imu_aug = IMUAugmentation(**imu_kwargs)
            
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply augmentations to batch.
        
        Args:
            batch: Dictionary with 'images' and 'imu' keys
            
        Returns:
            Augmented batch
        """
        augmented_batch = batch.copy()
        
        if self.enable_photometric and 'images' in batch:
            augmented_batch['images'] = self.photometric_aug(batch['images'])
            
        if self.enable_imu and 'imu' in batch:
            augmented_batch['imu'] = self.imu_aug(batch['imu'])
            
        return augmented_batch