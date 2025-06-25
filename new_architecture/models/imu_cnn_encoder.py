"""
1D CNN IMU Encoder for Visual-Inertial Odometry
Based on the original VIFT architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class IMUCNN1DEncoder(nn.Module):
    """
    3-layer 1D CNN for IMU feature extraction.
    Processes all IMU samples between consecutive frames.
    """
    
    def __init__(
        self,
        input_channels: int = 6,
        hidden_channels: List[int] = [64, 128, 256],
        output_dim: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Build CNN layers
        layers = []
        in_channels = input_channels
        
        for i, out_channels in enumerate(hidden_channels):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*layers)
        
        # Adaptive pooling to handle variable length sequences
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection to output dimension
        self.projection = nn.Linear(hidden_channels[-1], output_dim)
        
    def forward(self, imu_sequences: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Process variable-length IMU sequences with 1D CNN.
        
        Args:
            imu_sequences: List of batch sequences, each containing T-1 IMU segments
                          imu_sequences[b][t] has shape [num_samples, 6]
                          
        Returns:
            features: [B, T-1, output_dim] - IMU features for each transition
        """
        batch_size = len(imu_sequences)
        seq_len = len(imu_sequences[0])
        device = imu_sequences[0][0].device
        
        # Process each batch and time step
        all_features = []
        
        for b in range(batch_size):
            batch_features = []
            
            for t in range(seq_len):
                # Get IMU data for this transition: [num_samples, 6]
                imu_data = imu_sequences[b][t]
                
                # Handle empty sequences
                if imu_data.shape[0] == 0:
                    batch_features.append(torch.zeros(self.projection.out_features, device=device))
                    continue
                
                # Transpose for Conv1d: [6, num_samples]
                imu_data = imu_data.transpose(0, 1).unsqueeze(0)  # [1, 6, num_samples]
                
                # Apply CNN
                features = self.cnn(imu_data)  # [1, 256, num_samples]
                
                # Global average pooling
                features = self.adaptive_pool(features)  # [1, 256, 1]
                features = features.squeeze(-1).squeeze(0)  # [256]
                
                # Project to output dimension
                features = self.projection(features)  # [output_dim]
                
                batch_features.append(features)
            
            # Stack time steps: [T-1, output_dim]
            batch_features = torch.stack(batch_features, dim=0)
            all_features.append(batch_features)
        
        # Stack batches: [B, T-1, output_dim]
        return torch.stack(all_features, dim=0)


class IMUCNN1DEncoderEfficient(nn.Module):
    """
    More efficient version that processes all sequences in parallel.
    Better for fixed-length sequences or when padding is acceptable.
    """
    
    def __init__(
        self,
        input_channels: int = 6,
        hidden_channels: List[int] = [64, 128, 256],
        output_dim: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.1,
        expected_length: int = 50  # Expected number of IMU samples
    ):
        super().__init__()
        
        self.expected_length = expected_length
        
        # Build CNN layers
        layers = []
        in_channels = input_channels
        
        for i, out_channels in enumerate(hidden_channels):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection
        self.projection = nn.Linear(hidden_channels[-1], output_dim)
        
    def forward(self, imu_sequences: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Process IMU sequences efficiently by padding and batching.
        """
        batch_size = len(imu_sequences)
        seq_len = len(imu_sequences[0])
        device = imu_sequences[0][0].device
        
        # Flatten all sequences and pad to expected length
        all_sequences = []
        
        for b in range(batch_size):
            for t in range(seq_len):
                imu_data = imu_sequences[b][t]  # [num_samples, 6]
                
                # Pad or truncate to expected length
                if imu_data.shape[0] < self.expected_length:
                    # Pad with zeros
                    padding = torch.zeros(
                        self.expected_length - imu_data.shape[0], 6, 
                        device=device
                    )
                    imu_data = torch.cat([imu_data, padding], dim=0)
                elif imu_data.shape[0] > self.expected_length:
                    # Truncate
                    imu_data = imu_data[:self.expected_length]
                
                all_sequences.append(imu_data)
        
        # Stack and reshape: [B*T, 6, expected_length]
        all_sequences = torch.stack(all_sequences, dim=0)  # [B*T, expected_length, 6]
        all_sequences = all_sequences.transpose(1, 2)  # [B*T, 6, expected_length]
        
        # Process through CNN
        features = self.cnn(all_sequences)  # [B*T, 256, expected_length]
        
        # Global pooling
        features = self.global_pool(features)  # [B*T, 256, 1]
        features = features.squeeze(-1)  # [B*T, 256]
        
        # Project
        features = self.projection(features)  # [B*T, output_dim]
        
        # Reshape back to [B, T-1, output_dim]
        features = features.view(batch_size, seq_len, -1)
        
        return features