"""
Advanced IMU encoders for handling variable-length sequences with timing variations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class TimeAwareIMULSTMEncoder(nn.Module):
    """
    LSTM encoder that incorporates timing information to handle 
    variable sampling rates between frame pairs.
    """
    
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=3, 
                 output_dim=256, dropout=0.2, bidirectional=True,
                 use_time_encoding=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.use_time_encoding = use_time_encoding
        
        # Adjust input dimension if using time encoding
        lstm_input_dim = 64
        if use_time_encoding:
            # Add 1 for time delta + 1 for sample rate info
            input_projection_dim = input_dim + 2
        else:
            input_projection_dim = input_dim
            
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_projection_dim, lstm_input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output projection with multiple aggregation methods
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # We'll use multiple statistics for robustness
        aggregated_dim = lstm_output_dim * 3  # mean, max, final_state
        
        self.output_projection = nn.Sequential(
            nn.Linear(aggregated_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim)
        )
        
    def add_time_encoding(self, imu_data: torch.Tensor, num_samples: int, 
                         expected_samples: float = 50.0) -> torch.Tensor:
        """
        Add timing information to IMU data.
        
        Args:
            imu_data: [num_samples, 6] IMU measurements
            num_samples: Actual number of samples
            expected_samples: Expected number of samples (default 50 for 1000Hz/20Hz)
            
        Returns:
            IMU data with time encoding [num_samples, 8]
        """
        device = imu_data.device
        
        # Create normalized time steps [0, 1]
        time_steps = torch.linspace(0, 1, num_samples, device=device)
        time_steps = time_steps.unsqueeze(1)  # [num_samples, 1]
        
        # Sample rate deviation (how different from expected)
        sample_rate_factor = num_samples / expected_samples
        rate_encoding = torch.full((num_samples, 1), sample_rate_factor, device=device)
        
        # Concatenate all features
        return torch.cat([imu_data, time_steps, rate_encoding], dim=1)
        
    def forward(self, imu_sequences: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Process variable-length IMU sequences with timing awareness.
        
        Args:
            imu_sequences: List of B sequences, each containing T-1 variable-length IMU segments
                          
        Returns:
            imu_features: [B, T-1, output_dim] - Encoded IMU features
        """
        B = len(imu_sequences)
        T_minus_1 = len(imu_sequences[0])
        device = imu_sequences[0][0].device
        
        all_features = []
        
        # Process each batch element
        for b in range(B):
            sequence_features = []
            
            # Process each IMU segment between consecutive frames
            for t in range(T_minus_1):
                imu_segment = imu_sequences[b][t]  # [num_samples, 6]
                num_samples = imu_segment.shape[0]
                
                # Handle empty segments
                if num_samples == 0:
                    features = torch.zeros(self.output_dim, device=device)
                else:
                    # Add time encoding if enabled
                    if self.use_time_encoding:
                        imu_input = self.add_time_encoding(imu_segment, num_samples)
                    else:
                        imu_input = imu_segment
                    
                    # Project input
                    projected = self.input_projection(imu_input)  # [num_samples, 64]
                    projected = projected.unsqueeze(0)  # [1, num_samples, 64]
                    
                    # Apply LSTM
                    lstm_out, (h_n, c_n) = self.lstm(projected)
                    lstm_out = lstm_out.squeeze(0)  # [num_samples, hidden_dim*2]
                    
                    # Extract multiple aggregations for robustness
                    # 1. Mean pooling
                    mean_features = lstm_out.mean(dim=0)
                    
                    # 2. Max pooling
                    max_features = lstm_out.max(dim=0)[0]
                    
                    # 3. Final hidden state
                    if self.bidirectional:
                        h_forward = h_n[-2, 0, :]
                        h_backward = h_n[-1, 0, :]
                        final_features = torch.cat([h_forward, h_backward], dim=-1)
                    else:
                        final_features = h_n[-1, 0, :]
                    
                    # Concatenate all features
                    aggregated = torch.cat([mean_features, max_features, final_features], dim=-1)
                    
                    # Project to output dimension
                    features = self.output_projection(aggregated)
                
                sequence_features.append(features)
            
            # Stack features for this batch element
            sequence_features = torch.stack(sequence_features)  # [T-1, output_dim]
            all_features.append(sequence_features)
        
        # Stack all batch elements
        return torch.stack(all_features)  # [B, T-1, output_dim]


class RobustIMUEncoder(nn.Module):
    """
    A more robust IMU encoder that handles extreme variations in sample counts.
    Uses multiple processing strategies and learns to weight them.
    """
    
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=256, dropout=0.2):
        super().__init__()
        
        # Multiple processing paths
        # 1. Direct LSTM on raw sequence
        self.raw_lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True, 
                               bidirectional=True, dropout=dropout)
        
        # 2. Downsampled to fixed 50 samples
        self.fixed_lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True,
                                 bidirectional=True, dropout=dropout)
        
        # 3. Statistical features
        self.stats_proj = nn.Linear(input_dim * 4, hidden_dim * 2)  # mean, std, min, max
        
        # Attention to combine different representations
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, batch_first=True)
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def resample_to_fixed(self, imu_data: torch.Tensor, target_len: int = 50) -> torch.Tensor:
        """Resample IMU data to fixed length using linear interpolation."""
        current_len = imu_data.shape[0]
        
        if current_len == target_len:
            return imu_data
            
        # Interpolate each channel
        imu_data = imu_data.T.unsqueeze(0)  # [1, 6, current_len]
        resampled = F.interpolate(imu_data, size=target_len, mode='linear', align_corners=True)
        return resampled.squeeze(0).T  # [target_len, 6]
        
    def forward(self, imu_sequences: List[List[torch.Tensor]]) -> torch.Tensor:
        """Process IMU sequences using multiple strategies."""
        B = len(imu_sequences)
        T_minus_1 = len(imu_sequences[0])
        device = imu_sequences[0][0].device
        
        all_features = []
        
        for b in range(B):
            sequence_features = []
            
            for t in range(T_minus_1):
                imu_segment = imu_sequences[b][t]
                
                if imu_segment.shape[0] == 0:
                    features = torch.zeros(self.output_proj[0].in_features, device=device)
                else:
                    # Path 1: Raw LSTM
                    raw_out, (h_n, _) = self.raw_lstm(imu_segment.unsqueeze(0))
                    raw_feat = torch.cat([h_n[-2], h_n[-1]], dim=-1).squeeze(0)
                    
                    # Path 2: Fixed-length LSTM
                    resampled = self.resample_to_fixed(imu_segment)
                    fixed_out, (h_n, _) = self.fixed_lstm(resampled.unsqueeze(0))
                    fixed_feat = torch.cat([h_n[-2], h_n[-1]], dim=-1).squeeze(0)
                    
                    # Path 3: Statistical features
                    stats = torch.cat([
                        imu_segment.mean(dim=0),
                        imu_segment.std(dim=0),
                        imu_segment.min(dim=0)[0],
                        imu_segment.max(dim=0)[0]
                    ])
                    stats_feat = self.stats_proj(stats)
                    
                    # Combine with attention
                    features_stack = torch.stack([raw_feat, fixed_feat, stats_feat], dim=0).unsqueeze(0)
                    attended, _ = self.attention(features_stack, features_stack, features_stack)
                    features = attended.mean(dim=1).squeeze(0)
                
                features = self.output_proj(features)
                sequence_features.append(features)
                
            all_features.append(torch.stack(sequence_features))
            
        return torch.stack(all_features)