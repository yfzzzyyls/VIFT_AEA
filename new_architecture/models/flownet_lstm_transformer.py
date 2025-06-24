"""
FlowNet-LSTM-Transformer Architecture for Visual-Inertial Odometry
This is a completely new architecture that replaces VIFT with:
- FlowNet for visual motion extraction
- LSTM for variable-length IMU processing
- Transformer for multi-modal fusion and pose prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


class FlowNetLSTMTransformer(nn.Module):
    """
    Main model architecture combining FlowNet, LSTM, and Transformer
    for visual-inertial odometry on Aria dataset.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Visual encoder: FlowNet for motion extraction
        self.visual_encoder = FlowNetMotionEncoder(
            output_dim=config.visual_feature_dim
        )
        
        # IMU encoder: LSTM for temporal processing
        self.imu_encoder = IMULSTMEncoder(
            input_dim=6,  # ax, ay, az, gx, gy, gz
            hidden_dim=config.imu_hidden_dim,
            num_layers=config.imu_lstm_layers,
            output_dim=config.imu_feature_dim,
            dropout=config.dropout,
            bidirectional=config.imu_bidirectional
        )
        
        # Fusion and prediction: Transformer
        self.pose_predictor = PoseTransformer(
            visual_dim=config.visual_feature_dim,
            imu_dim=config.imu_feature_dim,
            d_model=config.transformer_dim,
            nhead=config.transformer_heads,
            num_layers=config.transformer_layers,
            dim_feedforward=config.transformer_feedforward,
            dropout=config.dropout
        )
        
        # Output heads for pose prediction
        self.translation_head = nn.Sequential(
            nn.Linear(config.transformer_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 3)
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(config.transformer_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 4)  # quaternion representation
        )
        
    def forward(self, images: torch.Tensor, imu_sequences: List[List[torch.Tensor]]) -> dict:
        """
        Forward pass through the model.
        
        Args:
            images: [B, T, 3, H, W] - Sequence of RGB images
            imu_sequences: List of B sequences, each containing T-1 variable-length IMU segments
                          imu_sequences[b][i] has shape [num_samples, 6]
        
        Returns:
            Dictionary containing:
                - 'poses': [B, T-1, 7] - Predicted poses (3 trans + 4 quat)
                - 'translation': [B, T-1, 3] - Translation predictions
                - 'rotation': [B, T-1, 4] - Rotation predictions (quaternion)
        """
        B, T = images.shape[:2]
        
        # Extract visual motion features between consecutive frames
        visual_features = self.visual_encoder(images)  # [B, T-1, visual_dim]
        
        # Process IMU sequences through LSTM
        imu_features = self.imu_encoder(imu_sequences)  # [B, T-1, imu_dim]
        
        # Fuse features and predict poses using transformer
        fused_features = self.pose_predictor(visual_features, imu_features)  # [B, T-1, transformer_dim]
        
        # Predict translation and rotation
        translation = self.translation_head(fused_features)  # [B, T-1, 3]
        rotation = self.rotation_head(fused_features)  # [B, T-1, 4]
        
        # Normalize quaternions
        rotation = F.normalize(rotation, p=2, dim=-1)
        
        # Combine into pose representation
        poses = torch.cat([translation, rotation], dim=-1)  # [B, T-1, 7]
        
        return {
            'poses': poses,
            'translation': translation,
            'rotation': rotation
        }


class FlowNetMotionEncoder(nn.Module):
    """
    FlowNet-based encoder for extracting motion features from image pairs.
    Uses a simplified FlowNet-C architecture with correlation layer.
    """
    
    def __init__(self, output_dim=256):
        super().__init__()
        
        # Shared convolutional backbone for feature extraction
        self.conv1 = self._conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = self._conv_block(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = self._conv_block(128, 256, kernel_size=5, stride=2, padding=2)
        
        # Correlation layer for motion computation
        self.corr = CorrelationLayer(max_displacement=10)
        
        # Processing after correlation
        self.conv_redir = self._conv_block(256, 32, kernel_size=1, stride=1, padding=0)
        self.conv3_1 = self._conv_block(473, 256, stride=1)  # 441 (corr) + 32 (redir)
        self.conv4 = self._conv_block(256, 512, stride=2)
        self.conv4_1 = self._conv_block(512, 512, stride=1)
        self.conv5 = self._conv_block(512, 512, stride=2)
        self.conv5_1 = self._conv_block(512, 512, stride=1)
        self.conv6 = self._conv_block(512, 1024, stride=2)
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )
        
    def _conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """Create a convolutional block with BN and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract motion features from consecutive image pairs.
        
        Args:
            images: [B, T, 3, H, W] - Sequence of images
            
        Returns:
            motion_features: [B, T-1, output_dim] - Motion features between consecutive frames
        """
        B, T, C, H, W = images.shape
        motion_features = []
        
        # Process consecutive frame pairs
        for i in range(T - 1):
            # Extract features from both frames
            feat1 = self.extract_features(images[:, i])  # [B, 256, H/8, W/8]
            feat2 = self.extract_features(images[:, i+1])
            
            # Compute correlation
            corr = self.corr(feat1, feat2)  # [B, 441, H/8, W/8]
            
            # Reduce feat1 channels and concatenate with correlation
            feat1_redir = self.conv_redir(feat1)  # [B, 32, H/8, W/8]
            concat_feat = torch.cat([corr, feat1_redir], dim=1)  # [B, 473, H/8, W/8]
            
            # Process through remaining layers
            out = self.conv3_1(concat_feat)
            out = self.conv4_1(self.conv4(out))
            out = self.conv5_1(self.conv5(out))
            out = self.conv6(out)
            
            # Global pooling and projection
            out = self.global_pool(out).squeeze(-1).squeeze(-1)  # [B, 1024]
            motion_feat = self.projection(out)  # [B, output_dim]
            
            motion_features.append(motion_feat)
            
        return torch.stack(motion_features, dim=1)  # [B, T-1, output_dim]
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract convolutional features from a single image."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class CorrelationLayer(nn.Module):
    """
    Correlation layer for FlowNet, computes correlation between two feature maps.
    """
    
    def __init__(self, max_displacement=10):
        super().__init__()
        self.max_displacement = max_displacement
        self.kernel_size = 2 * max_displacement + 1
        
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation between two feature maps.
        
        Args:
            feat1, feat2: [B, C, H, W] - Feature maps from two frames
            
        Returns:
            corr: [B, (2*max_disp+1)^2, H, W] - Correlation volume
        """
        B, C, H, W = feat1.shape
        
        # Pad feat2 for computing correlations at different displacements
        feat2_padded = F.pad(feat2, [self.max_displacement] * 4)
        
        # Compute correlation for each displacement
        corr_list = []
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                # Extract shifted version of feat2
                feat2_shifted = feat2_padded[:, :, i:i+H, j:j+W]
                
                # Compute dot product and normalize
                corr = (feat1 * feat2_shifted).sum(dim=1, keepdim=True) / C
                corr_list.append(corr)
                
        return torch.cat(corr_list, dim=1)  # [B, (2*max_disp+1)^2, H, W]


class IMULSTMEncoder(nn.Module):
    """
    LSTM encoder for processing variable-length IMU sequences.
    Handles all IMU samples between consecutive frames without downsampling.
    """
    
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=3, 
                 output_dim=256, dropout=0.2, bidirectional=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        
        # Input projection to increase feature dimension
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output projection
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, imu_sequences: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Process variable-length IMU sequences.
        
        Args:
            imu_sequences: List of B sequences, each containing T-1 variable-length IMU segments
                          imu_sequences[b][i] has shape [num_samples, 6]
                          
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
                
                # Handle empty segments (shouldn't happen but safety check)
                if imu_segment.shape[0] == 0:
                    features = torch.zeros(self.output_dim, device=device)
                else:
                    # Project input
                    projected = self.input_projection(imu_segment)  # [num_samples, 64]
                    projected = projected.unsqueeze(0)  # [1, num_samples, 64]
                    
                    # Apply LSTM
                    lstm_out, (h_n, c_n) = self.lstm(projected)
                    
                    # Use final hidden states from both directions
                    if self.bidirectional:
                        # h_n shape: [num_layers * 2, 1, hidden_dim]
                        h_forward = h_n[-2, 0, :]  # Last layer, forward
                        h_backward = h_n[-1, 0, :]  # Last layer, backward
                        hidden = torch.cat([h_forward, h_backward], dim=-1)
                    else:
                        hidden = h_n[-1, 0, :]  # [hidden_dim]
                    
                    # Project to output dimension
                    features = self.output_projection(hidden)  # [output_dim]
                
                sequence_features.append(features)
            
            # Stack features for this batch element
            sequence_features = torch.stack(sequence_features)  # [T-1, output_dim]
            all_features.append(sequence_features)
        
        # Stack all batch elements
        return torch.stack(all_features)  # [B, T-1, output_dim]


class PoseTransformer(nn.Module):
    """
    Transformer module for fusing visual and IMU features and predicting poses.
    Uses causal attention to ensure temporal consistency.
    """
    
    def __init__(self, visual_dim, imu_dim, d_model=512, nhead=8,
                 num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Project visual and IMU features to transformer dimension
        self.visual_projection = nn.Linear(visual_dim, d_model // 2)
        self.imu_projection = nn.Linear(imu_dim, d_model // 2)
        
        # Learnable modality embeddings
        self.visual_embedding = nn.Parameter(torch.randn(1, 1, d_model // 2))
        self.imu_embedding = nn.Parameter(torch.randn(1, 1, d_model // 2))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder with causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, visual_features: torch.Tensor, imu_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse visual and IMU features using transformer.
        
        Args:
            visual_features: [B, T-1, visual_dim] - Visual motion features
            imu_features: [B, T-1, imu_dim] - IMU temporal features
            
        Returns:
            fused_features: [B, T-1, d_model] - Fused features for pose prediction
        """
        B, T_minus_1 = visual_features.shape[:2]
        
        # Project features
        visual_proj = self.visual_projection(visual_features)  # [B, T-1, d_model/2]
        imu_proj = self.imu_projection(imu_features)  # [B, T-1, d_model/2]
        
        # Add modality embeddings
        visual_proj = visual_proj + self.visual_embedding
        imu_proj = imu_proj + self.imu_embedding
        
        # Concatenate modalities
        combined = torch.cat([visual_proj, imu_proj], dim=-1)  # [B, T-1, d_model]
        
        # Add positional encoding
        combined = self.pos_encoder(combined)
        
        # Generate causal mask
        causal_mask = self._generate_causal_mask(T_minus_1).to(combined.device)
        
        # Apply transformer with causal mask
        transformer_out = self.transformer(combined, mask=causal_mask, is_causal=True)
        
        # Final projection
        output = self.output_projection(transformer_out)
        
        return output
    
    def _generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Generate a causal mask for the transformer."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)