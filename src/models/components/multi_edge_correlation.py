"""
Multi-Edge Correlation Module for Multi-Frame Visual Odometry
Computes correlation volumes between current frame and multiple keyframes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import List, Optional, Tuple

# Add SEA-RAFT to path for CorrBlock
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'third_party', 'SEA-RAFT'))
from core.corr import CorrBlock


class MultiEdgeCorrelation(nn.Module):
    """
    Compute multi-frame correlation volumes using RAFT's native CorrBlock.
    
    Key features:
    - Reuses RAFT's optimized correlation implementation
    - Weighted aggregation based on correlation confidence
    - Mixed precision support for memory efficiency
    """
    
    def __init__(self, corr_radius: int = 4, corr_levels: int = 4):
        """
        Initialize multi-edge correlation module.
        
        Args:
            corr_radius: Correlation search radius (default 4 = ±64px after 3 updates)
            corr_levels: Number of correlation pyramid levels
        """
        super().__init__()
        
        # Create args object for CorrBlock
        self.corr_args = type('Args', (), {
            'corr_radius': corr_radius,
            'corr_levels': corr_levels,
        })()
        
        # Learnable temperature for confidence weighting
        self.confidence_temperature = nn.Parameter(torch.tensor(0.1))
        
        # Optional learnable fusion weights
        self.fusion_weight = nn.Parameter(torch.tensor(0.1))
        
    def compute_correlation(self, fmap1: torch.Tensor, fmap2: torch.Tensor) -> CorrBlock:
        """
        Create correlation block between two feature maps.
        
        Args:
            fmap1: First feature map [B, C, H, W]
            fmap2: Second feature map [B, C, H, W]
            
        Returns:
            CorrBlock instance
        """
        return CorrBlock(fmap1, fmap2, self.corr_args)
        
    def sample_correlation(self, corr_block: CorrBlock, 
                          coords: Optional[torch.Tensor] = None,
                          stride: int = 8) -> torch.Tensor:
        """
        Sample correlation volume at given coordinates.
        
        Args:
            corr_block: CorrBlock instance
            coords: Flow coordinates [B, 2, H, W] (default: sparse grid)
            stride: Downsampling stride for sparse lookup (default: 8)
            
        Returns:
            Sampled correlation features [B, levels*9, h, w] where h,w = H/stride, W/stride
        """
        if coords is None:
            # Create sparse query grid instead of dense
            B = corr_block.corr_pyramid[0].shape[0]
            H, W = corr_block.corr_pyramid[0].shape[2:4]
            
            # Create base coordinate grid
            base_coords = torch.stack(
                torch.meshgrid(
                    torch.arange(H, device=corr_block.corr_pyramid[0].device, dtype=torch.float32),
                    torch.arange(W, device=corr_block.corr_pyramid[0].device, dtype=torch.float32),
                    indexing='ij'
                ), dim=0
            )  # [2, H, W]
            
            # Downsample by stride for sparse lookup (DROID-SLAM style)
            coords = base_coords[:, ::stride, ::stride]  # [2, H/stride, W/stride]
            coords = coords.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H/stride, W/stride]
            
        return corr_block(coords)
        
    def compute_correlation_confidence(self, corr_volume: torch.Tensor) -> torch.Tensor:
        """
        Extract confidence score from correlation volume.
        
        Higher correlation magnitude indicates better feature matching.
        
        Args:
            corr_volume: Correlation features [B, C, H, W]
            
        Returns:
            Confidence map [B, 1, H, W]
        """
        # Average magnitude across correlation channels
        # Higher values indicate stronger correlation
        confidence = corr_volume.abs().mean(dim=1, keepdim=True)
        
        # Optional: Apply softplus for smoother gradients
        confidence = F.softplus(confidence)
        
        return confidence
        
    def aggregate_multi_edge_flow(self, current_feat: torch.Tensor, 
                                 keyframe_feats: List[torch.Tensor],
                                 coords: Optional[torch.Tensor] = None,
                                 return_weights: bool = False) -> torch.Tensor:
        """
        Weighted aggregation of multi-edge correlations using direct dot-product.
        
        Following StreamFlow's SIM approach for memory efficiency.
        
        Args:
            current_feat: Current frame features [B, C, H, W]
            keyframe_feats: List of keyframe features, each [C, H, W] or [B, C, H, W]
            coords: Optional flow coordinates (not used in dot-product method)
            return_weights: Whether to return aggregation weights
            
        Returns:
            Aggregated correlation features [B, C_out, H, W]
            Optional: weights [K, B, 1, H, W] if return_weights=True
        """
        if len(keyframe_feats) == 0:
            return None
            
        batch_size = current_feat.shape[0]
        device = current_feat.device
        
        # Ensure keyframe features have batch dimension
        processed_kf_feats = []
        for kf_feat in keyframe_feats:
            if kf_feat.dim() == 3:
                # Add batch dimension
                kf_feat = kf_feat.unsqueeze(0).expand(batch_size, -1, -1, -1)
            processed_kf_feats.append(kf_feat)
        
        # Option B: Direct dot-product correlation (StreamFlow style)
        corr_volumes = []
        confidence_scores = []
        
        for kf_feat in processed_kf_feats:
            # Simple dot product correlation
            corr = (current_feat * kf_feat).sum(dim=1, keepdim=True)  # [B, 1, H, W]
            
            # Downsample to save memory
            corr = F.avg_pool2d(corr, kernel_size=4, stride=4)  # [B, 1, H/4, W/4]
            
            # Normalize correlation
            corr = corr / (current_feat.shape[1] ** 0.5)
            
            # Extract confidence (mean absolute correlation)
            confidence = corr.abs().mean(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
            
            corr_volumes.append(corr)
            confidence_scores.append(confidence)
            
        # Stack for batch processing
        corr_stack = torch.stack(corr_volumes, dim=0)  # [K, B, 1, H/4, W/4]
        confidence_stack = torch.stack(confidence_scores, dim=0)  # [K, B, 1, 1, 1]
        
        # Compute attention weights based on confidence
        weights = F.softmax(confidence_stack / self.confidence_temperature, dim=0)
        
        # Weighted aggregation
        weighted_corr = (corr_stack * weights).sum(dim=0)  # [B, 1, H/4, W/4]
        
        # Upsample back to original resolution
        weighted_corr = F.interpolate(weighted_corr, 
                                    size=(current_feat.shape[2], current_feat.shape[3]),
                                    mode='bilinear', 
                                    align_corners=False)
        
        if return_weights:
            return weighted_corr, weights
        return weighted_corr
        
    def forward(self, current_feat: torch.Tensor, 
                keyframe_feats: List[torch.Tensor],
                direct_motion: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass combining direct and multi-frame correlations.
        
        Args:
            current_feat: Current frame features [B, C, H, W]
            keyframe_feats: List of keyframe features
            direct_motion: Optional direct motion features [B, C, H, W]
            
        Returns:
            Combined motion features [B, C, H, W]
        """
        # Get multi-frame correlation
        multi_corr = self.aggregate_multi_edge_flow(current_feat, keyframe_feats)
        
        if multi_corr is None:
            # No keyframes available, return direct motion if provided
            return direct_motion if direct_motion is not None else None
            
        # If we have direct motion, combine with multi-frame
        if direct_motion is not None:
            # Learnable weighted combination
            combined = direct_motion + self.fusion_weight * multi_corr
            return combined
        else:
            return multi_corr
            
    def get_memory_usage(self, batch_size: int, image_size: Tuple[int, int],
                        num_keyframes: int) -> dict:
        """
        Estimate memory usage for given configuration.
        
        Args:
            batch_size: Batch size
            image_size: Input image size (H, W)
            num_keyframes: Number of keyframes
            
        Returns:
            Dictionary with memory estimates
        """
        H, W = image_size
        # Feature maps are typically 1/8 resolution
        feat_h, feat_w = H // 8, W // 8
        
        # Correlation volume size per edge
        corr_channels = self.corr_args.corr_levels * (2 * self.corr_args.corr_radius + 1) ** 2
        corr_volume_size = batch_size * corr_channels * feat_h * feat_w * 4  # float32
        
        # Total for all edges
        total_corr_memory = corr_volume_size * num_keyframes
        
        # Convert to MB
        memory_mb = total_corr_memory / (1024 * 1024)
        
        return {
            'corr_volume_per_edge_mb': corr_volume_size / (1024 * 1024),
            'total_correlation_mb': memory_mb,
            'num_keyframes': num_keyframes,
            'correlation_channels': corr_channels
        }