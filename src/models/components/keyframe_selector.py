"""
Keyframe Selection Module for Multi-Frame Visual Odometry
ORB-SLAM3 inspired covisibility-based keyframe selection with temporal guards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from .feature_bank import FeatureBank


class KeyFrameSelector:
    """
    ORB-SLAM3 inspired covisibility-based keyframe selection.
    
    Key features:
    - Normalized covisibility scores for scenes with varying feature density
    - Temporal guard to prevent keyframe starvation during rapid motion
    - Efficient feature correlation as proxy for geometric overlap
    """
    
    def __init__(self, temporal_guard_frames: int = 30, 
                 min_covisibility_score: float = 0.3):
        """
        Initialize keyframe selector.
        
        Args:
            temporal_guard_frames: Force keyframe insertion every N frames
            min_covisibility_score: Minimum score to consider frames covisible
        """
        self.temporal_guard = temporal_guard_frames
        self.min_covisibility_score = min_covisibility_score
        self.last_keyframe_id = -1
        
        # Statistics
        self.total_keyframes = 0
        self.forced_keyframes = 0
        
    def compute_covisibility(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """
        Compute normalized covisibility score between two feature maps.
        
        Uses cosine similarity of global feature statistics as a proxy for
        geometric overlap. Normalized by valid feature count to handle
        scenes with varying feature density.
        
        Args:
            feat1: First feature map [C, H, W]
            feat2: Second feature map [C, H, W]
            
        Returns:
            Normalized covisibility score in [0, 1]
        """
        # Global pooling to get feature statistics
        # Use both mean and std for more robust matching
        feat1_mean = feat1.flatten(1).mean(dim=1)  # [C]
        feat1_std = feat1.flatten(1).std(dim=1)    # [C]
        feat1_global = torch.cat([feat1_mean, feat1_std])  # [2*C]
        
        feat2_mean = feat2.flatten(1).mean(dim=1)
        feat2_std = feat2.flatten(1).std(dim=1)
        feat2_global = torch.cat([feat2_mean, feat2_std])
        
        # Cosine similarity
        score = F.cosine_similarity(feat1_global.unsqueeze(0), 
                                  feat2_global.unsqueeze(0), dim=1)
        
        # Count valid features (non-zero response)
        valid_pixels1 = (feat1.abs().sum(dim=0) > 1e-6).float().sum()
        valid_pixels2 = (feat2.abs().sum(dim=0) > 1e-6).float().sum()
        
        # Normalize by minimum valid pixel count (ORB-SLAM3 style)
        # This prevents bias against frames with fewer features
        min_valid = torch.min(valid_pixels1, valid_pixels2)
        normalized_score = score * (min_valid / (feat1.shape[1] * feat1.shape[2])).sqrt()
        
        return normalized_score.item()
        
    def compute_spatial_overlap(self, feat1: torch.Tensor, feat2: torch.Tensor, 
                               grid_size: int = 4) -> float:
        """
        Compute spatial overlap using grid-based correlation.
        
        Divides features into spatial cells and computes local correlations,
        providing more fine-grained overlap estimation.
        
        Args:
            feat1: First feature map [C, H, W]
            feat2: Second feature map [C, H, W]
            grid_size: Number of grid cells per dimension
            
        Returns:
            Spatial overlap score
        """
        C, H, W = feat1.shape
        cell_h = H // grid_size
        cell_w = W // grid_size
        
        overlap_scores = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Extract cell
                h_start = i * cell_h
                h_end = (i + 1) * cell_h if i < grid_size - 1 else H
                w_start = j * cell_w
                w_end = (j + 1) * cell_w if j < grid_size - 1 else W
                
                cell1 = feat1[:, h_start:h_end, w_start:w_end]
                cell2 = feat2[:, h_start:h_end, w_start:w_end]
                
                # Compute cell correlation
                if cell1.numel() > 0 and cell2.numel() > 0:
                    cell_score = F.cosine_similarity(
                        cell1.flatten().unsqueeze(0),
                        cell2.flatten().unsqueeze(0)
                    ).item()
                    overlap_scores.append(cell_score)
                    
        return np.mean(overlap_scores) if overlap_scores else 0.0
        
    def select_keyframes(self, current_id: int, current_features: torch.Tensor,
                        feature_bank: FeatureBank, top_k: int = 3,
                        use_spatial_overlap: bool = False) -> List[int]:
        """
        Select keyframes based on covisibility and temporal constraints.
        
        Args:
            current_id: Current frame ID
            current_features: Current frame features [C, H, W] or [B, C, H, W]
            feature_bank: Feature bank instance
            top_k: Number of keyframes to select
            use_spatial_overlap: Whether to use fine-grained spatial overlap
            
        Returns:
            List of selected keyframe IDs
        """
        # Handle batch dimension
        if current_features.dim() == 4:
            current_features = current_features[0]
            
        # Check temporal guard - force keyframe if needed
        frames_since_last_kf = current_id - self.last_keyframe_id
        if frames_since_last_kf >= self.temporal_guard:
            self.last_keyframe_id = current_id
            self.total_keyframes += 1
            self.forced_keyframes += 1
            
            # Add as keyframe
            feature_bank.add_features(current_id, current_features)
            
            # If this is the first or forced keyframe, return empty
            if len(feature_bank) <= 1:
                return []
                
        # Compute covisibility scores with all cached frames
        scores = []
        
        for (cam_id, frame_id) in feature_bank.features.keys():
            if frame_id == current_id:
                continue
                
            # Get cached features
            kf_features_list = feature_bank.get_features([frame_id])
            if len(kf_features_list) == 0:
                continue
            kf_features = kf_features_list[0]
            
            # Ensure both features are on same device
            if current_features.device != kf_features.device:
                kf_features = kf_features.to(current_features.device)
            
            # Compute covisibility
            if use_spatial_overlap:
                score = self.compute_spatial_overlap(current_features, kf_features)
            else:
                score = self.compute_covisibility(current_features, kf_features)
                
            # Only consider frames above minimum threshold
            if score >= self.min_covisibility_score:
                scores.append((frame_id, score))
                
                # Update covisibility graph
                feature_bank.update_covisibility(current_id, frame_id, score)
                
        # Sort by covisibility score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top-k keyframes
        keyframe_ids = [fid for fid, score in scores[:top_k]]
        
        # If current frame has high covisibility, consider it as keyframe
        if scores and scores[0][1] > 0.8:  # High overlap threshold
            if current_id != self.last_keyframe_id:
                self.last_keyframe_id = current_id
                self.total_keyframes += 1
                feature_bank.add_features(current_id, current_features)
                
        return keyframe_ids
        
    def should_add_keyframe(self, current_id: int, avg_covisibility: float,
                           motion_magnitude: Optional[float] = None) -> bool:
        """
        Determine if current frame should be added as keyframe.
        
        Uses multiple criteria:
        - Temporal guard (force keyframe every N frames)
        - Low covisibility (new area being explored)
        - High motion (rapid movement)
        
        Args:
            current_id: Current frame ID
            avg_covisibility: Average covisibility with existing keyframes
            motion_magnitude: Optional motion magnitude estimate
            
        Returns:
            True if frame should be added as keyframe
        """
        # Temporal guard
        if current_id - self.last_keyframe_id >= self.temporal_guard:
            return True
            
        # Low covisibility (exploring new area)
        if avg_covisibility < 0.4:
            return True
            
        # High motion (if provided)
        if motion_magnitude is not None and motion_magnitude > 0.5:
            return True
            
        return False
        
    def get_statistics(self) -> dict:
        """Get keyframe selection statistics."""
        return {
            'total_keyframes': self.total_keyframes,
            'forced_keyframes': self.forced_keyframes,
            'forced_ratio': self.forced_keyframes / max(1, self.total_keyframes),
            'last_keyframe_id': self.last_keyframe_id
        }