"""
Feature Bank Module for Multi-Frame Visual Odometry
Efficient circular buffer for caching SEA-RAFT features with automatic FIFO eviction.
"""

from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict


class FeatureBank:
    """
    Efficient circular buffer for caching SEA-RAFT features.
    
    Features:
    - OrderedDict for O(1) FIFO eviction
    - CPU storage to save GPU memory
    - Support for future stereo extension via (cam_id, frame_id) keys
    - Covisibility graph for geometric relationships
    """
    
    def __init__(self, max_frames: int = 100, feature_dim: int = 256, 
                 spatial_size: Tuple[int, int] = (88, 64)):
        """
        Initialize feature bank.
        
        Args:
            max_frames: Maximum number of frames to cache
            feature_dim: Feature channel dimension
            spatial_size: Spatial dimensions of features (H, W)
        """
        self.max_frames = max_frames
        self.feature_dim = feature_dim
        self.spatial_size = spatial_size
        
        # Core storage
        self.features = OrderedDict()  # (cam_id, frame_id) -> features
        self.positions = {}  # frame_id -> 6DoF pose [x,y,z,qx,qy,qz,qw]
        self.timestamps = {}  # frame_id -> timestamp
        
        # Covisibility graph: frame_id -> [(frame_id, score)]
        self.covisibility_graph = {}
        self.max_covis_connections = 10  # Limit connections per frame
        
    def add_features(self, frame_id: int, features: torch.Tensor, 
                    position: Optional[torch.Tensor] = None,
                    timestamp: Optional[float] = None,
                    cam_id: int = 0):
        """
        Add features to bank with automatic FIFO eviction.
        
        Args:
            frame_id: Unique frame identifier
            features: Feature tensor [B, C, H, W] or [C, H, W]
            position: Optional 6DoF pose [7]
            timestamp: Optional timestamp
            cam_id: Camera ID (default 0 for monocular)
        """
        # Handle batch dimension
        if features.dim() == 4 and features.size(0) == 1:
            features = features.squeeze(0)
        elif features.dim() == 4:
            # If batch size > 1, only take first
            features = features[0]
            
        # Validate dimensions
        assert features.dim() == 3, f"Expected 3D features, got {features.dim()}D"
        assert features.size(0) == self.feature_dim, \
            f"Feature dim mismatch: {features.size(0)} vs {self.feature_dim}"
            
        # Store in CPU memory to save GPU
        key = (cam_id, frame_id)
        self.features[key] = features.detach().cpu()
        
        # Store metadata
        if position is not None:
            self.positions[frame_id] = position.detach().cpu()
        if timestamp is not None:
            self.timestamps[frame_id] = timestamp
            
        # FIFO eviction
        while len(self.features) > self.max_frames:
            evicted_key = next(iter(self.features))
            evicted_frame_id = evicted_key[1]
            
            # Remove from all storage
            self.features.popitem(last=False)
            self.positions.pop(evicted_frame_id, None)
            self.timestamps.pop(evicted_frame_id, None)
            
            # Clean up covisibility graph
            self.covisibility_graph.pop(evicted_frame_id, None)
            for connections in self.covisibility_graph.values():
                connections[:] = [(fid, score) for fid, score in connections 
                                 if fid != evicted_frame_id]
                                 
    def get_features(self, frame_ids: List[int], cam_id: int = 0) -> List[torch.Tensor]:
        """
        Retrieve features, moving to GPU on demand.
        
        Args:
            frame_ids: List of frame IDs to retrieve
            cam_id: Camera ID (default 0)
            
        Returns:
            List of feature tensors on GPU
        """
        features = []
        for fid in frame_ids:
            key = (cam_id, fid)
            if key in self.features:
                # Move to GPU on demand
                feat = self.features[key].cuda()
                features.append(feat)
        return features
        
    def get_nearest_keyframes(self, current_id: int, k: int = 5) -> List[int]:
        """
        Get k nearest keyframes based on covisibility.
        
        Args:
            current_id: Current frame ID
            k: Number of keyframes to return
            
        Returns:
            List of keyframe IDs sorted by covisibility score
        """
        if current_id not in self.covisibility_graph:
            # If no covisibility info, return most recent frames
            all_frame_ids = [key[1] for key in self.features.keys() 
                           if key[1] != current_id]
            return all_frame_ids[-k:]  # Most recent k frames
            
        # Sort by covisibility score
        connections = self.covisibility_graph[current_id]
        connections.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k frame IDs
        return [fid for fid, _ in connections[:k]]
        
    def update_covisibility(self, frame_id1: int, frame_id2: int, score: float):
        """
        Update covisibility graph with bidirectional connection.
        
        Args:
            frame_id1: First frame ID
            frame_id2: Second frame ID
            score: Covisibility score (higher is better)
        """
        # Add bidirectional connection
        for fid1, fid2 in [(frame_id1, frame_id2), (frame_id2, frame_id1)]:
            if fid1 not in self.covisibility_graph:
                self.covisibility_graph[fid1] = []
                
            # Update or add connection
            updated = False
            for i, (existing_fid, existing_score) in enumerate(self.covisibility_graph[fid1]):
                if existing_fid == fid2:
                    self.covisibility_graph[fid1][i] = (fid2, score)
                    updated = True
                    break
                    
            if not updated:
                self.covisibility_graph[fid1].append((fid2, score))
                
            # Keep only top connections to save memory
            if len(self.covisibility_graph[fid1]) > self.max_covis_connections:
                self.covisibility_graph[fid1].sort(key=lambda x: x[1], reverse=True)
                self.covisibility_graph[fid1] = self.covisibility_graph[fid1][:self.max_covis_connections]
                
    def clear(self):
        """Clear all stored features and metadata."""
        self.features.clear()
        self.positions.clear()
        self.timestamps.clear()
        self.covisibility_graph.clear()
        
    def __len__(self):
        """Return number of cached frames."""
        return len(self.features)
        
    def get_memory_usage_mb(self) -> float:
        """
        Estimate memory usage in MB.
        
        Returns:
            Estimated memory usage in megabytes
        """
        if len(self.features) == 0:
            return 0.0
            
        # Feature memory: num_frames * C * H * W * 4 bytes (float32)
        h, w = self.spatial_size
        feature_bytes = len(self.features) * self.feature_dim * h * w * 4
        
        # Metadata memory (rough estimate)
        metadata_bytes = len(self.positions) * 7 * 4  # 7 DoF * 4 bytes
        metadata_bytes += len(self.timestamps) * 8  # 8 bytes per float
        metadata_bytes += len(self.covisibility_graph) * self.max_covis_connections * 12  # Rough estimate
        
        total_mb = (feature_bytes + metadata_bytes) / (1024 * 1024)
        return total_mb