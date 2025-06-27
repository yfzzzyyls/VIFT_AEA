"""
Feature Tracker for MSCKF using SEA-RAFT
Extracts and tracks sparse features from SEA-RAFT's correlation volumes.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2


@dataclass 
class Feature:
    """Single feature track."""
    id: int
    observations: Dict[float, np.ndarray]  # timestamp -> [u, v] pixel coords
    descriptor: Optional[np.ndarray] = None
    position_3d: Optional[np.ndarray] = None  # Triangulated position
    
    def track_length(self) -> int:
        return len(self.observations)
        

class MSCKFFeatureTracker:
    """
    Manages feature tracks for MSCKF update.
    Integrates with SEA-RAFT multi-frame correlation.
    
    Key features:
    - Extracts sparse features from correlation volumes
    - Tracks features across frames using correlation
    - Manages feature lifecycle (creation, tracking, removal)
    """
    
    def __init__(self, 
                 max_features: int = 150,
                 min_track_length: int = 3,
                 feature_threshold: float = 0.5,
                 nms_radius: int = 15,
                 max_tracking_distance: float = 30.0):
        """
        Initialize feature tracker.
        
        Args:
            max_features: Maximum number of features to track
            min_track_length: Minimum track length for MSCKF update
            feature_threshold: Correlation threshold for feature detection
            nms_radius: Non-max suppression radius in pixels
            max_tracking_distance: Maximum pixel distance for matching
        """
        self.max_features = max_features
        self.min_track_length = min_track_length
        self.feature_threshold = feature_threshold
        self.nms_radius = nms_radius
        self.max_tracking_distance = max_tracking_distance
        
        # Active features
        self.features: Dict[int, Feature] = {}
        self.next_feature_id = 0
        
        # Previous frame data for tracking
        self.prev_frame_data = None
        self.prev_timestamp = None
        
    def extract_features_from_correlation(self, 
                                        correlation_volume: torch.Tensor,
                                        timestamp: float,
                                        image_shape: Tuple[int, int]) -> List[np.ndarray]:
        """
        Extract sparse features from SEA-RAFT correlation volume.
        
        Args:
            correlation_volume: [B, C, H, W] correlation from SEA-RAFT
            timestamp: Current frame timestamp
            image_shape: Original image (H, W) for coordinate scaling
            
        Returns:
            List of feature locations [u, v] in image coordinates
        """
        # Handle batch dimension
        if correlation_volume.dim() == 4:
            correlation_volume = correlation_volume[0]  # Take first batch
            
        # Average across correlation channels to get confidence map
        if correlation_volume.shape[0] > 1:
            confidence_map = correlation_volume.mean(dim=0)
        else:
            confidence_map = correlation_volume[0]
            
        # Convert to numpy
        confidence_map = confidence_map.detach().cpu().numpy()
        
        # Normalize to [0, 1]
        conf_min, conf_max = confidence_map.min(), confidence_map.max()
        if conf_max > conf_min:
            confidence_map = (confidence_map - conf_min) / (conf_max - conf_min)
        else:
            confidence_map = np.zeros_like(confidence_map)
            
        # Non-maximum suppression
        features = self._non_max_suppression(confidence_map)
        
        # Scale to image coordinates
        corr_h, corr_w = confidence_map.shape
        img_h, img_w = image_shape
        scale_h = img_h / corr_h
        scale_w = img_w / corr_w
        
        scaled_features = []
        for u, v in features:
            u_img = u * scale_w
            v_img = v * scale_h
            scaled_features.append(np.array([u_img, v_img]))
            
        return scaled_features[:self.max_features]
        
    def _non_max_suppression(self, confidence_map: np.ndarray) -> List[List[int]]:
        """
        Apply non-maximum suppression to extract sparse features.
        
        Args:
            confidence_map: 2D confidence map
            
        Returns:
            List of [u, v] feature locations
        """
        # Apply threshold
        mask = confidence_map > self.feature_threshold
        
        # Find local maxima
        kernel_size = 2 * self.nms_radius + 1
        
        # Use max pooling to find local maxima
        h, w = confidence_map.shape
        padded = np.pad(confidence_map, self.nms_radius, mode='constant', constant_values=0)
        
        features = []
        for y in range(h):
            for x in range(w):
                if not mask[y, x]:
                    continue
                    
                # Check if local maximum
                y_pad = y + self.nms_radius
                x_pad = x + self.nms_radius
                neighborhood = padded[y_pad-self.nms_radius:y_pad+self.nms_radius+1,
                                    x_pad-self.nms_radius:x_pad+self.nms_radius+1]
                
                if confidence_map[y, x] >= neighborhood.max():
                    features.append([x, y])  # [u, v] format
                    
        # Sort by confidence
        features.sort(key=lambda f: confidence_map[f[1], f[0]], reverse=True)
        
        return features
        
    def track_features(self, 
                      correlation_volume: torch.Tensor,
                      timestamp: float,
                      image_shape: Tuple[int, int],
                      camera_matrix: Optional[np.ndarray] = None) -> Dict[int, Feature]:
        """
        Track features using correlation volume from SEA-RAFT.
        
        Args:
            correlation_volume: Current correlation volume
            timestamp: Current timestamp
            image_shape: Image dimensions
            camera_matrix: Optional camera intrinsics for undistortion
            
        Returns:
            Dictionary of tracked features
        """
        # Extract features from current frame
        current_features = self.extract_features_from_correlation(
            correlation_volume, timestamp, image_shape
        )
        
        # First frame - just create new features
        if self.prev_frame_data is None:
            for feat_loc in current_features:
                self._create_new_feature(feat_loc, timestamp)
                
            self.prev_frame_data = current_features
            self.prev_timestamp = timestamp
            return self.features
            
        # Match with previous features
        matched_pairs = self._match_features(self.prev_frame_data, current_features)
        
        # Update existing tracks
        matched_current = set()
        for prev_idx, curr_idx in matched_pairs:
            # Find which feature ID this corresponds to
            for feat_id, feature in self.features.items():
                if self.prev_timestamp in feature.observations:
                    prev_obs = feature.observations[self.prev_timestamp]
                    if np.allclose(prev_obs, self.prev_frame_data[prev_idx], atol=1.0):
                        # Update track
                        feature.observations[timestamp] = current_features[curr_idx]
                        matched_current.add(curr_idx)
                        break
                        
        # Create new features for unmatched detections
        for i, feat_loc in enumerate(current_features):
            if i not in matched_current and len(self.features) < self.max_features:
                self._create_new_feature(feat_loc, timestamp)
                
        # Remove lost features
        self._remove_lost_features(timestamp)
        
        # Update previous frame data
        self.prev_frame_data = current_features
        self.prev_timestamp = timestamp
        
        return self.features
        
    def _match_features(self, 
                       prev_features: List[np.ndarray],
                       curr_features: List[np.ndarray]) -> List[Tuple[int, int]]:
        """
        Match features between frames using nearest neighbor.
        
        Returns:
            List of (prev_idx, curr_idx) matched pairs
        """
        if not prev_features or not curr_features:
            return []
            
        matches = []
        used_curr = set()
        
        # Simple nearest neighbor matching
        for i, prev_feat in enumerate(prev_features):
            best_dist = float('inf')
            best_j = -1
            
            for j, curr_feat in enumerate(curr_features):
                if j in used_curr:
                    continue
                    
                dist = np.linalg.norm(prev_feat - curr_feat)
                if dist < best_dist and dist < self.max_tracking_distance:
                    best_dist = dist
                    best_j = j
                    
            if best_j >= 0:
                matches.append((i, best_j))
                used_curr.add(best_j)
                
        return matches
        
    def _create_new_feature(self, location: np.ndarray, timestamp: float):
        """Create new feature track."""
        feature = Feature(
            id=self.next_feature_id,
            observations={timestamp: location}
        )
        self.features[self.next_feature_id] = feature
        self.next_feature_id += 1
        
    def _remove_lost_features(self, current_timestamp: float):
        """Remove features that weren't observed in current frame."""
        to_remove = []
        for feat_id, feature in self.features.items():
            if current_timestamp not in feature.observations:
                to_remove.append(feat_id)
                
        for feat_id in to_remove:
            del self.features[feat_id]
            
    def get_tracks_for_update(self, min_length: Optional[int] = None) -> List[Feature]:
        """
        Get feature tracks ready for MSCKF update.
        
        Args:
            min_length: Minimum track length (default: self.min_track_length)
            
        Returns:
            List of features with sufficient track length
        """
        if min_length is None:
            min_length = self.min_track_length
            
        ready_features = []
        for feature in self.features.values():
            if feature.track_length() >= min_length:
                ready_features.append(feature)
                
        return ready_features
        
    def remove_features(self, feature_ids: List[int]):
        """Remove features after MSCKF update."""
        for feat_id in feature_ids:
            if feat_id in self.features:
                del self.features[feat_id]
                
    def get_feature_observations(self, feature_id: int) -> List[Tuple[float, np.ndarray]]:
        """
        Get all observations of a feature sorted by timestamp.
        
        Returns:
            List of (timestamp, [u, v]) tuples
        """
        if feature_id not in self.features:
            return []
            
        feature = self.features[feature_id]
        observations = []
        for timestamp in sorted(feature.observations.keys()):
            observations.append((timestamp, feature.observations[timestamp]))
            
        return observations
        
    def visualize_tracks(self, image: np.ndarray, current_timestamp: float) -> np.ndarray:
        """
        Visualize feature tracks on image.
        
        Args:
            image: Current frame
            current_timestamp: Current timestamp
            
        Returns:
            Image with drawn feature tracks
        """
        vis_img = image.copy()
        
        # Define colors based on track length
        def get_color(track_length):
            if track_length < 3:
                return (0, 0, 255)  # Red - new
            elif track_length < 5:
                return (0, 255, 255)  # Yellow - medium
            else:
                return (0, 255, 0)  # Green - long
                
        # Draw features
        for feature in self.features.values():
            if current_timestamp in feature.observations:
                loc = feature.observations[current_timestamp].astype(int)
                color = get_color(feature.track_length())
                
                # Draw circle
                cv2.circle(vis_img, tuple(loc), 5, color, -1)
                
                # Draw track history
                timestamps = sorted(feature.observations.keys())
                if len(timestamps) > 1:
                    points = []
                    for ts in timestamps[-5:]:  # Last 5 observations
                        if ts in feature.observations:
                            points.append(feature.observations[ts].astype(int))
                            
                    if len(points) > 1:
                        pts = np.array(points, dtype=np.int32)
                        cv2.polylines(vis_img, [pts], False, color, 1)
                        
                # Draw feature ID
                cv2.putText(vis_img, str(feature.id), 
                           tuple(loc + np.array([7, -7])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                           
        # Draw statistics
        stats_text = f"Features: {len(self.features)} | Min length: {self.min_track_length}"
        cv2.putText(vis_img, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                   
        return vis_img
        
    def reset(self):
        """Reset tracker state."""
        self.features.clear()
        self.next_feature_id = 0
        self.prev_frame_data = None
        self.prev_timestamp = None