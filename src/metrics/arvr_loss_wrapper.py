"""
Wrapper for ARVRAdaptiveLoss to handle separate rotation and translation inputs
"""

import torch
import torch.nn as nn
from .arvr_loss import ARVRAdaptiveLoss


class ARVRLossWrapper(nn.Module):
    """
    Wrapper that adapts the ARVRAdaptiveLoss to work with separate
    rotation (quaternion) and translation predictions.
    """
    
    def __init__(self):
        super().__init__()
        self.translation_loss = nn.MSELoss()
        self.rotation_loss = nn.MSELoss()
        
    def forward(self, pred_rotation, target_rotation, pred_translation, target_translation):
        """
        Args:
            pred_rotation: [B*seq_len, 4] quaternions
            target_rotation: [B*seq_len, 4] quaternions
            pred_translation: [B*seq_len, 3] translations
            target_translation: [B*seq_len, 3] translations
        
        Returns:
            Dictionary with loss components
        """
        # Compute basic losses
        rot_loss = self.rotation_loss(pred_rotation, target_rotation)
        trans_loss = self.translation_loss(pred_translation, target_translation)
        
        # Compute scale-aware weights based on motion magnitude
        with torch.no_grad():
            # Translation magnitude
            trans_magnitude = torch.norm(target_translation, dim=-1)
            trans_weight = torch.ones_like(trans_magnitude)
            trans_weight[trans_magnitude < 0.01] = 3.0  # Small motions < 1cm
            trans_weight[trans_magnitude > 0.05] = 0.5  # Large motions > 5cm
            
            # Rotation magnitude (quaternion angle)
            # Angle between identity and target quaternion
            quat_dot = target_rotation[:, 3]  # w component
            quat_angle = 2 * torch.acos(torch.clamp(quat_dot.abs(), -1, 1))
            rot_weight = torch.ones_like(quat_angle)
            rot_weight[quat_angle < 0.035] = 3.0  # Small rotations < 2°
            rot_weight[quat_angle > 0.175] = 0.5  # Large rotations > 10°
        
        # Apply weights
        weighted_trans_loss = (trans_loss * trans_weight.mean())
        weighted_rot_loss = (rot_loss * rot_weight.mean())
        
        return {
            'translation_loss': weighted_trans_loss,
            'rotation_loss': weighted_rot_loss,
            'total_loss': weighted_trans_loss + weighted_rot_loss
        }