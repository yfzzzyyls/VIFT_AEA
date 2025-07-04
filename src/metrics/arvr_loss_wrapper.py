"""
Wrapper for ARVRAdaptiveLoss to handle separate rotation and translation inputs
"""

import torch
import torch.nn as nn
from .arvr_loss import ARVRAdaptiveLoss


class ProperQuaternionLoss(nn.Module):
    """
    Proper quaternion loss that accounts for double cover and uses geodesic distance.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_q, target_q):
        """
        Compute quaternion loss using proper geodesic distance.
        
        Args:
            pred_q: [B, 4] predicted quaternions (XYZW)
            target_q: [B, 4] target quaternions (XYZW)
        
        Returns:
            Scalar loss
        """
        # Normalize quaternions
        pred_q = pred_q / (torch.norm(pred_q, dim=-1, keepdim=True) + 1e-8)
        target_q = target_q / (torch.norm(target_q, dim=-1, keepdim=True) + 1e-8)
        
        # Compute dot product
        dot = torch.sum(pred_q * target_q, dim=-1)
        
        # Handle double cover: if dot < 0, flip the predicted quaternion
        # This ensures we're always taking the shorter path
        mask = dot < 0
        pred_q[mask] = -pred_q[mask]
        dot[mask] = -dot[mask]
        
        # Clamp to avoid numerical issues with acos
        dot = torch.clamp(dot, -1.0, 1.0)
        
        # Geodesic distance: angle = 2 * arccos(|dot|)
        # Use exact geodesic distance instead of approximation
        # This is crucial for small angles in AR/VR applications
        loss = 2.0 * torch.acos(torch.abs(dot))
        
        return loss.mean()


class ARVRLossWrapper(nn.Module):
    """
    Wrapper that adapts the ARVRAdaptiveLoss to work with separate
    rotation (quaternion) and translation predictions.
    """
    
    def __init__(self, use_log_scale=True, use_weighted_loss=False):
        super().__init__()
        self.use_log_scale = use_log_scale
        self.use_weighted_loss = use_weighted_loss
        self.rotation_loss = ProperQuaternionLoss()
        
    def forward(self, pred_rotation, target_rotation, pred_translation, target_translation):
        """
        Args:
            pred_rotation: [B*seq_len, 4] quaternions
            target_rotation: [B*seq_len, 4] quaternions
            pred_translation: [B*seq_len, 3] translations (in centimeters)
            target_translation: [B*seq_len, 3] translations (in centimeters)
        
        Returns:
            Dictionary with loss components
        """
        # Compute rotation loss (geodesic distance)
        rot_loss = self.rotation_loss(pred_rotation, target_rotation)
        
        # Compute translation loss using Huber/SmoothL1 loss for robustness
        # This is less sensitive to outliers than MSE and works better for small values
        trans_loss = torch.nn.functional.smooth_l1_loss(pred_translation, target_translation)
        
        # Scale losses to similar magnitudes BEFORE log transform
        # Translation is in centimeters, rotation is now in radians (after exact geodesic loss)
        # Typical values: rotation ~0.001 rad, translation ~0.5 cm
        # Use much more conservative scaling to prevent rotation from dominating
        # Start with 10x and tune based on loss balance
        # DISABLED: This scaling causes issues with log transform
        # rot_loss = rot_loss * 10.0
        
        # Optional: use log-scale to prevent underflow with very small values
        if self.use_log_scale:
            # Add 1 to prevent log(0)
            # Apply log AFTER scaling to maintain proper balance
            trans_loss = torch.log1p(trans_loss)
            rot_loss = torch.log1p(rot_loss)
        
        if self.use_weighted_loss:
            # Compute scale-aware weights based on motion magnitude
            with torch.no_grad():
                # Translation magnitude (in centimeters)
                trans_magnitude = torch.norm(target_translation, dim=-1)
                trans_weight = torch.ones_like(trans_magnitude)
                # Reduced weighting to avoid bias toward zero motion
                # Using centimeter scale thresholds
                trans_weight[trans_magnitude < 0.1] = 1.5   # < 1mm (0.1cm)
                trans_weight[trans_magnitude > 1.0] = 0.8   # > 1cm
                
                # Rotation magnitude (quaternion angle)
                quat_dot = target_rotation[:, 3]  # w component
                quat_angle = 2 * torch.acos(torch.clamp(quat_dot.abs(), -1, 1))
                rot_weight = torch.ones_like(quat_angle)
                rot_weight[quat_angle < 0.035] = 1.5  # Was 3.0
                rot_weight[quat_angle > 0.175] = 0.8  # Was 0.5
            
            # Apply weights
            weighted_trans_loss = trans_loss * trans_weight.mean()
            weighted_rot_loss = rot_loss * rot_weight.mean()
        else:
            # No weighting - treat all motions equally
            weighted_trans_loss = trans_loss
            weighted_rot_loss = rot_loss
        
        # Add a small regularization term to prevent trivial solutions
        reg_loss = 0.001 * (pred_rotation.norm(dim=-1).mean() + pred_translation.norm(dim=-1).mean())
        
        # Remove regularization for now - it might be causing the constant predictions
        # The model should learn naturally from the data
        small_trans_penalty = 0.0
        
        return {
            'translation_loss': weighted_trans_loss,
            'rotation_loss': weighted_rot_loss,
            'regularization_loss': reg_loss + small_trans_penalty,
            'total_loss': weighted_trans_loss + weighted_rot_loss + reg_loss + small_trans_penalty
        }