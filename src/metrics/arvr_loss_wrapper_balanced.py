"""
Balanced wrapper for ARVRAdaptiveLoss that properly scales rotation and translation losses
"""

import torch
import torch.nn as nn


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
        loss = 2.0 * torch.acos(torch.abs(dot))
        
        return loss.mean()


class BalancedARVRLossWrapper(nn.Module):
    """
    Balanced loss wrapper that properly scales rotation and translation components
    to prevent one from dominating the other.
    
    Key improvements:
    1. Scales rotation loss down by 0.1x to match translation loss magnitude
    2. Uses robust Smooth L1 loss for translation
    3. Removes problematic regularization terms
    4. Optional adaptive weighting based on motion magnitude
    """
    
    def __init__(self, 
                 rotation_scale=0.1,
                 translation_scale=1.0, 
                 use_adaptive_weighting=False,
                 use_log_scale=False):
        super().__init__()
        self.rotation_scale = rotation_scale
        self.translation_scale = translation_scale
        self.use_adaptive_weighting = use_adaptive_weighting
        self.use_log_scale = use_log_scale
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
        # Compute rotation loss (geodesic distance in radians)
        rot_loss = self.rotation_loss(pred_rotation, target_rotation)
        
        # Compute translation loss using Smooth L1 (more robust than MSE)
        trans_loss = torch.nn.functional.smooth_l1_loss(
            pred_translation * self.translation_scale,
            target_translation * self.translation_scale
        )
        
        # Scale rotation loss to match translation loss magnitude
        # Typical values: rotation ~2.0 rad, translation ~0.2
        # So we scale rotation by 0.1 to get ~0.2
        scaled_rot_loss = rot_loss * self.rotation_scale
        
        if self.use_adaptive_weighting:
            # Compute adaptive weights based on motion magnitude
            with torch.no_grad():
                # Translation magnitude (in centimeters)
                trans_magnitude = torch.norm(target_translation, dim=-1)
                trans_weight = torch.ones_like(trans_magnitude)
                # Gentle weighting to avoid bias
                trans_weight[trans_magnitude < 0.5] = 1.2   # < 5mm
                trans_weight[trans_magnitude > 5.0] = 0.8   # > 5cm
                
                # Rotation magnitude (quaternion angle in radians)
                # For quaternion q = [x,y,z,w], angle = 2*acos(|w|)
                quat_w = torch.abs(target_rotation[:, 3])
                quat_angle = 2 * torch.acos(torch.clamp(quat_w, -1, 1))
                rot_weight = torch.ones_like(quat_angle)
                rot_weight[quat_angle < 0.02] = 1.2  # < ~1 degree
                rot_weight[quat_angle > 0.2] = 0.8   # > ~11 degrees
            
            # Apply weights
            weighted_trans_loss = trans_loss * trans_weight.mean()
            weighted_rot_loss = scaled_rot_loss * rot_weight.mean()
        else:
            # No adaptive weighting
            weighted_trans_loss = trans_loss
            weighted_rot_loss = scaled_rot_loss
        
        # Optional log scale (not recommended for quaternion model)
        if self.use_log_scale:
            weighted_trans_loss = torch.log1p(weighted_trans_loss)
            weighted_rot_loss = torch.log1p(weighted_rot_loss)
        
        # No regularization - let the model learn naturally
        reg_loss = 0.0
        
        return {
            'translation_loss': weighted_trans_loss,
            'rotation_loss': weighted_rot_loss,
            'regularization_loss': reg_loss,
            'total_loss': weighted_trans_loss + weighted_rot_loss,
            # Additional info for logging
            'raw_translation_loss': trans_loss,
            'raw_rotation_loss': rot_loss,
            'loss_ratio': weighted_rot_loss / (weighted_trans_loss + 1e-8)
        }