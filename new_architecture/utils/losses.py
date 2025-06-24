"""
Loss functions for visual-inertial odometry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def quaternion_geodesic_loss(pred_quat: torch.Tensor, gt_quat: torch.Tensor) -> torch.Tensor:
    """
    Compute geodesic distance between predicted and ground truth quaternions.
    Uses numerically stable atan2 formulation to avoid gradient issues.
    
    Args:
        pred_quat: Predicted quaternions [B, T, 4] or [B*T, 4]
        gt_quat: Ground truth quaternions [B, T, 4] or [B*T, 4]
        
    Returns:
        Mean geodesic distance in radians
    """
    # Reshape to [N, 4] for batch processing
    original_shape = pred_quat.shape
    pred_quat = pred_quat.reshape(-1, 4)
    gt_quat = gt_quat.reshape(-1, 4)
    
    # Normalize quaternions
    pred_quat = F.normalize(pred_quat, p=2, dim=-1, eps=1e-8)
    gt_quat = F.normalize(gt_quat, p=2, dim=-1, eps=1e-8)
    
    # Compute dot product (handle double cover with abs)
    dot_product = torch.sum(pred_quat * gt_quat, dim=-1)
    abs_dot = dot_product.abs()
    
    # Use atan2 for numerical stability
    # angle = 2 * atan2(||q1 - q2||, ||q1 + q2||) for same hemisphere
    # angle = 2 * atan2(||q1 + q2||, ||q1 - q2||) for opposite hemisphere
    
    # Clamp to avoid numerical issues
    abs_dot_clamped = abs_dot.clamp(max=1.0 - 1e-7)
    
    # Compute angle using stable formula
    angle = 2.0 * torch.atan2(
        torch.sqrt(1.0 - abs_dot_clamped**2),
        abs_dot_clamped
    )
    
    return angle.mean()


def scale_consistency_loss(translations: torch.Tensor) -> torch.Tensor:
    """
    Compute scale consistency loss to prevent scale drift.
    Encourages consistent scale across the sequence.
    
    Args:
        translations: Predicted translations [B, T, 3]
        
    Returns:
        Scale consistency loss
    """
    # Compute translation magnitudes
    trans_norms = torch.norm(translations, dim=-1)  # [B, T]
    
    # Compute mean scale per sequence
    mean_scale = trans_norms.mean(dim=1, keepdim=True)  # [B, 1]
    
    # Penalize deviation from mean scale
    scale_loss = ((trans_norms - mean_scale) ** 2).mean()
    
    return scale_loss


def temporal_smoothness_loss(poses: torch.Tensor) -> torch.Tensor:
    """
    Compute temporal smoothness loss to encourage smooth trajectories.
    
    Args:
        poses: Predicted poses [B, T, 7] (3 trans + 4 quat)
        
    Returns:
        Temporal smoothness loss
    """
    # Compute first-order differences
    trans_diff = poses[:, 1:, :3] - poses[:, :-1, :3]  # [B, T-1, 3]
    
    # Compute second-order differences (acceleration)
    trans_acc = trans_diff[:, 1:] - trans_diff[:, :-1]  # [B, T-2, 3]
    
    # Penalize large accelerations
    smoothness_loss = torch.norm(trans_acc, dim=-1).mean()
    
    return smoothness_loss


def compute_pose_loss(
    pred_poses: torch.Tensor,
    gt_poses: torch.Tensor,
    sequence_lengths: Optional[torch.Tensor] = None,
    translation_weight: float = 1.0,
    rotation_weight: float = 10.0,
    scale_weight: float = 20.0,
    smoothness_weight: float = 0.1
) -> Dict[str, torch.Tensor]:
    """
    Compute complete pose loss with all components.
    
    Args:
        pred_poses: Predicted poses [B, T, 7] (3 trans + 4 quat)
        gt_poses: Ground truth poses [B, T, 7]
        sequence_lengths: Actual sequence lengths for masking [B]
        translation_weight: Weight for translation loss
        rotation_weight: Weight for rotation loss
        scale_weight: Weight for scale consistency loss
        smoothness_weight: Weight for temporal smoothness loss
        
    Returns:
        Dictionary containing individual losses and total loss
    """
    B, T, _ = pred_poses.shape
    
    # Create mask for variable-length sequences
    if sequence_lengths is not None:
        # Create mask [B, T]
        mask = torch.arange(T, device=pred_poses.device).unsqueeze(0) < (sequence_lengths.unsqueeze(1) - 1)
        mask = mask.float()
    else:
        mask = torch.ones(B, T, device=pred_poses.device)
    
    # Split poses
    pred_trans = pred_poses[..., :3]
    pred_rot = pred_poses[..., 3:]
    gt_trans = gt_poses[..., :3]
    gt_rot = gt_poses[..., 3:]
    
    # Translation loss (MSE)
    trans_loss = F.mse_loss(pred_trans, gt_trans, reduction='none')
    trans_loss = (trans_loss.mean(dim=-1) * mask).sum() / mask.sum()
    
    # Rotation loss (geodesic)
    rot_loss = quaternion_geodesic_loss(pred_rot, gt_rot)
    
    # Scale consistency loss
    scale_loss = scale_consistency_loss(pred_trans)
    
    # Temporal smoothness loss
    smoothness_loss = temporal_smoothness_loss(pred_poses)
    
    # Total loss
    total_loss = (
        translation_weight * trans_loss +
        rotation_weight * rot_loss +
        scale_weight * scale_loss +
        smoothness_weight * smoothness_loss
    )
    
    return {
        'total_loss': total_loss,
        'translation_loss': trans_loss,
        'rotation_loss': rot_loss,
        'scale_loss': scale_loss,
        'smoothness_loss': smoothness_loss
    }


class RobustPoseLoss(nn.Module):
    """
    Robust pose loss using Huber loss for translation and geodesic loss for rotation.
    More robust to outliers than standard MSE.
    """
    
    def __init__(
        self,
        translation_weight: float = 1.0,
        rotation_weight: float = 10.0,
        scale_weight: float = 20.0,
        huber_delta: float = 1.0
    ):
        super().__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.scale_weight = scale_weight
        self.huber_loss = nn.HuberLoss(delta=huber_delta)
        
    def forward(
        self,
        pred_poses: torch.Tensor,
        gt_poses: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute robust pose loss.
        
        Args:
            pred_poses: Predicted poses [B, T, 7]
            gt_poses: Ground truth poses [B, T, 7]
            mask: Optional mask for valid poses [B, T]
            
        Returns:
            Loss dictionary
        """
        # Split poses
        pred_trans = pred_poses[..., :3]
        pred_rot = pred_poses[..., 3:]
        gt_trans = gt_poses[..., :3]
        gt_rot = gt_poses[..., 3:]
        
        # Translation loss (Huber)
        if mask is not None:
            # Apply mask by setting invalid positions to match ground truth
            pred_trans_masked = pred_trans * mask.unsqueeze(-1) + gt_trans * (1 - mask.unsqueeze(-1))
            trans_loss = self.huber_loss(pred_trans_masked, gt_trans)
        else:
            trans_loss = self.huber_loss(pred_trans, gt_trans)
        
        # Rotation loss
        rot_loss = quaternion_geodesic_loss(pred_rot, gt_rot)
        
        # Scale consistency
        scale_loss = scale_consistency_loss(pred_trans)
        
        # Total loss
        total_loss = (
            self.translation_weight * trans_loss +
            self.rotation_weight * rot_loss +
            self.scale_weight * scale_loss
        )
        
        return {
            'total_loss': total_loss,
            'translation_loss': trans_loss,
            'rotation_loss': rot_loss,
            'scale_loss': scale_loss
        }