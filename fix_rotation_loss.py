#!/usr/bin/env python3
"""
Fix rotation loss to use proper quaternion distance metric
"""

import torch
import torch.nn as nn
import numpy as np


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
        # Loss = 1 - |dot| is a good approximation and more stable
        loss = 1.0 - torch.abs(dot)
        
        return loss.mean()


class ImprovedARVRLoss(nn.Module):
    """
    Improved AR/VR loss with proper quaternion handling.
    """
    
    def __init__(self):
        super().__init__()
        self.translation_loss = nn.MSELoss()
        self.rotation_loss = ProperQuaternionLoss()
        
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
        # Translation loss (unchanged)
        trans_loss = self.translation_loss(pred_translation, target_translation)
        
        # Rotation loss with proper quaternion distance
        rot_loss = self.rotation_loss(pred_rotation, target_rotation)
        
        # Scale rotation loss to be in similar range as translation
        # Rotation loss is in [0, 1], scale it up
        rot_loss = rot_loss * 10.0  # Adjust this scaling factor as needed
        
        # Compute scale-aware weights based on motion magnitude
        with torch.no_grad():
            # Translation magnitude
            trans_magnitude = torch.norm(target_translation, dim=-1)
            trans_weight = torch.ones_like(trans_magnitude)
            trans_weight[trans_magnitude < 0.01] = 3.0  # Small motions < 1cm
            trans_weight[trans_magnitude > 0.05] = 0.5  # Large motions > 5cm
            
            # For rotation, use the loss value as a proxy for magnitude
            # Small rotations will have small loss values
            rot_magnitude = 1.0 - torch.abs(torch.sum(pred_rotation * target_rotation, dim=-1))
            rot_weight = torch.ones_like(rot_magnitude)
            rot_weight[rot_magnitude < 0.001] = 3.0  # Very small rotations
            rot_weight[rot_magnitude > 0.1] = 0.5    # Large rotations
        
        # Apply weights
        weighted_trans_loss = trans_loss * trans_weight.mean()
        weighted_rot_loss = rot_loss * rot_weight.mean()
        
        return {
            'translation_loss': weighted_trans_loss,
            'rotation_loss': weighted_rot_loss,
            'total_loss': weighted_trans_loss + weighted_rot_loss
        }


def test_quaternion_loss():
    """Test the quaternion loss function."""
    loss_fn = ProperQuaternionLoss()
    
    # Test 1: Identical quaternions
    q1 = torch.tensor([[0, 0, 0, 1], [0.7071, 0, 0, 0.7071]], dtype=torch.float32)
    q2 = q1.clone()
    loss = loss_fn(q1, q2)
    print(f"Test 1 - Identical quaternions: {loss.item():.6f} (should be ~0)")
    
    # Test 2: Opposite quaternions (same rotation due to double cover)
    q3 = -q1
    loss = loss_fn(q1, q3)
    print(f"Test 2 - Opposite quaternions: {loss.item():.6f} (should be ~0)")
    
    # Test 3: 90 degree rotation difference
    q4 = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=torch.float32)
    q5 = torch.tensor([[0.7071, 0, 0, 0.7071], [0, 0.7071, 0, 0.7071]], dtype=torch.float32)
    loss = loss_fn(q4, q5)
    print(f"Test 3 - 90° rotation: {loss.item():.6f} (should be ~0.293)")
    
    # Test 4: 180 degree rotation difference
    q6 = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)
    q7 = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)
    loss = loss_fn(q6, q7)
    print(f"Test 4 - 180° rotation: {loss.item():.6f} (should be ~1.0)")
    
    # Test the issue from diagnostics
    print("\nTest 5 - Diagnostic issue:")
    gt_q = torch.tensor([[-1.4114380e-04, 1.1369586e-04, -1.5597045e-04, 1.0000000e+00]])
    pred_q = torch.tensor([[9.99999821e-01, -5.56499814e-04, 2.72401056e-04, 1.11984475e-04]])
    
    # Note: the model seems to output WXYZ while ground truth is XYZW
    # Let's test both interpretations
    print(f"  As-is loss: {loss_fn(pred_q, gt_q).item():.6f}")
    
    # Swap WXYZ to XYZW in prediction
    pred_q_swapped = torch.tensor([[-5.56499814e-04, 2.72401056e-04, 1.11984475e-04, 9.99999821e-01]])
    print(f"  With swapped prediction: {loss_fn(pred_q_swapped, gt_q).item():.6f}")


if __name__ == "__main__":
    print("Testing quaternion loss function...")
    test_quaternion_loss()
    
    print("\n\nSuggested fix:")
    print("1. Replace ARVRLossWrapper with ImprovedARVRLoss in the model")
    print("2. Check if model outputs WXYZ instead of XYZW format")
    print("3. Ensure proper quaternion normalization in model output")