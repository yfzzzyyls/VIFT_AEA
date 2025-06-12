
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryAwareARVRLoss(nn.Module):
    """AR/VR Loss with trajectory awareness and human motion regularization"""
    
    def __init__(self, 
                 translation_weight=1.0,
                 rotation_weight=1.0,
                 trajectory_weight=0.2,
                 smoothness_weight=0.1,
                 diversity_weight=0.05):
        super().__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.trajectory_weight = trajectory_weight
        self.smoothness_weight = smoothness_weight
        self.diversity_weight = diversity_weight
        
    def forward(self, predictions, targets):
        # predictions/targets: [batch_size, 10, 7]
        batch_size = predictions.shape[0]
        
        # Separate translation and rotation
        pred_trans = predictions[:, :, :3]
        pred_rot = predictions[:, :, 3:]
        target_trans = targets[:, :, :3]
        target_rot = targets[:, :, 3:]
        
        # 1. Frame-to-frame loss (standard)
        trans_loss = F.mse_loss(pred_trans, target_trans)
        
        # Rotation loss (geodesic)
        pred_rot_norm = F.normalize(pred_rot, p=2, dim=-1)
        target_rot_norm = F.normalize(target_rot, p=2, dim=-1)
        dot = (pred_rot_norm * target_rot_norm).sum(dim=-1)
        dot = torch.clamp(dot, -1.0, 1.0)
        rot_loss = (1.0 - torch.abs(dot)).mean()
        
        # 2. Trajectory coherence
        # Accumulate poses to check end-to-end error
        pred_cumsum = torch.cumsum(pred_trans, dim=1)
        target_cumsum = torch.cumsum(target_trans, dim=1)
        trajectory_loss = F.mse_loss(pred_cumsum, target_cumsum)
        
        # 3. Motion smoothness (penalize jerky motion)
        if predictions.shape[1] > 2:
            # Velocity
            pred_vel = pred_trans[:, 1:] - pred_trans[:, :-1]
            # Acceleration  
            pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
            smoothness_loss = torch.mean(torch.norm(pred_acc, dim=-1))
        else:
            smoothness_loss = torch.tensor(0.0, device=predictions.device)
        
        # 4. Motion diversity (prevent constant predictions)
        motion_std = torch.std(pred_trans, dim=1).mean()
        diversity_loss = 1.0 / (motion_std + 1e-6)
        
        # Combine losses
        total_loss = (
            self.translation_weight * trans_loss +
            self.rotation_weight * rot_loss +
            self.trajectory_weight * trajectory_loss +
            self.smoothness_weight * smoothness_loss +
            self.diversity_weight * diversity_loss
        )
        
        return total_loss
