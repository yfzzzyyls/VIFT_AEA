import torch
import math
from torch import nn
from scipy.spatial.transform import Rotation
import numpy as np

class PoseTransformerQuaternion(nn.Module):
    """
    VIFT Original Transformer modified to output quaternions instead of Euler angles.
    Output: [tx, ty, tz, qx, qy, qz, qw] (7 dimensions)
    """
    def __init__(self, input_dim=768, embedding_dim=128, num_layers=2, nhead=8, dim_feedforward=512, dropout=0.1):
        super(PoseTransformerQuaternion, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_dim, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ), 
            num_layers=num_layers
        )
        
        # Modified output layer for quaternion (7D instead of 6D)
        self.fc2 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.embedding_dim, 7)  # Changed from 6 to 7
        )
        
        # Better initialization to encourage learning motion patterns
        with torch.no_grad():
            # Use smaller gain for more stable training
            nn.init.xavier_uniform_(self.fc2[-1].weight, gain=0.1)
            # Initialize bias to small values to encourage learning
            # Small translation bias and near-identity quaternion
            self.fc2[-1].bias.data = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            # Add small noise to break symmetry
            self.fc2[-1].bias.data[:3] += torch.randn(3) * 0.01  # Small translation noise
    
    def positional_embedding(self, seq_length):
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * -(math.log(10000.0) / self.embedding_dim))
        pos_embedding = torch.zeros(seq_length, self.embedding_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        pos_embedding = pos_embedding.unsqueeze(0)
        return pos_embedding

    def generate_square_subsequent_mask(self, sz, device=None, dtype=None):
        """Generate a square causal mask for sequence.
        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        return torch.triu(
                torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
                diagonal=1
        )

    def forward(self, batch, gt):
        # Handle different input formats
        if isinstance(batch, dict):
            # Dictionary format from inference
            visual_features = batch.get('visual_features')
            imu_features = batch.get('imu_features')
            if visual_features is not None and imu_features is not None:
                # Concatenate if separate
                visual_inertial_features = torch.cat([visual_features, imu_features], dim=-1)
            else:
                # Try to get combined features
                visual_inertial_features = batch.get('images', batch.get('features'))
                if visual_inertial_features is None:
                    raise ValueError("Could not find features in batch dict")
        elif isinstance(batch, (list, tuple)):
            # Tuple format
            visual_inertial_features, _, _ = batch
        else:
            # Direct tensor
            visual_inertial_features = batch
            
        seq_length = visual_inertial_features.size(1)

        # Generate causal mask
        pos_embedding = self.positional_embedding(seq_length).to(visual_inertial_features.device)
        visual_inertial_features = self.fc1(visual_inertial_features)
        visual_inertial_features += pos_embedding

        
        # Passing through the transformer encoder with the mask
        mask = self.generate_square_subsequent_mask(seq_length, visual_inertial_features.device)
        output = self.transformer_encoder(visual_inertial_features, mask=mask, is_causal=True)

        # Pass the output through the fully connected layer
        output = self.fc2(output)
        
        # Normalize quaternions (last 4 dimensions)
        # Split translation and rotation
        translation = output[:, :, :3]
        quaternion_raw = output[:, :, 3:]
        
        # Normalize quaternions to unit length
        quaternion_norm = torch.norm(quaternion_raw, dim=-1, keepdim=True) + 1e-8
        quaternion_normalized = quaternion_raw / quaternion_norm
        
        # Concatenate back together
        output_normalized = torch.cat([translation, quaternion_normalized], dim=-1)
        
        return output_normalized


class PoseTransformerQuaternionEulerInput(nn.Module):
    """
    VIFT Original Transformer that accepts Euler angle ground truth but outputs quaternions.
    This is for backward compatibility with existing training data.
    Input GT: [tx, ty, tz, rx, ry, rz] (6D Euler)
    Output: [tx, ty, tz, qx, qy, qz, qw] (7D quaternion)
    """
    def __init__(self, input_dim=768, embedding_dim=128, num_layers=2, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        # Use the quaternion transformer
        self.transformer = PoseTransformerQuaternion(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
    
    def forward(self, batch, gt):
        # Get quaternion output from transformer
        output_quat = self.transformer(batch, gt)
        
        # During training, we'll need to convert GT Euler to quaternions for loss computation
        # But the output is already in quaternion format
        return output_quat
    
    @staticmethod
    def convert_euler_to_quaternion(euler_poses):
        """
        Convert Euler angle poses to quaternion poses
        Input: [B, seq_len, 6] with [tx, ty, tz, rx, ry, rz]
        Output: [B, seq_len, 7] with [tx, ty, tz, qx, qy, qz, qw]
        """
        B, seq_len, _ = euler_poses.shape
        quat_poses = torch.zeros(B, seq_len, 7, device=euler_poses.device)
        
        # Copy translation
        quat_poses[:, :, :3] = euler_poses[:, :, :3]
        
        # Convert rotation from Euler to quaternion
        for b in range(B):
            for s in range(seq_len):
                euler = euler_poses[b, s, 3:].cpu().numpy()
                # Convert Euler angles to quaternion using scipy
                r = Rotation.from_euler('xyz', euler)
                quat = r.as_quat()  # Returns [x, y, z, w]
                quat_poses[b, s, 3:] = torch.from_numpy(quat).to(euler_poses.device)
        
        return quat_poses