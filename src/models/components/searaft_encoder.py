"""
SEA-RAFT Feature Encoder for VIFT
Replaces the 6-layer CNN with SEA-RAFT's pretrained feature extractor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
import sys


class SEARAFTFeatureEncoder(nn.Module):
    """
    Feature encoder using SEA-RAFT's pretrained feature network.
    Extracts motion-aware features without computing full optical flow.
    """
    
    def __init__(self, opt):
        super().__init__()
        
        # Load pretrained SEA-RAFT
        print("Loading SEA-RAFT model...")
        
        # Add SEA-RAFT to path
        repo_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'third_party', 'SEA-RAFT')
        repo_dir = os.path.abspath(repo_dir)
        if not os.path.exists(repo_dir):
            raise RuntimeError(
                "SEA-RAFT not found. Please run:\n"
                "  python setup_searaft.py\n"
                "before training with --use-searaft"
            )
        
        # Add to path
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        
        try:
            # Import RAFT (imports should be fixed by setup_searaft.py)
            from core.raft import RAFT
            
            # Create model with small configuration
            # These must match the pretrained model exactly
            args = type('Args', (), {
                'small': True,
                'mixed_precision': False,
                'dropout': 0,
                'upsample': True,
                'corr_implementation': 'reg',
                'corr_radius': 4,
                'radius': 4,  # Same as corr_radius
                'k_conv': 1,
                'use_k_conv': False,
                'dim': 128,  # Hidden dimension for small model
                'iters': 12,  # Number of iterations
                'use_var': False,  # Variance prediction
                'var_min': -5,
                'var_max': 10,
                # ResNet encoder parameters (must match pretrained weights)
                'block_dims': [64, 128, 256],  # Standard dimensions
                'initial_dim': 64,
                'pretrain': 'resnet18',  # ResNet18 backbone
                # CorrBlock parameters
                'num_blocks': 4,
                'K': 4,
                'return_logbase': False,
                # Update block parameters
                'stride': 1,
                'block_channel': 128
            })()
            
            model = RAFT(args)
            
            # Load pretrained weights
            checkpoint_path = os.path.join(repo_dir, 'SEA-RAFT-Sintel.pth')
            if os.path.exists(checkpoint_path):
                try:
                    # PyTorch 2.6+ requires weights_only=False for older checkpoints
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    model.load_state_dict(checkpoint, strict=False)
                    print("✓ Loaded pretrained SEA-RAFT weights")
                except Exception as e:
                    print(f"⚠ ERROR: Failed to load weights: {e}")
                    print("⚠ WARNING: Using random initialization - this will NOT work well!")
                    print("⚠ SEA-RAFT requires pretrained weights to extract motion features.")
            else:
                print("⚠ ERROR: No pretrained weights found!")
                print("⚠ SEA-RAFT REQUIRES pretrained weights to work properly.")
                print("⚠ It was trained for weeks on large optical flow datasets.")
                print("\n  Please download weights from:")
                print("  https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW")
                print("  Save as: third_party/SEA-RAFT/SEA-RAFT-Sintel.pth")
                print("\n⚠ WARNING: Continuing with random initialization - expect poor results!")
            
            self.fnet = model.fnet
            self.fnet.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load SEA-RAFT: {e}")
        
        # Freeze all parameters
        for param in self.fnet.parameters():
            param.requires_grad = False
        
        # Input normalization for RAFT (expects [-1,1] RGB)
        self.register_buffer('mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        
        # Bottleneck to reduce channels (256→128)
        self.bottleneck = nn.Conv2d(256, 128, 1, 1, 0)
        
        # Motion encoder preserving spatial structure
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),  # Downsample
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # → [B, 256, 4, 4]
        )
        
        # Output projection
        self.output_proj = nn.Linear(256 * 16, opt.v_f_len)
        
        print(f"SEARAFTFeatureEncoder initialized with output dim: {opt.v_f_len}")
    
    def normalize(self, x):
        """Normalize from [0,1] to [-1,1]"""
        return (x - self.mean) / self.std
    
    def encode_image(self, x):
        """
        Process image pairs through SEA-RAFT feature extractor.
        
        Args:
            x: [B, 6, H, W] concatenated RGB frames
            
        Returns:
            features: [B, v_f_len] motion features
        """
        # Split concatenated frames
        img1 = x[:, :3]  # First RGB frame
        img2 = x[:, 3:6]  # Second RGB frame
        
        # Normalize for RAFT
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        
        # Extract features without gradients
        with torch.no_grad():
            # fnet returns features
            feat1 = self.fnet(img1)  # [B, 256, H/8, W/8]
            feat2 = self.fnet(img2)
        
        # Motion features via difference
        motion = feat2 - feat1  # [B, 256, H/8, W/8]
        
        # Bottleneck to save memory and computation
        motion = self.bottleneck(motion)  # [B, 128, H/8, W/8]
        
        # Encode with spatial awareness
        features = self.motion_encoder(motion)  # [B, 256, 4, 4]
        
        # Flatten spatial tokens
        features = features.flatten(1)  # [B, 4096]
        
        # Project to output dimension
        return self.output_proj(features)  # [B, v_f_len]
    
    def forward(self, img, imu):
        """
        Forward pass matching original encoder interface.
        
        Args:
            img: Images [B, seq_len, 3, H, W]
            imu: IMU data (passed through unchanged)
            
        Returns:
            v: Visual features [B, seq_len-1, v_f_len]
            i: Inertial features (from original encoder)
        """
        # This method is for compatibility if called directly
        # In practice, vsvio.py will call encode_image directly
        
        # Create image pairs
        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2)  # [B, seq_len-1, 6, H, W]
        batch_size = v.size(0)
        seq_len = v.size(1)
        
        # Process all image pairs
        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        v = self.encode_image(v)  # [B*seq_len, v_f_len]
        v = v.view(batch_size, seq_len, -1)  # [B, seq_len, v_f_len]
        
        # Return visual features only (IMU handled separately in vsvio.py)
        return v, None