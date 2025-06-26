"""
SEA-RAFT Feature Encoder for VIFT
Replaces the 6-layer CNN with SEA-RAFT's pretrained feature extractor.
Extended with multi-frame correlation support via feature bank.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
import sys
from typing import Optional, List, Tuple

# Import multi-frame components
from .feature_bank import FeatureBank
from .keyframe_selector import KeyFrameSelector
from .multi_edge_correlation import MultiEdgeCorrelation


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
        
        # Multi-frame correlation components
        # Now using sparse sub-window lookup (DROID-SLAM style) to avoid memory explosion
        self.use_multiframe = getattr(opt, 'use_multiframe', True)
        if self.use_multiframe:
            print("Initializing multi-frame correlation components...")
            self.feature_bank = FeatureBank(max_frames=100)
            self.keyframe_selector = KeyFrameSelector(temporal_guard_frames=30)
            self.multi_edge_corr = MultiEdgeCorrelation(corr_radius=4, corr_levels=4)
            
            # Fusion layer for combining direct + multi-frame motion
            # Now using simple dot-product correlation (1 channel)
            self.motion_fusion = nn.Sequential(
                nn.Conv2d(1, 64, 3, 1, 1),  # Expand correlation to features
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.ReLU()
            )
            
            # Learnable fusion weight
            self.multi_frame_weight = nn.Parameter(torch.tensor(0.1))
        
        print(f"SEARAFTFeatureEncoder initialized with output dim: {opt.v_f_len}")
        if self.use_multiframe:
            print("✓ Multi-frame correlation enabled")
    
    def normalize(self, x):
        """Normalize from [0,1] to [-1,1]"""
        return (x - self.mean) / self.std
    
    def extract_features(self, img):
        """Extract features from single image."""
        img_norm = self.normalize(img)
        with torch.no_grad():
            return self.fnet(img_norm)
    
    def encode_image(self, x, frame_ids: Optional[List[int]] = None):
        """
        Process image pairs through SEA-RAFT feature extractor.
        
        Args:
            x: [B, 6, H, W] concatenated RGB frames
            frame_ids: Optional list of frame IDs for multi-frame correlation
            
        Returns:
            features: [B, v_f_len] motion features
        """
        # Split concatenated frames
        img1 = x[:, :3]  # First RGB frame
        img2 = x[:, 3:6]  # Second RGB frame
        
        # Extract features
        feat1 = self.extract_features(img1)  # [B, 256, H/8, W/8]
        feat2 = self.extract_features(img2)
        
        # Direct motion features via difference
        motion_direct = feat2 - feat1  # [B, 256, H/8, W/8]
        
        # Multi-frame correlation if enabled
        if self.use_multiframe and frame_ids is not None:
            # Handle batch processing - take last frame ID from each batch element
            if isinstance(frame_ids, list) and len(frame_ids) == x.size(0):
                current_ids = frame_ids
            else:
                # If single sequence, use last frame
                current_ids = [frame_ids[-1]] if isinstance(frame_ids, list) else [frame_ids]
            
            # Process each batch element
            batch_motions = []
            for b in range(x.size(0)):
                current_feat = feat2[b:b+1]  # Keep batch dim
                current_id = current_ids[b] if b < len(current_ids) else current_ids[0]
                
                # Add to feature bank
                self.feature_bank.add_features(current_id, feat2[b])
                
                # Select keyframes
                keyframe_ids = self.keyframe_selector.select_keyframes(
                    current_id, feat2[b], self.feature_bank, top_k=3
                )
                
                if len(keyframe_ids) > 0:
                    # Get keyframe features
                    keyframe_feats = self.feature_bank.get_features(keyframe_ids)
                    
                    # Compute multi-frame correlation
                    with torch.amp.autocast('cuda', enabled=True):
                        multi_corr = self.multi_edge_corr.aggregate_multi_edge_flow(
                            current_feat, keyframe_feats
                        )
                    
                    if multi_corr is not None:
                        # Project correlation to motion space
                        multi_motion = self.motion_fusion(multi_corr)  # [1, 128, H, W]
                        
                        # Combine with direct motion before bottleneck
                        # Expand multi_motion to match motion_direct channels
                        multi_motion_expanded = F.pad(multi_motion, (0, 0, 0, 0, 0, 128))  # [1, 256, H, W]
                        combined_motion = motion_direct[b:b+1] + self.multi_frame_weight * multi_motion_expanded
                        batch_motions.append(combined_motion)
                    else:
                        batch_motions.append(motion_direct[b:b+1])
                else:
                    batch_motions.append(motion_direct[b:b+1])
            
            # Concatenate batch
            motion = torch.cat(batch_motions, dim=0)
        else:
            motion = motion_direct
        
        # Bottleneck to save memory and computation
        motion = self.bottleneck(motion)  # [B, 128, H/8, W/8]
        
        # Encode with spatial awareness
        features = self.motion_encoder(motion)  # [B, 256, 4, 4]
        
        # Flatten spatial tokens
        features = features.flatten(1)  # [B, 4096]
        
        # Project to output dimension
        return self.output_proj(features)  # [B, v_f_len]
    
    def forward(self, img, imu, frame_ids: Optional[List[int]] = None):
        """
        Forward pass matching original encoder interface.
        
        Args:
            img: Images [B, seq_len, 3, H, W]
            imu: IMU data (passed through unchanged)
            frame_ids: Optional frame IDs for multi-frame correlation
            
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
        
        # Pass frame IDs if available
        if frame_ids is not None and self.use_multiframe:
            # Create frame ID list for each pair in the batch
            batch_frame_ids = []
            for b in range(batch_size):
                for s in range(seq_len):
                    # Use the second frame ID of each pair
                    if isinstance(frame_ids[0], list):
                        # frame_ids is [B, seq_len]
                        batch_frame_ids.append(frame_ids[b][s + 1])
                    else:
                        # frame_ids is [seq_len]
                        batch_frame_ids.append(frame_ids[s + 1])
            v = self.encode_image(v, batch_frame_ids)
        else:
            v = self.encode_image(v)
            
        v = v.view(batch_size, seq_len, -1)  # [B, seq_len, v_f_len]
        
        # Return visual features only (IMU handled separately in vsvio.py)
        return v, None
        
    def clear_feature_bank(self):
        """Clear the feature bank (useful between sequences)."""
        if self.use_multiframe:
            self.feature_bank.clear()
            
    def get_feature_bank_stats(self):
        """Get feature bank statistics."""
        if self.use_multiframe:
            return {
                'num_frames': len(self.feature_bank),
                'memory_usage_mb': self.feature_bank.get_memory_usage_mb(),
                'keyframe_stats': self.keyframe_selector.get_statistics()
            }
        return {}