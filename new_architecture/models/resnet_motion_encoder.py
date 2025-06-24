"""
ResNet18-based Motion Encoder for Visual-Inertial Odometry
A more efficient alternative to FlowNet for motion feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple


class ResNet18MotionEncoder(nn.Module):
    """
    ResNet18-based encoder for extracting motion features from image pairs.
    Much more efficient than FlowNet while still capturing good motion representations.
    """
    
    def __init__(self, output_dim=256, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final FC layer and avgpool
        # Keep up to layer4 (output is 512 channels)
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 64 channels
            resnet.layer2,  # 128 channels
            resnet.layer3,  # 256 channels
            resnet.layer4,  # 512 channels
        )
        
        # Freeze early layers if using pretrained
        if pretrained:
            for name, param in self.feature_extractor.named_parameters():
                if 'layer3' not in name and 'layer4' not in name:
                    param.requires_grad = False
        
        # Motion fusion module - combines features from two frames
        self.motion_fusion = nn.Sequential(
            nn.Conv2d(512 * 2, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final projection to output dimension
        self.projection = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )
        
        # Initialize new layers
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for new layers."""
        for m in [self.motion_fusion, self.projection]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, 0, 0.01)
                    nn.init.constant_(layer.bias, 0)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from a single image using ResNet18."""
        return self.feature_extractor(x)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract motion features from consecutive image pairs.
        
        Args:
            images: [B, T, 3, H, W] - Sequence of images
            
        Returns:
            motion_features: [B, T-1, output_dim] - Motion features between consecutive frames
        """
        B, T, C, H, W = images.shape
        motion_features = []
        
        # Process consecutive frame pairs
        for i in range(T - 1):
            # Extract features from both frames
            feat1 = self.extract_features(images[:, i])      # [B, 512, H/32, W/32]
            feat2 = self.extract_features(images[:, i+1])    # [B, 512, H/32, W/32]
            
            # Concatenate features to capture motion
            concat_feat = torch.cat([feat1, feat2], dim=1)   # [B, 1024, H/32, W/32]
            
            # Fuse features to extract motion
            motion_feat = self.motion_fusion(concat_feat)    # [B, 256, H/32, W/32]
            
            # Global pooling
            motion_feat = self.global_pool(motion_feat)      # [B, 256, 1, 1]
            motion_feat = motion_feat.flatten(1)             # [B, 256]
            
            # Project to output dimension
            motion_feat = self.projection(motion_feat)       # [B, output_dim]
            
            motion_features.append(motion_feat)
        
        return torch.stack(motion_features, dim=1)  # [B, T-1, output_dim]


class ResNet18DifferenceEncoder(nn.Module):
    """
    ResNet18-based encoder that processes frame differences.
    Even more efficient by computing differences before feature extraction.
    """
    
    def __init__(self, output_dim=256, pretrained=True):
        super().__init__()
        
        # Modify first conv to accept 6 channels (concatenated frames)
        # or 3 channels (frame difference)
        resnet = models.resnet18(pretrained=pretrained)
        
        # We'll use frame differences (3 channels)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Copy pretrained weights by averaging across RGB channels
        if pretrained:
            # Average the pretrained conv1 weights across input channels
            pretrained_weight = resnet.conv1.weight.data
            self.conv1.weight.data = pretrained_weight
        
        # Rest of ResNet18
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract motion features from frame differences.
        
        Args:
            images: [B, T, 3, H, W] - Sequence of images
            
        Returns:
            motion_features: [B, T-1, output_dim] - Motion features
        """
        B, T, C, H, W = images.shape
        motion_features = []
        
        for i in range(T - 1):
            # Compute frame difference
            diff = images[:, i+1] - images[:, i]  # [B, 3, H, W]
            
            # Extract features from difference
            x = self.conv1(diff)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            motion_feat = self.fc(x)
            
            motion_features.append(motion_feat)
        
        return torch.stack(motion_features, dim=1)  # [B, T-1, output_dim]


# Factory function to create encoders
def create_visual_encoder(encoder_type='resnet18', output_dim=256, pretrained=True):
    """
    Create visual encoder based on type.
    
    Args:
        encoder_type: 'resnet18', 'resnet18_diff', or 'flownet'
        output_dim: Output feature dimension
        pretrained: Whether to use pretrained weights
    
    Returns:
        Visual encoder module
    """
    if encoder_type == 'resnet18':
        return ResNet18MotionEncoder(output_dim=output_dim, pretrained=pretrained)
    elif encoder_type == 'resnet18_diff':
        return ResNet18DifferenceEncoder(output_dim=output_dim, pretrained=pretrained)
    elif encoder_type == 'flownet':
        # Import only when needed to avoid circular imports
        from .flownet_lstm_transformer import FlowNetMotionEncoder
        return FlowNetMotionEncoder(output_dim=output_dim)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")