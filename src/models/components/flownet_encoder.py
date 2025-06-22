"""FlowNet-C encoder for VIFT.

FlowNet-C uses separate convolutional towers for each image and a correlation layer
to compute similarities between feature maps, which is ideal for visual odometry tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, batchNorm=True):
    """Convolutional layer with batch norm and LeakyReLU."""
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                     padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                     padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


class CorrelationLayer(nn.Module):
    """Correlation layer for FlowNet-C.
    
    Computes correlation between two feature maps at different displacements.
    """
    
    def __init__(self, max_displacement=20, stride=2):
        super(CorrelationLayer, self).__init__()
        self.max_displacement = max_displacement
        self.stride = stride
        self.kernel_size = 2 * max_displacement + 1
        self.corr_channels = self.kernel_size * self.kernel_size
        
    def forward(self, x1, x2):
        """Compute correlation between x1 and x2.
        
        Args:
            x1: First feature map [B, C, H, W]
            x2: Second feature map [B, C, H, W]
            
        Returns:
            Correlation tensor [B, (2*max_disp+1)^2, H, W]
        """
        B, C, H, W = x1.shape
        
        # Pad x2 for displacement computation
        pad_size = self.max_displacement
        x2_padded = F.pad(x2, [pad_size, pad_size, pad_size, pad_size])
        
        # Initialize correlation output
        out_h = H // self.stride
        out_w = W // self.stride
        correlation = torch.zeros(B, self.corr_channels, out_h, out_w).to(x1.device)
        
        # Compute correlation for each displacement
        for i, dx in enumerate(range(-self.max_displacement, self.max_displacement + 1)):
            for j, dy in enumerate(range(-self.max_displacement, self.max_displacement + 1)):
                # Extract shifted x2
                x2_shift = x2_padded[:, :, 
                          pad_size + dy:pad_size + dy + H,
                          pad_size + dx:pad_size + dx + W]
                
                # Compute dot product
                corr = torch.sum(x1 * x2_shift, dim=1, keepdim=True)
                
                # Downsample if needed
                if self.stride > 1:
                    corr = F.avg_pool2d(corr, self.stride, self.stride)
                
                correlation[:, i * self.kernel_size + j] = corr.squeeze(1)
                
        # Normalize by number of channels
        correlation = correlation / C
        
        return correlation


class FlowNetCEncoder(nn.Module):
    """FlowNet-C encoder for VIFT.
    
    Replaces the simple CNN encoder with FlowNet-C architecture that's better
    suited for motion estimation tasks.
    """
    
    def __init__(self, opt):
        super(FlowNetCEncoder, self).__init__()
        self.opt = opt
        
        # Convolutional tower for processing individual images
        # Shared between both images
        self.conv1 = conv(3, 64, kernel_size=7, stride=2, padding=3)  # 256x128
        self.conv2 = conv(64, 128, kernel_size=5, stride=2, padding=2)  # 128x64
        self.conv3 = conv(128, 256, kernel_size=5, stride=2, padding=2)  # 64x32
        
        # Correlation layer
        self.corr = CorrelationLayer(max_displacement=10, stride=2)
        
        # Processing after correlation
        # Input: correlation (441 channels) + conv3 features (32 channels after reduction)
        corr_channels = 441  # (2*10+1)^2 = 21^2 = 441
        self.conv_redir = conv(256, 32, kernel_size=1, stride=1, padding=0)  # Reduce conv3 features
        
        # Combined processing
        self.conv3_1 = conv(corr_channels + 32, 256, stride=1)
        self.conv4 = conv(256, 512, stride=2)  # 32x16
        self.conv4_1 = conv(512, 512, stride=1)
        self.conv5 = conv(512, 512, stride=2)  # 16x8
        self.conv5_1 = conv(512, 512, stride=1)
        self.conv6 = conv(512, 1024, stride=2)  # 8x4
        
        # Compute the output shape
        with torch.no_grad():
            __tmp = Variable(torch.zeros(1, 6, opt.img_w, opt.img_h))
            __tmp_out = self.encode_image_pair(__tmp)
            self.feature_size = int(np.prod(__tmp_out.size()))
        
        # Final projection to feature vector
        self.visual_head = nn.Linear(self.feature_size, opt.v_f_len)
        
        # Inertial encoder (same as original)
        from .vsvio import Inertial_encoder
        self.inertial_encoder = Inertial_encoder(opt)
        
    def extract_features(self, img):
        """Extract features from a single image."""
        out_conv1 = self.conv1(img)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        return out_conv3
        
    def encode_image_pair(self, x):
        """Process a pair of images with FlowNet-C.
        
        Args:
            x: Concatenated images [B, 6, H, W] (first 3 channels: img1, last 3: img2)
            
        Returns:
            Feature tensor
        """
        # Split concatenated images
        img1 = x[:, :3, :, :]
        img2 = x[:, 3:, :, :]
        
        # Extract features from both images
        feat1 = self.extract_features(img1)  # [B, 256, H/8, W/8]
        feat2 = self.extract_features(img2)  # [B, 256, H/8, W/8]
        
        # Compute correlation
        corr = self.corr(feat1, feat2)  # [B, 441, H/16, W/16]
        
        # Reduce feat1 channels
        feat1_redir = self.conv_redir(feat1)  # [B, 32, H/8, W/8]
        
        # Downsample feat1_redir to match correlation size
        feat1_redir = F.avg_pool2d(feat1_redir, 2, 2)  # [B, 32, H/16, W/16]
        
        # Concatenate correlation and features
        concat_feat = torch.cat([corr, feat1_redir], dim=1)  # [B, 473, H/16, W/16]
        
        # Continue processing
        out = self.conv3_1(concat_feat)
        out = self.conv4_1(self.conv4(out))
        out = self.conv5_1(self.conv5(out))
        out = self.conv6(out)
        
        return out
        
    def forward(self, img, imu):
        """Forward pass matching original encoder interface.
        
        Args:
            img: Images [B, seq_len, 3, H, W]
            imu: IMU data [B, seq_len*11, 6]
            
        Returns:
            v: Visual features [B, seq_len-1, v_f_len]
            i: Inertial features [B, seq_len-1, i_f_len]
        """
        # Create image pairs (same as original encoder)
        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2)  # [B, seq_len-1, 6, H, W]
        batch_size = v.size(0)
        seq_len = v.size(1)
        
        # Process all image pairs
        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        v = self.encode_image_pair(v)  # [B*seq_len, 1024, h, w]
        v = v.view(batch_size * seq_len, -1)  # [B*seq_len, features]
        v = self.visual_head(v)  # [B*seq_len, v_f_len]
        v = v.view(batch_size, seq_len, -1)  # [B, seq_len, v_f_len]
        
        # Process IMU (same as original)
        imu = torch.cat([imu[:, i * 10:i * 10 + 11, :].unsqueeze(1) 
                        for i in range(seq_len)], dim=1)
        imu = self.inertial_encoder(imu)
        
        return v, imu