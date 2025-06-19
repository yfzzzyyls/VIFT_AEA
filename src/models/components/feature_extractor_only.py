import torch
import torch.nn as nn
from .flexible_encoder import FlexibleEncoder, initialization

class FeatureExtractorVIO(nn.Module):
    """Lightweight VIO model that only includes the feature extraction components."""
    
    def __init__(self, opt, samples_per_interval=None):
        super().__init__()
        self.window_size = opt.seq_len
        self.Feature_net = FlexibleEncoder(opt, samples_per_interval)
        initialization(self)

    def forward(self, img, imu):
        """Extract visual and IMU features only."""
        fv, fi = self.Feature_net(img, imu)
        return fv, fi