import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import math
from .flexible_encoder import FlexibleEncoder

class PoseTransformer(nn.Module):
    def __init__(self, opt):
        super(PoseTransformer, self).__init__()

        self.embedding_dim = opt.embedding_dim
        self.num_layers = opt.num_layers

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_dim, 
                nhead=opt.nhead, 
                dim_feedforward=opt.dim_feedforward,
                dropout=opt.dropout,
                batch_first=True
            ), 
            num_layers=self.num_layers
        )
        # Add the fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embedding_dim, 6))
    
    def positional_embedding(self, seq_length):
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * -(math.log(10000.0) / self.embedding_dim))
        pos_embedding = torch.zeros(seq_length, self.embedding_dim)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        pos_embedding = pos_embedding.unsqueeze(0)
        return pos_embedding

    def forward(self, visual_inertial_features):
        seq_length = visual_inertial_features.size(1)

        # Generate positional embedding
        pos_embedding = self.positional_embedding(seq_length).to(visual_inertial_features.device)
        visual_inertial_features += pos_embedding

        # Passing through the transformer encoder
        output = self.transformer_encoder(visual_inertial_features, mask=None)

        # Pass the output through the fully connected layer
        output = self.fc(output)

        return output


class FlexibleTransformerVIO(nn.Module):
    """TransformerVIO with flexible IMU sample rate handling."""
    
    def __init__(self, opt, samples_per_interval=None):
        super().__init__()
        self.window_size = opt.seq_len
        self.Feature_net = FlexibleEncoder(opt, samples_per_interval)
        self.Pose_net = PoseTransformer(opt)
        initialization(self)

    def forward(self, img, imu):
        fv, fi = self.Feature_net(img, imu)
        visual_inertial_feature = torch.cat([fv, fi], dim=-1) 

        # Continue processing as before
        poses = self.Pose_net(visual_inertial_feature)
        return poses


def initialization(net):
    """Initialize network parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(0)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    start, end = n//4, n//2
                    param.data[start:end].fill_(1.)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()