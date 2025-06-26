import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs
import torch.nn.functional as F
import math

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0.0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )

# Note: Old fixed 11-sample Inertial_encoder removed. Use FlexibleInertialEncoder instead.


class FlexibleInertialEncoder(nn.Module):
    """Enhanced IMU encoder that accepts variable-length IMU sequences."""
    
    def __init__(self, opt):
        super(FlexibleInertialEncoder, self).__init__()
        
        # Keep same 3-layer Conv1D architecture as original
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout)
        )
        
        # Multi-scale temporal aggregation using average pooling for robustness
        self.pool_local = nn.AdaptiveAvgPool1d(8)    # Local motion details
        self.pool_mid = nn.AdaptiveAvgPool1d(4)      # Mid-level patterns
        self.pool_global = nn.AdaptiveAvgPool1d(1)   # Global motion statistics
        
        # Combine multi-scale features
        self.combine = nn.Sequential(
            nn.Linear(256 * 13, 512),  # 8+4+1=13 pooled positions
            nn.ReLU(),
            nn.Dropout(opt.imu_dropout),
            nn.Linear(512, opt.i_f_len)  # Project to same dimension as original
        )
    
    def forward(self, x):
        # x: (N, seq_len, num_samples, 6) - variable num_samples per transition!
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        outputs = []
        for i in range(seq_len):
            # Get IMU sequence for this transition
            imu_seq = x[:, i]  # [N, num_samples, 6]
            
            # Handle empty sequences
            if imu_seq.shape[1] == 0:
                outputs.append(torch.zeros(batch_size, 256, device=x.device))
                continue
            
            # Transpose for Conv1d: [N, 6, num_samples]
            imu_seq = imu_seq.permute(0, 2, 1)
            
            # Extract features at all temporal positions
            features = self.encoder_conv(imu_seq)  # [N, 256, num_samples]
            
            # Multi-scale pooling to capture different temporal resolutions
            local_feat = self.pool_local(features).flatten(1)    # [N, 256*8]
            mid_feat = self.pool_mid(features).flatten(1)        # [N, 256*4]
            global_feat = self.pool_global(features).flatten(1)   # [N, 256*1]
            
            # Concatenate multi-scale features
            multi_scale = torch.cat([local_feat, mid_feat, global_feat], dim=1)  # [N, 256*13]
            
            # Combine and project to output dimension
            output = self.combine(multi_scale)  # [N, 256]
            outputs.append(output)
        
        # Stack back to original output format
        return torch.stack(outputs, dim=1)  # [N, seq_len, 256]


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        
        # Choose visual encoder based on flag
        self.use_searaft = getattr(opt, 'use_searaft', False)
        
        if self.use_searaft:
            # Use SEA-RAFT feature encoder
            from .searaft_encoder import SEARAFTFeatureEncoder
            self.visual_encoder = SEARAFTFeatureEncoder(opt)
        else:
            # Use original CNN encoder
            self.conv1 = conv(True, 6, 64, kernel_size=7, stride=2, dropout=0.2)
            self.conv2 = conv(True, 64, 128, kernel_size=5, stride=2, dropout=0.2)
            self.conv3 = conv(True, 128, 256, kernel_size=5, stride=2, dropout=0.2)
            self.conv3_1 = conv(True, 256, 256, kernel_size=3, stride=1, dropout=0.2)
            self.conv4 = conv(True, 256, 512, kernel_size=3, stride=2, dropout=0.2)
            self.conv4_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
            self.conv5 = conv(True, 512, 512, kernel_size=3, stride=2, dropout=0.2)
            self.conv5_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
            self.conv6 = conv(True, 512, 1024, kernel_size=3, stride=2, dropout=0.5)
            
            # Compute the shape based on diff image size
            __tmp = Variable(torch.zeros(1, 6, opt.img_w, opt.img_h))
            __tmp = self.encode_image(__tmp)
            self.visual_head = nn.Linear(int(np.prod(__tmp.size())), opt.v_f_len)
        
        # Always use flexible IMU encoder for variable-length sequences
        self.inertial_encoder = FlexibleInertialEncoder(opt)

    def forward(self, img, imu, frame_ids=None):
        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2)
        batch_size = v.size(0)
        seq_len = v.size(1)

        # Process visual features
        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        
        if self.use_searaft:
            # SEA-RAFT encoder with optional frame IDs for multi-frame
            if frame_ids is not None:
                # Create frame ID list for each pair in the batch
                batch_frame_ids = []
                for b in range(batch_size):
                    for s in range(seq_len):
                        # Use the second frame ID of each pair
                        if isinstance(frame_ids[0], list):
                            # frame_ids is [B, seq_len+1]
                            batch_frame_ids.append(frame_ids[b][s + 1])
                        else:
                            # frame_ids is [seq_len+1]
                            batch_frame_ids.append(frame_ids[s + 1])
                v = self.visual_encoder.encode_image(v, batch_frame_ids)
            else:
                v = self.visual_encoder.encode_image(v)
        else:
            # Original CNN encoder needs visual head
            v = self.encode_image(v)
            v = v.view(batch_size * seq_len, -1)  # Flatten
            v = self.visual_head(v)
        
        v = v.view(batch_size, seq_len, -1)  # (batch, seq_len, v_f_len)
        
        # IMU CNN - expects variable length per transition
        # Expected format: [batch, seq_len, num_samples, 6]
        imu_features = self.inertial_encoder(imu)
        
        return v, imu_features

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6


# The fusion module
class Fusion_module(nn.Module):
    def __init__(self, opt):
        super(Fusion_module, self).__init__()
        self.fuse_method = opt.fuse_method
        self.f_len = opt.i_f_len + opt.v_f_len
        if self.fuse_method == 'soft':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, self.f_len))
        elif self.fuse_method == 'hard':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, 2 * self.f_len))

    def forward(self, v, i):
        if self.fuse_method == 'cat':
            return torch.cat((v, i), -1)
        elif self.fuse_method == 'soft':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            return feat_cat * weights
        elif self.fuse_method == 'hard':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            weights = weights.view(v.shape[0], v.shape[1], self.f_len, 2)
            mask = F.gumbel_softmax(weights, tau=1, hard=True, dim=-1)
            return feat_cat * mask[:, :, :, 0]

# The policy network module
class PolicyNet(nn.Module):
    def __init__(self, opt):
        super(PolicyNet, self).__init__()
        in_dim = opt.rnn_hidden_size + opt.i_f_len
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 2))

    def forward(self, x, temp):
        logits = self.net(x)
        hard_mask = F.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)
        return logits, hard_mask

# The pose estimation network
class Pose_RNN(nn.Module):
    def __init__(self, opt):
        super(Pose_RNN, self).__init__()

        # The main RNN network
        f_len = opt.v_f_len + opt.i_f_len
        self.rnn = nn.LSTM(
            input_size=f_len,
            hidden_size=opt.rnn_hidden_size,
            num_layers=2,
            dropout=opt.rnn_dropout_between,
            batch_first=True)

        self.fuse = Fusion_module(opt)

        # The output networks
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        self.regressor = nn.Sequential(
            nn.Linear(opt.rnn_hidden_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6))

    def forward(self, fv, fv_alter, fi, dec, prev=None):
        if prev is not None:
            prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())
        
        # Select between fv and fv_alter
        if dec is not None:
            v_in = fv * dec[:, :, :1] + fv_alter * dec[:, :, -1:] if fv_alter is not None else fv
        else:
            v_in = fv
        fused = self.fuse(v_in, fi)
        
        out, hc = self.rnn(fused) if prev is None else self.rnn(fused, prev)
        out = self.rnn_drop_out(out)
        pose = self.regressor(out)

        hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
        return pose, hc



class DeepVIO(nn.Module):
    def __init__(self, opt):
        super(DeepVIO, self).__init__()

        self.Feature_net = Encoder(opt)
        self.Pose_net = Pose_RNN(opt)
        self.Policy_net = PolicyNet(opt)
        self.opt = opt
        
        initialization(self)

    def forward(self, img, imu, is_first=True, hc=None, temp=5, selection='gumbel-softmax', p=0.5):

        fv, fi = self.Feature_net(img, imu)
        batch_size = fv.shape[0]
        seq_len = fv.shape[1]

        poses, decisions, logits= [], [], []
        hidden = torch.zeros(batch_size, self.opt.rnn_hidden_size).to(fv.device) if hc is None else hc[0].contiguous()[:, -1, :]
        fv_alter = torch.zeros_like(fv) # zero padding in the paper, can be replaced by other 
        
        for i in range(seq_len):
            if i == 0 and is_first:
                # The first relative pose is estimated by both images and imu by default
                pose, hc = self.Pose_net(fv[:, i:i+1, :], None, fi[:, i:i+1, :], None, hc)
            else:
                if selection == 'gumbel-softmax':
                    # Otherwise, sample the decision from the policy network
                    p_in = torch.cat((fi[:, i, :], hidden), -1)
                    logit, decision = self.Policy_net(p_in.detach(), temp)
                    decision = decision.unsqueeze(1)
                    logit = logit.unsqueeze(1)
                    pose, hc = self.Pose_net(fv[:, i:i+1, :], fv_alter[:, i:i+1, :], fi[:, i:i+1, :], decision, hc)
                    decisions.append(decision)
                    logits.append(logit)
                elif selection == 'random':
                    decision = (torch.rand(fv.shape[0], 1, 2) < p).float()
                    decision[:,:,1] = 1-decision[:,:,0]
                    decision = decision.to(fv.device)
                    logit = 0.5*torch.ones((fv.shape[0], 1, 2)).to(fv.device)
                    pose, hc = self.Pose_net(fv[:, i:i+1, :], fv_alter[:, i:i+1, :], fi[:, i:i+1, :], decision, hc)
                    decisions.append(decision)
                    logits.append(logit)
            poses.append(pose)
            hidden = hc[0].contiguous()[:, -1, :]

        poses = torch.cat(poses, dim=1)
        decisions = torch.cat(decisions, dim=1)
        logits = torch.cat(logits, dim=1)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        return poses, decisions, probs, hc

class VINet(nn.Module):
    def __init__(self, opt):
        super(VINet, self).__init__()
        
        self.Feature_net = Encoder(opt)
        self.Pose_net = Pose_RNN(opt)
        self.opt = opt
        
        initialization(self)

    def forward(self, img, imu, is_first=True, hc=None):

        fv, fi = self.Feature_net(img, imu)
        batch_size = fv.shape[0]
        seq_len = fv.shape[1]

        poses = []
        hidden = torch.zeros(batch_size, self.opt.rnn_hidden_size).to(fv.device) if hc is None else hc[0].contiguous()[:, -1, :]
        fv_alter = torch.zeros_like(fv) # zero padding in the paper, can be replaced by other 
        
        for i in range(seq_len):
            if i == 0 and is_first:
                # The first relative pose is estimated by both images and imu by default
                pose, hc = self.Pose_net(fv[:, i:i+1, :], None, fi[:, i:i+1, :], None, hc)
            else:
                # Directly pass the fused features without selection
                pose, hc = self.Pose_net(fv[:, i:i+1, :], fv_alter[:, i:i+1, :], fi[:, i:i+1, :], None, hc)

            poses.append(pose)
            hidden = hc[0].contiguous()[:, -1, :]

        poses = torch.cat(poses, dim=1)

        return poses, hc


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

        # Generate causal mask
        pos_embedding = self.positional_embedding(seq_length).to(visual_inertial_features.device)
        visual_inertial_features += pos_embedding

        # Passing through the transformer encoder with the mask
        #output = self.transformer_encoder(visual_inertial_features, mask=mask, is_causal=True)
        output = self.transformer_encoder(visual_inertial_features, mask=None)

        # Pass the output through the fully connected layer
        output = self.fc(output)

        return output



class TransformerVIO(nn.Module):
    def __init__(self, opt):
        super(TransformerVIO, self).__init__()
        self.window_size = opt.seq_len
        self.Feature_net = Encoder(opt)
        self.Pose_net = PoseTransformer(opt)
        initialization(self)

    def forward(self, img, imu, frame_ids=None):
        fv, fi = self.Feature_net(img, imu, frame_ids)
        visual_inertial_feature = torch.cat([fv, fi], dim=-1) 

        # Continue processing as before
        poses = self.Pose_net(visual_inertial_feature)
        return poses


def initialization(net):
    #Initilization
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
