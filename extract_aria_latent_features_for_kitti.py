#!/usr/bin/env python3
"""
Extract latent features from Aria data using the Visual-Selective-VIO encoder
in a format compatible with KITTI-trained VIFT model.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.components.aria_kitti_format_dataset import AriaKITTIFormat
from src.utils import custom_transform
from src.models.components.vsvio import Encoder


class ObjFromDict:
    """Helper to convert dictionary to object."""
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)


class FeatureEncodingModel(torch.nn.Module):
    """Wrapper model for Visual-Selective-VIO encoder."""
    def __init__(self, params):
        super(FeatureEncodingModel, self).__init__()
        self.Feature_net = Encoder(params)
        
    def forward(self, imgs, imus):
        feat_v, feat_i = self.Feature_net(imgs, imus)
        return feat_v, feat_i


def extract_features_for_sequences(sequences, data_root, save_dir, batch_size=1):
    """Extract latent features for given Aria sequences."""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Define transforms (same as KITTI)
    transform = custom_transform.Compose([
        custom_transform.ToTensor(),
        custom_transform.Resize((256, 512))
    ])
    
    # Initialize the encoder model
    params = {
        "img_w": 512, 
        "img_h": 256, 
        "v_f_len": 512, 
        "i_f_len": 256,
        "imu_dropout": 0.1, 
        "seq_len": 11
    }
    params = ObjFromDict(params)
    
    model = FeatureEncodingModel(params)
    
    # Load pretrained weights
    pretrained_path = "pretrained_models/vf_512_if_256_3e-05.model"
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained model not found at {pretrained_path}")
        
    pretrained_w = torch.load(pretrained_path, map_location='cpu')
    model_dict = model.state_dict()
    update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}
    
    # Verify all parameters are loaded
    assert len(update_dict) == len(model_dict), "Some weights are not loaded"
    
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    
    # Freeze weights
    for param in model.Feature_net.parameters():
        param.requires_grad = False
    
    model.eval()
    model.to("cuda")
    
    # Process each sequence
    for seq in sequences:
        print(f"\nProcessing sequence {seq}...")
        
        # Create dataset for this sequence
        dataset = AriaKITTIFormat(
            root=data_root,
            sequence_length=11,
            train_seqs=[seq],
            transform=transform
        )
        
        # Create dataloader
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # Create sequence directory
        seq_save_dir = os.path.join(save_dir, seq)
        os.makedirs(seq_save_dir, exist_ok=True)
        
        # Extract features
        with torch.no_grad():
            for i, ((imgs, imus, rot, w), gts) in enumerate(tqdm(loader, desc=f"Sequence {seq}")):
                # Move to GPU
                imgs = imgs.to("cuda").float()
                imus = imus.to("cuda").float()
                
                # Extract features
                feat_v, feat_i = model(imgs, imus)
                
                # Concatenate features (same as KITTI)
                latent_vector = torch.cat((feat_v, feat_i), 2)
                latent_vector = latent_vector.squeeze(0)
                
                # Save latent features and metadata
                np.save(os.path.join(seq_save_dir, f"{i}.npy"), 
                       latent_vector.cpu().detach().numpy())
                np.save(os.path.join(seq_save_dir, f"{i}_gt.npy"), 
                       gts.cpu().detach().numpy())
                np.save(os.path.join(seq_save_dir, f"{i}_rot.npy"), 
                       rot.cpu().detach().numpy())
                np.save(os.path.join(seq_save_dir, f"{i}_w.npy"), 
                       w.cpu().detach().numpy())
        
        print(f"Saved {len(loader)} feature files for sequence {seq}")


def main():
    # Define sequences
    test_sequences = ['016', '017', '018', '019']
    
    # Paths
    data_root = '/home/external/VIFT_AEA/aria_processed'
    save_dir = '/home/external/VIFT_AEA/aria_latent_kitti_format'
    
    print(f"Extracting features from {data_root}")
    print(f"Saving to {save_dir}")
    print(f"Test sequences: {test_sequences}")
    
    # Extract features
    extract_features_for_sequences(test_sequences, data_root, save_dir)
    
    print("\nFeature extraction complete!")
    print(f"Features saved to: {save_dir}")


if __name__ == "__main__":
    main()