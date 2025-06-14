"""
Aria Latent Tester - uses KITTI tester logic but with Aria data
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.data.components.aria_kitti_format_dataset import AriaKITTIFormat
from src.utils.custom_transform import Compose, ToTensor, Resize
from src.testers.kitti_latent_tester import KITTILatentTester


class AriaLatentTester(KITTILatentTester):
    """
    Aria tester that inherits from KITTI tester but uses Aria data.
    """
    
    def __init__(self, 
                 val_seqs=['016', '017', '018', '019'],
                 data_dir='/home/external/VIFT_AEA/aria_processed',
                 seq_len=11,
                 img_w=512,
                 img_h=256,
                 wrapper_weights_path='',
                 device='cuda',
                 v_f_len=512,
                 i_f_len=256,
                 use_history_in_eval=True,
                 **kwargs):
        
        # Set Aria-specific parameters
        self.val_seqs = val_seqs
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.img_w = img_w
        self.img_h = img_h
        self.device = device
        self.v_f_len = v_f_len
        self.i_f_len = i_f_len
        self.use_history_in_eval = use_history_in_eval
        
        # Create transform
        self.transform = Compose([
            ToTensor(),
            Resize(size=(img_h, img_w))
        ])
        
        # Load wrapper model if needed
        if wrapper_weights_path and Path(wrapper_weights_path).exists():
            self._load_wrapper_model(wrapper_weights_path)
        else:
            self.wrapper_model = None
    
    def _load_wrapper_model(self, weights_path):
        """Load the Visual-Selective-VIO encoder if needed."""
        # This would load the encoder model if we need to generate features on the fly
        # For now, we assume pre-extracted features are being used
        self.wrapper_model = None
        print(f"Note: Wrapper model loading not implemented for Aria tester")
    
    def _get_test_dataloader(self, seq):
        """Create dataloader for a single Aria sequence."""
        dataset = AriaKITTIFormat(
            root=self.data_dir,
            sequence_length=self.seq_len,
            train_seqs=[seq],
            transform=self.transform
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return dataloader
    
    def test(self, net):
        """Test on all Aria sequences."""
        results = []
        
        for seq_idx, seq in enumerate(self.val_seqs):
            print(f"Testing sequence {seq_idx + 1} of {len(self.val_seqs)}: {seq}")
            
            # Get dataloader for this sequence
            dataloader = self._get_test_dataloader(seq)
            
            # Initialize lists for this sequence
            seq_results = {
                'seq': seq,
                'pred_poses': [],
                'gt_poses': [],
                'errors_t': [],
                'errors_r': []
            }
            
            # Process sequence
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=f"Sequence {seq}")):
                    imgs, imus, rots, weights = inputs
                    
                    # Move to device
                    imgs = imgs.to(self.device)
                    imus = imus.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass
                    predictions = net(imgs, imus)
                    
                    # Store results
                    seq_results['pred_poses'].append(predictions.cpu().numpy())
                    seq_results['gt_poses'].append(targets.cpu().numpy())
            
            # Concatenate results
            seq_results['pred_poses'] = np.concatenate(seq_results['pred_poses'], axis=0)
            seq_results['gt_poses'] = np.concatenate(seq_results['gt_poses'], axis=0)
            
            results.append(seq_results)
        
        return results