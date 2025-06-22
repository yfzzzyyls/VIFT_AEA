"""
Tester for Aria data using pre-extracted latent features.
This bypasses the VIFT feature extraction and uses latents directly.
"""

from .base_tester import BaseTester
import torch
import numpy as np
from typing import Dict, Any
from torch.utils.data import DataLoader
from src.data.components.aria_latent_dataset_kitti_format import AriaLatentKITTIFormat
from dataclasses import dataclass
import os

class AriaLatentDirectTester(BaseTester):
    """
    Tester that uses pre-extracted Aria latent features directly.
    Compatible with KITTI-trained models that expect latent inputs.
    """
    def __init__(self, val_seqs, data_dir, seq_len, img_w, img_h, wrapper_weights_path, device, v_f_len, i_f_len, use_history_in_eval=False, **kwargs):
        super().__init__()
        self.val_seq = val_seqs
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.device = device
        self.use_history_in_eval = use_history_in_eval
        
        # Create data loaders for pre-extracted latent features
        self.dataloaders = []
        for seq in val_seqs:
            dataset = AriaLatentKITTIFormat(
                root=os.path.join(data_dir, '..', 'aria_latent_kitti_format'),
                sequences=[seq]
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            self.dataloaders.append(loader)
            print(f"Loaded sequence {seq} with {len(dataset)} samples")
    
    def test(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Test the model on pre-extracted latent features"""
        results = {}
        
        for i, (seq, dataloader) in enumerate(zip(self.val_seq, self.dataloaders)):
            print(f"\nTesting sequence {i+1} of {len(self.val_seq)}: {seq}")
            
            pose_predictions = []
            ground_truth_poses = []
            
            with torch.no_grad():
                for batch_idx, (latent_data, gt_pose) in enumerate(dataloader):
                    # latent_data is tuple of (features, gt, rot, weight)
                    features, _, rot, weight = latent_data
                    
                    # Move to device
                    features = features.to(self.device)
                    gt_pose = gt_pose.to(self.device)
                    
                    # For pose transformer, we just need the features
                    # The model expects input in format (features, rot, weight)
                    x = (features, rot.to(self.device), weight.to(self.device))
                    
                    # Get prediction
                    pred = model(x, gt_pose)
                    
                    # Store results
                    pose_predictions.append(pred.cpu().numpy())
                    ground_truth_poses.append(gt_pose.cpu().numpy())
                    
                    if batch_idx % 100 == 0:
                        print(f"  Processed {batch_idx}/{len(dataloader)} batches")
            
            # Concatenate all predictions
            pose_predictions = np.concatenate(pose_predictions, axis=0)
            ground_truth_poses = np.concatenate(ground_truth_poses, axis=0)
            
            results[seq] = {
                'estimated_poses': pose_predictions,
                'gt_poses': ground_truth_poses
            }
            
            print(f"  Completed sequence {seq}: {pose_predictions.shape[0]} predictions")
        
        return results
    
    def save_results(self, results: Dict[str, Any], save_dir: str):
        """Save test results"""
        os.makedirs(save_dir, exist_ok=True)
        
        for seq_name, seq_data in results.items():
            # Save predictions and ground truth
            np.save(os.path.join(save_dir, f'{seq_name}_estimated_poses.npy'), seq_data['estimated_poses'])
            np.save(os.path.join(save_dir, f'{seq_name}_gt_poses.npy'), seq_data['gt_poses'])
            
            # Calculate and print basic metrics
            predictions = seq_data['estimated_poses']
            ground_truth = seq_data['gt_poses']
            
            # Translation error
            trans_errors = np.linalg.norm(predictions[:, :3] - ground_truth[:, :3], axis=1)
            
            # Rotation error (simple L2 for now)
            rot_errors = np.linalg.norm(predictions[:, 3:] - ground_truth[:, 3:], axis=1)
            
            print(f"\nSequence {seq_name} Results:")
            print(f"  Translation Error - Mean: {np.mean(trans_errors)*100:.2f}cm, Max: {np.max(trans_errors)*100:.2f}cm")
            print(f"  Rotation Error - Mean: {np.mean(rot_errors):.4f}, Max: {np.max(rot_errors):.4f}")