#!/usr/bin/env python3
"""
Diagnostic script to investigate rotation error issues
"""

import os
import sys
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append('src')

from src.models.multihead_vio import MultiHeadVIOModel
from train_pretrained_relative import RelativePoseDataset
from torch.utils.data import DataLoader


def quaternion_angle(q1, q2):
    """Calculate angle between two quaternions (XYZW convention)."""
    # Normalize
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    q2 = q2 / (np.linalg.norm(q2) + 1e-8)
    
    # Dot product
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    
    # Handle double cover
    if dot < 0:
        dot = -dot
    
    # Angle
    angle = 2 * np.arccos(dot)
    return angle


def analyze_dataset():
    """Analyze the dataset to understand rotation patterns."""
    print("\n=== DATASET ANALYSIS ===")
    
    # Load test dataset
    test_dataset = RelativePoseDataset(
        "aria_latent_data_pretrained/test",
        pose_scale=100.0,
        max_samples=10  # Just look at first 10
    )
    
    # Analyze rotation patterns
    all_rotations = []
    
    for i in range(min(10, len(test_dataset))):
        features, imus, poses = test_dataset[i]
        
        print(f"\nSample {i}:")
        print(f"  Features shape: {features.shape}")
        print(f"  Poses shape: {poses.shape}")
        
        # Check first pose (should be identity)
        first_pose = poses[0]
        print(f"  First pose translation: {first_pose[:3].numpy()}")
        print(f"  First pose quaternion: {first_pose[3:].numpy()}")
        
        # Analyze rotation magnitudes
        for j in range(1, len(poses)):
            quat = poses[j, 3:].numpy()
            # Check if quaternion is normalized
            norm = np.linalg.norm(quat)
            if abs(norm - 1.0) > 0.01:
                print(f"  WARNING: Frame {j} quaternion not normalized! Norm = {norm}")
            
            # Calculate rotation angle from identity
            identity = np.array([0, 0, 0, 1])  # XYZW
            angle = quaternion_angle(quat, identity)
            all_rotations.append(np.degrees(angle))
            
            if j <= 3:  # Show first few
                print(f"  Frame {j}: quat={quat}, angle={np.degrees(angle):.4f}°")
    
    # Statistics
    all_rotations = np.array(all_rotations)
    print(f"\nRotation statistics (degrees):")
    print(f"  Mean: {np.mean(all_rotations):.4f}")
    print(f"  Std: {np.std(all_rotations):.4f}")
    print(f"  Max: {np.max(all_rotations):.4f}")
    print(f"  95%: {np.percentile(all_rotations, 95):.4f}")


def test_model_predictions(checkpoint_path):
    """Test model predictions to understand the issue."""
    print("\n=== MODEL PREDICTION ANALYSIS ===")
    
    # Load model
    model = MultiHeadVIOModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Create test dataset
    test_dataset = RelativePoseDataset(
        "aria_latent_data_pretrained/test",
        pose_scale=100.0,
        max_samples=5
    )
    
    # Test on individual samples
    for i in range(min(5, len(test_dataset))):
        features, imus, poses = test_dataset[i]
        
        # Prepare batch
        batch = {
            'images': features.unsqueeze(0),
            'imus': imus.unsqueeze(0),
            'poses': poses.unsqueeze(0)
        }
        
        if torch.cuda.is_available():
            batch = {k: v.cuda() for k, v in batch.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(batch)
        
        # Extract predictions
        pred_rotation = outputs['rotation'][0].cpu().numpy()  # [11, 4]
        gt_rotation = poses[:, 3:].numpy()  # [11, 4]
        
        print(f"\nSample {i}:")
        
        # Compare frame by frame (skip first frame)
        rotation_errors = []
        for j in range(1, len(pred_rotation)):
            pred_q = pred_rotation[j]
            gt_q = gt_rotation[j]
            
            # Normalize predictions
            pred_q = pred_q / (np.linalg.norm(pred_q) + 1e-8)
            
            # Calculate error
            angle_error = quaternion_angle(pred_q, gt_q)
            rotation_errors.append(np.degrees(angle_error))
            
            if j <= 3:  # Show first few
                print(f"  Frame {j}:")
                print(f"    GT quaternion:   {gt_q}")
                print(f"    Pred quaternion: {pred_q}")
                print(f"    Error: {np.degrees(angle_error):.4f}°")
        
        mean_error = np.mean(rotation_errors)
        print(f"  Mean rotation error: {mean_error:.4f}°")


def check_loss_function():
    """Check if the loss function is working correctly."""
    print("\n=== LOSS FUNCTION CHECK ===")
    
    # Create dummy data
    batch_size = 2
    seq_len = 11
    
    # Random predictions and targets
    pred_rotation = torch.randn(batch_size * (seq_len - 1), 4)
    pred_rotation = pred_rotation / torch.norm(pred_rotation, dim=-1, keepdim=True)
    
    target_rotation = torch.randn(batch_size * (seq_len - 1), 4)
    target_rotation = target_rotation / torch.norm(target_rotation, dim=-1, keepdim=True)
    
    pred_translation = torch.randn(batch_size * (seq_len - 1), 3)
    target_translation = torch.randn(batch_size * (seq_len - 1), 3)
    
    # Test loss
    from src.metrics.arvr_loss_wrapper import ARVRLossWrapper
    loss_fn = ARVRLossWrapper()
    
    loss_dict = loss_fn(pred_rotation, target_rotation, pred_translation, target_translation)
    
    print(f"Loss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.6f}")
    
    # Test with identical inputs (should be near zero)
    print("\nWith identical inputs:")
    loss_dict_zero = loss_fn(target_rotation, target_rotation, target_translation, target_translation)
    for k, v in loss_dict_zero.items():
        print(f"  {k}: {v.item():.6f}")


def analyze_pretrained_features():
    """Check if pretrained features contain rotation information."""
    print("\n=== PRETRAINED FEATURE ANALYSIS ===")
    
    # Load a few samples
    test_dir = "aria_latent_data_pretrained/test"
    
    for i in range(3):
        features = np.load(os.path.join(test_dir, f"{i}.npy"))
        poses = np.load(os.path.join(test_dir, f"{i}_gt.npy"))
        
        print(f"\nSample {i}:")
        print(f"  Feature shape: {features.shape}")
        print(f"  Feature stats: mean={np.mean(features):.4f}, std={np.std(features):.4f}")
        print(f"  Feature range: [{np.min(features):.4f}, {np.max(features):.4f}]")
        
        # Check if features vary between frames
        feature_diff = np.diff(features, axis=0)
        print(f"  Inter-frame feature variation: {np.mean(np.abs(feature_diff)):.6f}")
        
        # Check correlation with rotation changes
        rotation_changes = []
        for j in range(1, len(poses)):
            angle = quaternion_angle(poses[j, 3:], poses[j-1, 3:])
            rotation_changes.append(angle)
        rotation_changes = np.array(rotation_changes)
        
        # Simple correlation check
        feature_changes = np.mean(np.abs(feature_diff), axis=1)
        if len(feature_changes) == len(rotation_changes):
            correlation = np.corrcoef(feature_changes, rotation_changes)[0, 1]
            print(f"  Feature-rotation correlation: {correlation:.4f}")


def main():
    """Run all diagnostics."""
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose rotation error issues')
    parser.add_argument('--checkpoint', type=str, 
                       default='logs/checkpoints_lite_scale_100.0/last.ckpt',
                       help='Path to model checkpoint')
    args = parser.parse_args()
    
    print("=" * 60)
    print("ROTATION ERROR DIAGNOSTIC")
    print("=" * 60)
    
    # 1. Analyze dataset
    analyze_dataset()
    
    # 2. Check loss function
    check_loss_function()
    
    # 3. Analyze pretrained features
    analyze_pretrained_features()
    
    # 4. Test model predictions
    if os.path.exists(args.checkpoint):
        test_model_predictions(args.checkpoint)
    else:
        print(f"\nWARNING: Checkpoint not found at {args.checkpoint}")
        print("Skipping model prediction analysis")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()