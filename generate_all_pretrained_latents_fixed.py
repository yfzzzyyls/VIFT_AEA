#!/usr/bin/env python3
"""
Generate pretrained latent features for all splits with corrected relative poses.
This version directly outputs data in the correct format, eliminating the need for preprocessing.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the corrected relative pose conversion
from train_pretrained_relative import convert_absolute_to_relative

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_split(split: str, model_path: str, data_dir: str, output_dir: str, pose_scale: float = 100.0):
    """Process a single data split and save with corrected relative poses."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {split} split")
    logger.info(f"{'='*60}")
    
    # Import here to avoid loading before needed
    from generate_pretrained_latents import (
        load_visual_selective_vio_model,
        generate_latent_features,
        AriaDataset
    )
    
    # Create output directory
    output_path = Path(output_dir) / split
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load the Visual-Selective-VIO model
    logger.info("Loading Visual-Selective-VIO model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visual_encoder = load_visual_selective_vio_model(model_path, device)
    
    # Create dataset
    dataset = AriaDataset(f"{data_dir}/{split}")
    logger.info(f"Found {len(dataset)} sequences")
    
    # Process sequences
    sample_idx = 0
    rotation_angles = []
    translation_norms = []
    
    for seq_idx in tqdm(range(len(dataset)), desc=f"Processing {split} sequences"):
        try:
            # Get sequence data
            visual_data, _, poses = dataset[seq_idx]
            
            # Generate latent features using pretrained model
            latent_features = generate_latent_features(
                visual_encoder,
                visual_data,
                device,
                sequence_length=11,
                stride=5
            )
            
            # Convert poses to numpy and process in subsequences
            poses_np = poses.numpy()  # Full sequence poses
            
            # Process each subsequence
            for i, features in enumerate(latent_features):
                # Get corresponding pose subsequence
                start_idx = i * 5
                end_idx = start_idx + 11
                
                if end_idx > len(poses_np):
                    break
                
                # Extract absolute poses for this subsequence
                absolute_poses = poses_np[start_idx:end_idx]
                
                # Convert to relative poses using the fixed quaternion handling
                relative_poses = convert_absolute_to_relative(absolute_poses)
                
                # Scale translation from meters to centimeters
                relative_poses[:, :3] *= pose_scale
                
                # Save data
                np.save(output_path / f"{sample_idx}.npy", features)
                np.save(output_path / f"{sample_idx}_gt.npy", relative_poses)
                
                # Create dummy rotation and weight files for compatibility
                np.save(output_path / f"{sample_idx}_rot.npy", np.eye(3))
                np.save(output_path / f"{sample_idx}_w.npy", np.ones(1))
                
                # Collect statistics (skip first frame which is always identity)
                for j in range(1, len(relative_poses)):
                    # Translation norm
                    trans_norm = np.linalg.norm(relative_poses[j, :3])
                    translation_norms.append(trans_norm)
                    
                    # Rotation angle from quaternion (XYZW format)
                    quat = relative_poses[j, 3:]
                    angle = 2 * np.arccos(np.clip(quat[3], -1, 1)) * 180 / np.pi
                    rotation_angles.append(angle)
                
                sample_idx += 1
                
        except Exception as e:
            logger.error(f"Error processing sequence {seq_idx}: {e}")
            continue
    
    # Print statistics
    if rotation_angles:
        rotation_angles = np.array(rotation_angles)
        translation_norms = np.array(translation_norms)
        
        logger.info(f"\nDataset Statistics for {split}:")
        logger.info(f"Rotation angles (degrees):")
        logger.info(f"  Mean: {np.mean(rotation_angles):.4f}")
        logger.info(f"  Std:  {np.std(rotation_angles):.4f}")
        logger.info(f"  Max:  {np.max(rotation_angles):.4f}")
        logger.info(f"  95%:  {np.percentile(rotation_angles, 95):.4f}")
        
        logger.info(f"\nTranslation norms (cm):")
        logger.info(f"  Mean: {np.mean(translation_norms):.4f}")
        logger.info(f"  Std:  {np.std(translation_norms):.4f}")
        logger.info(f"  Max:  {np.max(translation_norms):.4f}")
        logger.info(f"  95%:  {np.percentile(translation_norms, 95):.4f}")
    
    logger.info(f"\n✓ Completed {split} split: {sample_idx} samples saved")
    return sample_idx


def process_with_custom_splits(model_path: str, data_dir: str, output_dir: str, 
                              pose_scale: float, train_ratio: float, 
                              val_ratio: float, test_ratio: float):
    """Process all data with custom train/val/test splits."""
    
    from generate_pretrained_latents import (
        load_visual_selective_vio_model,
        generate_latent_features,
        AriaDataset
    )
    
    logger.info("Processing with custom splits...")
    
    # Collect all sequences from all directories
    all_sequences = []
    for split_name in ['train', 'val', 'test']:
        split_path = f"{data_dir}/{split_name}"
        if os.path.exists(split_path):
            dataset = AriaDataset(split_path)
            for i in range(len(dataset)):
                all_sequences.append((split_path, i))
    
    if not all_sequences:
        logger.error("No sequences found!")
        return 0
    
    # Shuffle for random split
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(all_sequences)
    
    # Calculate split indices with proper rounding
    total_seqs = len(all_sequences)
    train_count = round(total_seqs * train_ratio)
    val_count = round(total_seqs * val_ratio)
    test_count = total_seqs - train_count - val_count  # Remainder goes to test
    
    # Ensure we don't have negative test count
    if test_count < 0:
        # Adjust validation count if needed
        val_count = total_seqs - train_count
        test_count = 0
    
    train_end = train_count
    val_end = train_end + val_count
    
    # Split sequences
    splits = {
        'train': all_sequences[:train_end],
        'val': all_sequences[train_end:val_end],
        'test': all_sequences[val_end:]
    }
    
    logger.info(f"Total sequences: {total_seqs}")
    logger.info(f"Train sequences: {len(splits['train'])} ({len(splits['train'])/total_seqs*100:.1f}%)")
    logger.info(f"Val sequences: {len(splits['val'])} ({len(splits['val'])/total_seqs*100:.1f}%)")
    logger.info(f"Test sequences: {len(splits['test'])} ({len(splits['test'])/total_seqs*100:.1f}%)")
    
    # Load model once
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visual_encoder = load_visual_selective_vio_model(model_path, device)
    
    # Process each split
    total_samples = 0
    
    for split_name, sequences in splits.items():
        output_path = Path(output_dir) / split_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        sample_idx = 0
        rotation_angles = []
        translation_norms = []
        
        logger.info(f"\nProcessing {split_name} split...")
        
        for data_path, seq_idx in tqdm(sequences, desc=f"Processing {split_name}"):
            try:
                # Load sequence
                dataset = AriaDataset(data_path)
                visual_data, _, poses = dataset[seq_idx]
                
                # Generate features
                latent_features = generate_latent_features(
                    visual_encoder, visual_data, device,
                    sequence_length=11, stride=5
                )
                
                # Process subsequences
                poses_np = poses.numpy()
                
                for i, features in enumerate(latent_features):
                    start_idx = i * 5
                    end_idx = start_idx + 11
                    
                    if end_idx > len(poses_np):
                        break
                    
                    # Convert to relative poses
                    absolute_poses = poses_np[start_idx:end_idx]
                    relative_poses = convert_absolute_to_relative(absolute_poses)
                    relative_poses[:, :3] *= pose_scale
                    
                    # Save
                    np.save(output_path / f"{sample_idx}.npy", features)
                    np.save(output_path / f"{sample_idx}_gt.npy", relative_poses)
                    np.save(output_path / f"{sample_idx}_rot.npy", np.eye(3))
                    np.save(output_path / f"{sample_idx}_w.npy", np.ones(1))
                    
                    # Statistics
                    for j in range(1, len(relative_poses)):
                        trans_norm = np.linalg.norm(relative_poses[j, :3])
                        translation_norms.append(trans_norm)
                        quat = relative_poses[j, 3:]
                        angle = 2 * np.arccos(np.clip(quat[3], -1, 1)) * 180 / np.pi
                        rotation_angles.append(angle)
                    
                    sample_idx += 1
                    
            except Exception as e:
                logger.error(f"Error processing sequence: {e}")
                continue
        
        logger.info(f"Completed {split_name}: {sample_idx} samples")
        total_samples += sample_idx
    
    return total_samples


def main():
    """Generate all pretrained latent features with corrected relative poses."""
    
    import argparse
    parser = argparse.ArgumentParser(description='Generate pretrained features with corrected relative poses')
    parser.add_argument('--model_path', type=str, default="pretrained_models/vf_512_if_256_3e-05.model",
                       help='Path to Visual-Selective-VIO model')
    parser.add_argument('--data_dir', type=str, default="data/aria_processed",
                       help='Input data directory')
    parser.add_argument('--output_dir', type=str, default="aria_latent_data_fixed",
                       help='Output directory for features')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training data ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation data ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Test data ratio (default: 0.2)')
    parser.add_argument('--pose_scale', type=float, default=100.0,
                       help='Pose scale factor (default: 100.0 for meter to cm)')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        logger.error(f"Train/val/test ratios must sum to 1.0, got {total_ratio}")
        return
    
    # Configuration
    model_path = args.model_path
    data_dir = args.data_dir
    output_dir = args.output_dir
    pose_scale = args.pose_scale
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        logger.error("Please download the Visual-Selective-VIO model first.")
        return
    
    # Check model size
    model_size = os.path.getsize(model_path)
    if model_size < 100_000_000:  # Less than 100MB
        logger.error(f"Model file seems corrupted (size: {model_size} bytes)")
        logger.error("Expected size: ~185MB. Please re-download the model.")
        return
    
    logger.info(f"Model found: {model_path} ({model_size / 1024 / 1024:.1f} MB)")
    
    # Check if we need to reorganize data based on custom splits
    if args.train_ratio != 0.7 or args.val_ratio != 0.1 or args.test_ratio != 0.2:
        logger.info(f"Using custom split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
        # Process with custom splits
        total_samples = process_with_custom_splits(
            model_path, data_dir, output_dir, pose_scale,
            args.train_ratio, args.val_ratio, args.test_ratio
        )
    else:
        # Process existing splits
        splits = ['train', 'val', 'test']
        total_samples = 0
        
        for split in splits:
            split_path = f"{data_dir}/{split}"
            if not os.path.exists(split_path):
                logger.warning(f"Split directory not found: {split_path}")
                continue
                
            num_samples = process_split(split, model_path, data_dir, output_dir, pose_scale)
            total_samples += num_samples
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Feature generation complete!")
    logger.info(f"Total samples generated: {total_samples}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*60}")
    
    logger.info("\nNext steps:")
    logger.info("1. Train the model with fixed data:")
    logger.info(f"   python train_pretrained_relative.py --data_dir {output_dir}")
    logger.info("\n2. Evaluate the trained model:")
    logger.info("   python evaluate_with_metrics.py --checkpoint logs/checkpoints_lite_scale_100.0/best_model.ckpt --data_dir aria_latent_data_fixed")


if __name__ == "__main__":
    main()