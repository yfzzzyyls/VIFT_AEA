#!/usr/bin/env python3
"""
Example script for batch inference using trained FlowNet-LSTM-Transformer model.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm

# Add project paths
import sys
sys.path.append(str(Path(__file__).parent))

from models.flownet_lstm_transformer import FlowNetLSTMTransformer
from configs.flownet_lstm_transformer_config import ModelConfig
from data.aria_variable_imu_dataset import AriaVariableIMUDataset, collate_variable_imu
from torch.utils.data import DataLoader
from utils.pose_utils import integrate_poses


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config or use default
    if 'config' in checkpoint:
        config = checkpoint['config']
        if hasattr(config, 'model'):
            model_config = config.model
        else:
            model_config = ModelConfig()
    else:
        model_config = ModelConfig()
    
    # Create model
    model = FlowNetLSTMTransformer(model_config)
    
    # Load weights
    state_dict = checkpoint['model_state_dict']
    # Handle DDP wrapped models
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    return model


def run_inference(model, dataloader, device):
    """Run inference on entire dataset."""
    all_predictions = []
    all_metadata = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
            # Move data to device
            images = batch['images'].to(device)
            imu_sequences = batch['imu_sequences']
            
            # Move IMU sequences to device
            for b in range(len(imu_sequences)):
                for t in range(len(imu_sequences[b])):
                    imu_sequences[b][t] = imu_sequences[b][t].to(device)
            
            # Run model
            outputs = model(images, imu_sequences)
            poses = outputs['poses']  # [B, T-1, 7]
            
            # Store predictions
            all_predictions.append(poses.cpu().numpy())
            
            # Store metadata
            for i in range(poses.shape[0]):
                metadata = {
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'sequence_length': poses.shape[1],
                    'start_idx': batch['start_indices'][i].item() if 'start_indices' in batch else None
                }
                all_metadata.append(metadata)
    
    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    return all_predictions, all_metadata


def save_results(predictions, metadata, output_dir):
    """Save inference results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw predictions
    np.save(output_dir / 'predictions.npy', predictions)
    
    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save integrated trajectories
    trajectories = []
    for i in range(len(predictions)):
        traj = integrate_poses(predictions[i])
        trajectories.append(traj)
    
    np.save(output_dir / 'trajectories.npy', np.array(trajectories))
    
    print(f"Results saved to {output_dir}")
    print(f"- predictions.npy: {predictions.shape}")
    print(f"- trajectories.npy: {len(trajectories)} trajectories")
    print(f"- metadata.json: {len(metadata)} entries")


def main():
    parser = argparse.ArgumentParser(description="Run inference with FlowNet-LSTM-Transformer")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='../aria_processed',
                       help='Path to processed Aria data')
    parser.add_argument('--output-dir', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for inference')
    parser.add_argument('--sequence-length', type=int, default=31,
                       help='Sequence length')
    parser.add_argument('--image-height', type=int, default=704,
                       help='Image height')
    parser.add_argument('--image-width', type=int, default=704,
                       help='Image width')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Create dataset
    dataset = AriaVariableIMUDataset(
        data_dir=args.data_dir,
        split=args.split,
        variable_length=False,  # Fixed length for inference
        sequence_length=args.sequence_length,
        image_size=(args.image_height, args.image_width),
        stride=args.sequence_length  # Non-overlapping sequences
    )
    
    print(f"Dataset: {len(dataset)} sequences")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_variable_imu
    )
    
    # Run inference
    predictions, metadata = run_inference(model, dataloader, device)
    
    # Save results
    save_results(predictions, metadata, args.output_dir)
    
    print("\nInference completed!")


if __name__ == "__main__":
    main()