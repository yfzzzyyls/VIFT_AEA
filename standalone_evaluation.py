#!/usr/bin/env python3
"""
Standalone evaluation script for VIFT-AEA model on Aria test data.
This script bypasses Lightning and KITTI dependencies for robust evaluation.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class PoseTransformer(nn.Module):
    """Simplified PoseTransformer model for evaluation"""
    
    def __init__(self, d_model=512, nhead=8, num_layers=6, input_dim=768, output_dim=6):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        x = self.transformer(x)       # [batch_size, seq_len, d_model]
        x = self.output_head(x)       # [batch_size, seq_len, output_dim]
        return x

class AriaTestDataset(torch.utils.data.Dataset):
    """Dataset for loading Aria test data from .npy files"""
    
    def __init__(self, test_dir):
        self.test_dir = Path(test_dir)
        
        # Find all feature files (files without '_' in name)
        self.feature_files = []
        for f in os.listdir(self.test_dir):
            if f.endswith('.npy') and '_' not in f:
                idx = int(f.replace('.npy', ''))
                gt_file = self.test_dir / f"{idx}_gt.npy"
                if gt_file.exists():
                    self.feature_files.append(idx)
        
        self.feature_files.sort()
        print(f"📊 Found {len(self.feature_files)} test samples")
        
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        file_idx = self.feature_files[idx]
        
        # Load features and targets
        features = np.load(self.test_dir / f"{file_idx}.npy")
        targets = np.load(self.test_dir / f"{file_idx}_gt.npy")
        
        return torch.from_numpy(features).float(), torch.from_numpy(targets).float()

def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from Lightning checkpoint"""
    print(f"📂 Loading checkpoint from: {checkpoint_path}")
    
    try:
        # Load checkpoint
        # Load checkpoint with weights_only=False to handle functools.partial objects
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract model state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'net.' prefix if present (Lightning adds this)
        clean_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('net.'):
                clean_key = key[4:]  # Remove 'net.' prefix
            else:
                clean_key = key
            clean_state_dict[clean_key] = value
        
        # Create model (use dimensions from first test sample to infer input_dim)
        model = PoseTransformer(
            d_model=512,
            nhead=8,
            num_layers=6,
            input_dim=768,  # 512 (visual) + 256 (IMU)
            output_dim=6    # 6-DOF pose
        )
        
        # Load state dict
        model.load_state_dict(clean_state_dict, strict=False)
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def compute_metrics(predictions, targets):
    """Compute evaluation metrics"""
    
    # Convert to numpy if needed
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    # Flatten for metric computation
    pred_flat = predictions.reshape(-1, predictions.shape[-1])
    target_flat = targets.reshape(-1, targets.shape[-1])
    
    metrics = {}
    
    # Mean Squared Error
    mse = np.mean((pred_flat - target_flat) ** 2)
    metrics['mse'] = mse
    
    # Root Mean Squared Error
    metrics['rmse'] = np.sqrt(mse)
    
    # Mean Absolute Error
    metrics['mae'] = np.mean(np.abs(pred_flat - target_flat))
    
    # Per-dimension metrics
    for i in range(pred_flat.shape[1]):
        dim_mse = np.mean((pred_flat[:, i] - target_flat[:, i]) ** 2)
        metrics[f'mse_dim_{i}'] = dim_mse
        metrics[f'rmse_dim_{i}'] = np.sqrt(dim_mse)
        metrics[f'mae_dim_{i}'] = np.mean(np.abs(pred_flat[:, i] - target_flat[:, i]))
    
    # Translation vs Rotation (assuming first 3 dims are translation, last 3 are rotation)
    if pred_flat.shape[1] == 6:
        trans_mse = np.mean((pred_flat[:, :3] - target_flat[:, :3]) ** 2)
        rot_mse = np.mean((pred_flat[:, 3:] - target_flat[:, 3:]) ** 2)
        metrics['translation_mse'] = trans_mse
        metrics['rotation_mse'] = rot_mse
        metrics['translation_rmse'] = np.sqrt(trans_mse)
        metrics['rotation_rmse'] = np.sqrt(rot_mse)
    
    return metrics

def evaluate_model(model, test_loader, device):
    """Evaluate model on test data"""
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    print("🔍 Running inference on test data...")
    
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
            features = features.to(device)
            targets = targets.to(device)
            
            # Forward pass
            predictions = model(features)
            
            # Store results
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    print(f"📐 Prediction shape: {all_predictions.shape}")
    print(f"📐 Target shape: {all_targets.shape}")
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets)
    
    return metrics, all_predictions, all_targets

def print_metrics(metrics):
    """Print evaluation metrics in a nice format"""
    
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    
    # Overall metrics
    print(f"🎯 Overall Performance:")
    print(f"   MSE:  {metrics['mse']:.6f}")
    print(f"   RMSE: {metrics['rmse']:.6f}")
    print(f"   MAE:  {metrics['mae']:.6f}")
    
    # Translation vs Rotation
    if 'translation_mse' in metrics:
        print(f"\n🚀 Translation (xyz):")
        print(f"   MSE:  {metrics['translation_mse']:.6f}")
        print(f"   RMSE: {metrics['translation_rmse']:.6f}")
        
        print(f"\n🔄 Rotation (rpy):")
        print(f"   MSE:  {metrics['rotation_mse']:.6f}")
        print(f"   RMSE: {metrics['rotation_rmse']:.6f}")
    
    # Per-dimension breakdown
    print(f"\n📏 Per-Dimension Breakdown:")
    for i in range(6):  # Assuming 6-DOF pose
        dim_name = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz'][i]
        if f'mse_dim_{i}' in metrics:
            print(f"   {dim_name}: MSE={metrics[f'mse_dim_{i}']:.6f}, "
                  f"RMSE={metrics[f'rmse_dim_{i}']:.6f}, "
                  f"MAE={metrics[f'mae_dim_{i}']:.6f}")
    
    print("="*60)

def save_results(metrics, predictions, targets, output_file):
    """Save evaluation results to file"""
    
    results = {
        'metrics': {k: float(v) for k, v in metrics.items()},
        'summary': {
            'num_samples': predictions.shape[0],
            'sequence_length': predictions.shape[1],
            'pose_dimensions': predictions.shape[2],
            'total_predictions': predictions.shape[0] * predictions.shape[1]
        }
    }
    
    # Save metrics to JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save raw predictions and targets
    base_name = output_file.replace('.json', '')
    np.save(f"{base_name}_predictions.npy", predictions.numpy())
    np.save(f"{base_name}_targets.npy", targets.numpy())
    
    print(f"💾 Results saved to: {output_file}")
    print(f"💾 Predictions saved to: {base_name}_predictions.npy")
    print(f"💾 Targets saved to: {base_name}_targets.npy")

def main():
    parser = argparse.ArgumentParser(description='Standalone evaluation for VIFT-AEA')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--test_data', required=True, help='Path to test data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use for evaluation')
    parser.add_argument('--output', default='evaluation_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🖥️ Using device: {device}")
    
    # Check paths
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    if not os.path.exists(args.test_data):
        print(f"❌ Test data directory not found: {args.test_data}")
        return
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, device)
    if model is None:
        print("❌ Failed to load model")
        return
    
    # Create dataset and dataloader
    dataset = AriaTestDataset(args.test_data)
    test_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    if len(dataset) == 0:
        print("❌ No test samples found!")
        return
    
    # Run evaluation
    metrics, predictions, targets = evaluate_model(model, test_loader, device)
    
    # Print results
    print_metrics(metrics)
    
    # Save results
    save_results(metrics, predictions, targets, args.output)
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"📊 Evaluated {len(dataset)} test samples")
    print(f"📄 Results saved to: {args.output}")

if __name__ == "__main__":
    main()
