#!/usr/bin/env python3
"""
Corrected evaluation script that uses the exact same model architecture as training
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the actual model used in training
from models.components.pose_transformer import PoseTransformer

class AriaTestDataset(torch.utils.data.Dataset):
    """Dataset for loading Aria test data from .npy files"""
    
    def __init__(self, test_dir):
        self.test_dir = Path(test_dir)
        self.samples = []
        
        # Load all test samples
        files = [f for f in self.test_dir.iterdir() if f.name.endswith('.npy') and not f.name.endswith('_gt.npy') and not f.name.endswith('_rot.npy') and not f.name.endswith('_w.npy')]
        
        for file in sorted(files, key=lambda x: int(x.stem)):
            idx = int(file.stem)
            feature_file = self.test_dir / f"{idx}.npy"
            gt_file = self.test_dir / f"{idx}_gt.npy"
            
            if feature_file.exists() and gt_file.exists():
                self.samples.append(idx)
        
        print(f"📊 Loaded {len(self.samples)} test samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_idx = self.samples[idx]
        
        # Load features and ground truth
        features = np.load(self.test_dir / f"{sample_idx}.npy")
        gt = np.load(self.test_dir / f"{sample_idx}_gt.npy")
        
        return torch.from_numpy(features).float(), torch.from_numpy(gt).float()

def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from Lightning checkpoint with correct architecture"""
    print(f"📂 Loading checkpoint from: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Create model with the correct architecture used in training
        model = PoseTransformer(
            input_dim=768,      # 512 (visual) + 256 (IMU)
            embedding_dim=128,  # From training config
            num_layers=2,       # From training config  
            nhead=8,
            dim_feedforward=512,
            dropout=0.1
        )
        
        # Extract state dict (Lightning adds 'net.' prefix)
        state_dict = checkpoint['state_dict']
        
        # Remove 'net.' prefix from keys
        clean_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('net.'):
                clean_key = key[4:]  # Remove 'net.' prefix
                clean_state_dict[clean_key] = value
        
        # Load state dict
        model.load_state_dict(clean_state_dict, strict=True)
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully with correct architecture")
        print(f"📏 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
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
    
    print(f"📊 Predictions shape: {predictions.shape} -> {pred_flat.shape}")
    print(f"📊 Targets shape: {targets.shape} -> {target_flat.shape}")
    
    # Print sample values for debugging
    print(f"📈 Sample predictions: {pred_flat[0]}")
    print(f"📈 Sample targets: {target_flat[0]}")
    print(f"📈 Prediction ranges: min={np.min(pred_flat, axis=0)}, max={np.max(pred_flat, axis=0)}")
    print(f"📈 Target ranges: min={np.min(target_flat, axis=0)}, max={np.max(target_flat, axis=0)}")
    
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
    
    # Translation vs Rotation (first 3 dims are rotation, last 3 are translation)
    if pred_flat.shape[1] == 6:
        rot_mse = np.mean((pred_flat[:, :3] - target_flat[:, :3]) ** 2)
        trans_mse = np.mean((pred_flat[:, 3:] - target_flat[:, 3:]) ** 2)
        metrics['rotation_mse'] = rot_mse
        metrics['translation_mse'] = trans_mse
        metrics['rotation_rmse'] = np.sqrt(rot_mse)
        metrics['translation_rmse'] = np.sqrt(trans_mse)
    
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
            
            # Create a dummy batch format expected by the model
            # Model expects: batch = (visual_inertial_features, _, _)
            batch = (features, None, None)
            
            # Forward pass - model expects (batch, gt) format from training
            predictions = model(batch, targets)  # Pass targets as dummy, model returns predictions
            
            # Store results
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    print(f"📐 Final prediction shape: {all_predictions.shape}")
    print(f"📐 Final target shape: {all_targets.shape}")
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets)
    
    return metrics, all_predictions, all_targets

def print_metrics(metrics):
    """Print evaluation metrics in a nice format"""
    
    print("\n" + "="*60)
    print("📊 CORRECTED EVALUATION RESULTS")
    print("="*60)
    
    print(f"🔢 Overall Metrics:")
    print(f"   MSE: {metrics['mse']:.4f}")
    print(f"   RMSE: {metrics['rmse']:.4f} meters")
    print(f"   MAE: {metrics['mae']:.4f} meters")
    
    if 'rotation_rmse' in metrics and 'translation_rmse' in metrics:
        print(f"\n🔄 Rotation vs Translation:")
        print(f"   Rotation RMSE: {metrics['rotation_rmse']:.4f} rad ({metrics['rotation_rmse']*57.2958:.1f}°)")
        print(f"   Translation RMSE: {metrics['translation_rmse']:.4f} meters")
    
    print(f"\n📈 Per-Dimension RMSE:")
    dim_names = ['rx', 'ry', 'rz', 'tx', 'ty', 'tz']
    for i in range(6):
        if f'rmse_dim_{i}' in metrics:
            if i < 3:  # Rotation dimensions
                print(f"   {dim_names[i]} (rotation): {metrics[f'rmse_dim_{i}']:.4f} rad ({metrics[f'rmse_dim_{i}']*57.2958:.1f}°)")
            else:  # Translation dimensions  
                print(f"   {dim_names[i]} (translation): {metrics[f'rmse_dim_{i}']:.4f} meters")
    
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
    np.save(f"{base_name}_predictions_corrected.npy", predictions.numpy())
    np.save(f"{base_name}_targets_corrected.npy", targets.numpy())
    
    print(f"💾 Corrected results saved to: {output_file}")
    print(f"💾 Corrected predictions saved to: {base_name}_predictions_corrected.npy")
    print(f"💾 Corrected targets saved to: {base_name}_targets_corrected.npy")

def main():
    parser = argparse.ArgumentParser(description='Corrected evaluation for VIFT-AEA')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--test_data', required=True, help='Path to test data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use for evaluation')
    parser.add_argument('--output', default='corrected_evaluation_results.json',
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
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, device)
    if model is None:
        return
    
    # Load test dataset
    test_dataset = AriaTestDataset(args.test_data)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    # Evaluate model
    metrics, predictions, targets = evaluate_model(model, test_loader, device)
    
    # Print results
    print_metrics(metrics)
    
    # Save results
    save_results(metrics, predictions, targets, args.output)

if __name__ == "__main__":
    main()
