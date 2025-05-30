#!/usr/bin/env python3
"""
Auto-detecting evaluation script for any model architecture
"""
import argparse
import json
import numpy as np
import torch
from torch import nn
from pathlib import Path
from tqdm import tqdm
import os
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.components.pose_transformer import PoseTransformer
from models.components.latent_simple_dense_net import LatentSimpleDenseNet
from data.components.latent_kitti_dataset import LatentVectorDataset

class AutoEvaluator:
    def __init__(self, checkpoint_path, test_data_dir, batch_size=16, device="cuda"):
        self.checkpoint_path = checkpoint_path
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"🖥️ Using device: {self.device}")
        
    def load_checkpoint_info(self):
        """Load checkpoint and extract model architecture info"""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Extract hyperparameters
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
            print(f"📋 Hyperparameters: {hparams}")
        
        # Analyze state dict to determine architecture
        state_dict = checkpoint['state_dict']
        
        # Remove model prefix if present
        clean_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('net.'):
                clean_key = key[4:]  # Remove 'net.' prefix
                clean_state_dict[clean_key] = value
        
        return clean_state_dict, checkpoint
    
    def detect_architecture(self, state_dict):
        """Auto-detect model architecture from state dict"""
        
        # Check for transformer layers
        has_transformer = any('transformer_encoder' in key for key in state_dict.keys())
        
        if has_transformer:
            # Count transformer layers
            layer_indices = set()
            for key in state_dict.keys():
                if 'transformer_encoder.layers.' in key:
                    parts = key.split('.')
                    if len(parts) >= 3 and parts[2].isdigit():
                        layer_indices.add(int(parts[2]))
            
            num_layers = len(layer_indices)
            
            # Detect embedding dimension from fc1
            if 'fc1.0.weight' in state_dict:
                input_dim, embedding_dim = state_dict['fc1.0.weight'].shape
                embedding_dim = state_dict['fc1.0.weight'].shape[0]
            else:
                embedding_dim = 128  # Default
            
            # Detect number of heads from attention weights
            nhead = 8  # Default
            if 'transformer_encoder.layers.0.self_attn.in_proj_weight' in state_dict:
                attn_weight_shape = state_dict['transformer_encoder.layers.0.self_attn.in_proj_weight'].shape
                # in_proj_weight has shape [3*embed_dim, embed_dim] for multihead attention
                nhead = attn_weight_shape[0] // (3 * embedding_dim)
            
            print(f"🔍 Detected PoseTransformer: {num_layers} layers, {embedding_dim} embedding, {nhead} heads")
            
            return {
                'type': 'PoseTransformer',
                'input_dim': 768,
                'embedding_dim': embedding_dim,
                'num_layers': num_layers,
                'nhead': nhead,
                'dim_feedforward': 512,
                'dropout': 0.1
            }
        else:
            # Assume dense network - try to detect sizes from state dict
            input_size = 768  # Default
            if 'model.0.weight' in state_dict:
                input_size = state_dict['model.0.weight'].shape[1]
                lin1_size = state_dict['model.0.weight'].shape[0]
            else:
                lin1_size = 128
            
            print(f"🔍 Detected Dense Network: input={input_size}, hidden={lin1_size}")
            return {
                'type': 'LatentSimpleDenseNet',
                'input_size': input_size,
                'lin1_size': lin1_size,
                'lin2_size': lin1_size,
                'lin3_size': lin1_size,
                'output_size': 6
            }
    
    def create_model(self, arch_info):
        """Create model based on detected architecture"""
        if arch_info['type'] == 'PoseTransformer':
            model = PoseTransformer(
                input_dim=arch_info['input_dim'],
                embedding_dim=arch_info['embedding_dim'],
                num_layers=arch_info['num_layers'],
                nhead=arch_info['nhead'],
                dim_feedforward=arch_info.get('dim_feedforward', 512),
                dropout=arch_info.get('dropout', 0.1)
            )
        elif arch_info['type'] == 'LatentSimpleDenseNet':
            model = LatentSimpleDenseNet(
                input_size=arch_info.get('input_size', 768),
                lin1_size=arch_info.get('lin1_size', 128),
                lin2_size=arch_info.get('lin2_size', 128),
                lin3_size=arch_info.get('lin3_size', 128),
                output_size=arch_info.get('output_size', 6)
            )
        else:
            raise ValueError(f"Unknown architecture type: {arch_info['type']}")
        
        return model.to(self.device)
    
    def load_model(self):
        """Load and initialize model"""
        clean_state_dict, checkpoint = self.load_checkpoint_info()
        arch_info = self.detect_architecture(clean_state_dict)
        model = self.create_model(arch_info)
        
        try:
            model.load_state_dict(clean_state_dict, strict=True)
            print("✅ Model loaded successfully with auto-detected architecture")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
        
        model.eval()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"📏 Model parameters: {param_count:,}")
        
        return model
    
    def evaluate(self):
        """Run evaluation"""
        model = self.load_model()
        if model is None:
            return
        
        # Load test dataset
        test_dataset = LatentVectorDataset(self.test_data_dir)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        print(f"📊 Loaded {len(test_dataset)} test samples")
        
        # Run inference
        all_predictions = []
        all_targets = []
        
        print("🔍 Running inference on test data...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch_inputs, targets = batch
                # Move batch inputs to device
                batch_inputs = [x.to(self.device) for x in batch_inputs]
                targets = targets.to(self.device)
                
                # Model expects (batch, gt) format
                predictions = model(batch_inputs, targets)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate all results
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        print(f"📐 Final prediction shape: {predictions.shape}")
        print(f"📐 Final target shape: {targets.shape}")
        
        # Flatten for metrics calculation
        predictions_flat = predictions.view(-1, 6).numpy()
        targets_flat = targets.view(-1, 6).numpy()
        
        print(f"📊 Predictions shape: {predictions.shape} -> {predictions_flat.shape}")
        print(f"📊 Targets shape: {targets.shape} -> {targets_flat.shape}")
        
        # Sample statistics
        print(f"📈 Sample predictions: {predictions_flat[0]}")
        print(f"📈 Sample targets: {targets_flat[0]}")
        print(f"📈 Prediction ranges: min={predictions_flat.min(axis=0)}, max={predictions_flat.max(axis=0)}")
        print(f"📈 Target ranges: min={targets_flat.min(axis=0)}, max={targets_flat.max(axis=0)}")
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions_flat, targets_flat)
        self.print_results(metrics)
        self.save_results(metrics, predictions_flat, targets_flat)
        
        return metrics
    
    def calculate_metrics(self, predictions, targets):
        """Calculate comprehensive metrics"""
        # Overall metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # Per-dimension metrics
        dim_metrics = {}
        for i in range(6):
            mse_dim = np.mean((predictions[:, i] - targets[:, i]) ** 2)
            rmse_dim = np.sqrt(mse_dim)
            mae_dim = np.mean(np.abs(predictions[:, i] - targets[:, i]))
            
            dim_metrics[f'mse_dim_{i}'] = mse_dim
            dim_metrics[f'rmse_dim_{i}'] = rmse_dim
            dim_metrics[f'mae_dim_{i}'] = mae_dim
        
        # Rotation vs Translation metrics
        rotation_mse = np.mean((predictions[:, :3] - targets[:, :3]) ** 2)
        translation_mse = np.mean((predictions[:, 3:] - targets[:, 3:]) ** 2)
        rotation_rmse = np.sqrt(rotation_mse)
        translation_rmse = np.sqrt(translation_mse)
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'rotation_mse': float(rotation_mse),
            'translation_mse': float(translation_mse),
            'rotation_rmse': float(rotation_rmse),
            'translation_rmse': float(translation_rmse),
            **{k: float(v) for k, v in dim_metrics.items()}
        }
        
        return metrics
    
    def print_results(self, metrics):
        """Print formatted results"""
        print("\n" + "="*60)
        print("📊 AUTO EVALUATION RESULTS")
        print("="*60)
        print(f"🔢 Overall Metrics:")
        print(f"   MSE: {metrics['mse']:.4f}")
        print(f"   RMSE: {metrics['rmse']:.4f} meters")
        print(f"   MAE: {metrics['mae']:.4f} meters")
        
        print(f"\n🔄 Rotation vs Translation:")
        print(f"   Rotation RMSE: {metrics['rotation_rmse']:.4f} rad ({np.degrees(metrics['rotation_rmse']):.1f}°)")
        print(f"   Translation RMSE: {metrics['translation_rmse']:.4f} meters")
        
        print(f"\n📈 Per-Dimension RMSE:")
        dims = ['rx (rotation)', 'ry (rotation)', 'rz (rotation)', 'tx (translation)', 'ty (translation)', 'tz (translation)']
        for i, dim_name in enumerate(dims):
            rmse_val = metrics[f'rmse_dim_{i}']
            if i < 3:  # Rotation
                print(f"   {dim_name}: {rmse_val:.4f} rad ({np.degrees(rmse_val):.1f}°)")
            else:  # Translation
                print(f"   {dim_name}: {rmse_val:.4f} meters")
        print("="*60)
    
    def save_results(self, metrics, predictions, targets):
        """Save results to files"""
        results = {
            'metrics': metrics,
            'summary': {
                'num_samples': len(predictions) // 11,  # Assuming 11-frame sequences
                'sequence_length': 11,
                'pose_dimensions': 6,
                'total_predictions': len(predictions)
            }
        }
        
        # Save metrics
        with open('auto_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save arrays
        np.save('auto_evaluation_predictions.npy', predictions)
        np.save('auto_evaluation_targets.npy', targets)
        
        print(f"💾 Results saved to: auto_evaluation_results.json")
        print(f"💾 Predictions saved to: auto_evaluation_predictions.npy")
        print(f"💾 Targets saved to: auto_evaluation_targets.npy")

def main():
    parser = argparse.ArgumentParser(description="Auto-detecting evaluation script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    evaluator = AutoEvaluator(
        checkpoint_path=args.checkpoint,
        test_data_dir=args.test_data,
        batch_size=args.batch_size,
        device=args.device
    )
    
    evaluator.evaluate()

if __name__ == "__main__":
    main()