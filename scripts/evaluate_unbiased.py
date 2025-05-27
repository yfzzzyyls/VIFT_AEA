#!/usr/bin/env python3
"""
Unbiased Test Evaluation Script for VIFT-AEA
Run this script after training to evaluate on the unbiased test dataset
(AriaEveryday sequences 20, 22, 24 that were not used during training)
"""
import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Add the project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.components.latent_kitti_dataset import LatentVectorDataset
from src.models.vio_module import VIOLitModule
from src.models.components.pose_transformer import PoseTransformer

def compute_comprehensive_metrics(predictions, ground_truth):
    """
    Compute comprehensive pose estimation metrics
    
    Args:
        predictions: [N, 7] predicted poses (tx, ty, tz, qx, qy, qz, qw)
        ground_truth: [N, 7] ground truth poses
    
    Returns:
        dict: Comprehensive metrics
    """
    pred = predictions.cpu().numpy()
    gt = ground_truth.cpu().numpy()
    
    # Translation errors (first 3 components)
    trans_pred = pred[:, :3]
    trans_gt = gt[:, :3]
    trans_errors = np.linalg.norm(trans_pred - trans_gt, axis=1)
    
    # Rotation errors (last 4 components - quaternions)
    quat_pred = pred[:, 3:]
    quat_gt = gt[:, 3:]
    
    # Normalize quaternions properly
    quat_pred_norm = quat_pred / (np.linalg.norm(quat_pred, axis=1, keepdims=True) + 1e-8)
    quat_gt_norm = quat_gt / (np.linalg.norm(quat_gt, axis=1, keepdims=True) + 1e-8)
    
    # Compute quaternion dot product (handle both q and -q representing same rotation)
    quat_dot = np.abs(np.sum(quat_pred_norm * quat_gt_norm, axis=1))
    quat_dot = np.clip(quat_dot, 0, 1)  # Ensure valid range [0,1]
    
    # Convert to rotation angle error (in degrees)
    quat_dot_safe = np.clip(quat_dot, 0, 0.9999999)
    rot_errors = 2 * np.arccos(quat_dot_safe) * 180 / np.pi
    
    # Fix any potential numerical issues
    rot_errors = np.nan_to_num(rot_errors, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Accuracy thresholds
    good_trans_1cm = np.sum(trans_errors < 0.01)
    good_trans_5cm = np.sum(trans_errors < 0.05)
    good_trans_10cm = np.sum(trans_errors < 0.1)
    good_rot_1deg = np.sum(rot_errors < 1.0)
    good_rot_5deg = np.sum(rot_errors < 5.0)
    good_rot_10deg = np.sum(rot_errors < 10.0)
    
    metrics = {
        # Translation metrics (meters)
        'translation_rmse': float(np.sqrt(np.mean(trans_errors**2))),
        'translation_mae': float(np.mean(trans_errors)),
        'translation_median': float(np.median(trans_errors)),
        'translation_max': float(np.max(trans_errors)),
        'translation_min': float(np.min(trans_errors)),
        'translation_std': float(np.std(trans_errors)),
        
        # Rotation metrics (degrees)
        'rotation_rmse': float(np.sqrt(np.mean(rot_errors**2))),
        'rotation_mae': float(np.mean(rot_errors)),
        'rotation_median': float(np.median(rot_errors)),
        'rotation_max': float(np.max(rot_errors)),
        'rotation_min': float(np.min(rot_errors)),
        'rotation_std': float(np.std(rot_errors)),
        
        # Overall metrics
        'overall_mse': float(np.mean((pred - gt)**2)),
        'overall_rmse': float(np.sqrt(np.mean((pred - gt)**2))),
        'overall_mae': float(np.mean(np.abs(pred - gt))),
        
        # Per-axis translation RMSE
        'tx_rmse': float(np.sqrt(np.mean((trans_pred[:, 0] - trans_gt[:, 0])**2))),
        'ty_rmse': float(np.sqrt(np.mean((trans_pred[:, 1] - trans_gt[:, 1])**2))),
        'tz_rmse': float(np.sqrt(np.mean((trans_pred[:, 2] - trans_gt[:, 2])**2))),
        
        # Accuracy thresholds
        'trans_1cm_count': int(good_trans_1cm),
        'trans_5cm_count': int(good_trans_5cm),
        'trans_10cm_count': int(good_trans_10cm),
        'rot_1deg_count': int(good_rot_1deg),
        'rot_5deg_count': int(good_rot_5deg),
        'rot_10deg_count': int(good_rot_10deg),
        'trans_1cm_percent': float(100 * good_trans_1cm / len(trans_errors)),
        'trans_5cm_percent': float(100 * good_trans_5cm / len(trans_errors)),
        'trans_10cm_percent': float(100 * good_trans_10cm / len(trans_errors)),
        'rot_1deg_percent': float(100 * good_rot_1deg / len(rot_errors)),
        'rot_5deg_percent': float(100 * good_rot_5deg / len(rot_errors)),
        'rot_10deg_percent': float(100 * good_rot_10deg / len(rot_errors)),
        
        # Sample count
        'total_samples': int(len(predictions))
    }
    
    return metrics

def load_trained_model(checkpoint_path, device='cpu'):
    """Load the trained VIFT-AEA model from checkpoint"""
    print(f"📂 Loading model from: {checkpoint_path}")
    
    # Create model architecture (same as training)
    net = PoseTransformer(
        input_dim=768,      # VIFT visual-inertial features
        embedding_dim=128,
        num_layers=2,
        nhead=8,
        dim_feedforward=512,
        dropout=0.1
    )
    
    # Create lightning module
    model = VIOLitModule(
        net=net,
        optimizer=None,
        scheduler=None,
        criterion=torch.nn.MSELoss(),
        compile=False,
        tester=None,
        metrics_calculator=None
    )
    
    # Load checkpoint (handle PyTorch 2.6+ weights_only issue)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    return model

def run_unbiased_evaluation(model, test_loader, device='cpu'):
    """Run evaluation on the unbiased test dataset"""
    print(f"🔍 Running unbiased evaluation on {len(test_loader)} batches...")
    
    all_predictions = []
    all_ground_truth = []
    all_losses = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Unpack batch (VIFT format)
            (visual_inertial_features, rot, w), target = batch
            
            # Prepare VIFT-style batch
            vift_batch = (visual_inertial_features.to(device), None, None)
            target = target.to(device)
            
            # Forward pass
            predictions = model.forward(vift_batch, target)
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(predictions, target)
            all_losses.append(loss.item())
            
            # Store predictions and targets
            all_predictions.append(predictions.cpu())
            all_ground_truth.append(target.cpu())
    
    # Concatenate all results
    predictions = torch.cat(all_predictions, dim=0)
    ground_truth = torch.cat(all_ground_truth, dim=0)
    avg_loss = np.mean(all_losses)
    
    print(f"✅ Evaluation complete: {len(predictions)} samples")
    print(f"📊 Average test loss (MSE): {avg_loss:.6f}")
    
    return predictions, ground_truth, avg_loss

def print_evaluation_report(metrics, test_loss, checkpoint_path):
    """Print comprehensive evaluation report"""
    print("\n" + "="*75)
    print("🎯 VIFT-AEA UNBIASED TEST EVALUATION REPORT")
    print("   Test sequences: 20, 22, 24 from AriaEveryday (unseen during training)")
    print("="*75)
    
    print(f"\n📂 MODEL CHECKPOINT:")
    print(f"   Path: {checkpoint_path}")
    print(f"   Test Loss (MSE): {test_loss:.6f}")
    print(f"   Total Samples: {metrics['total_samples']}")
    
    print(f"\n📍 TRANSLATION ERROR ANALYSIS (meters):")
    print(f"   RMSE:     {metrics['translation_rmse']:.4f} m")
    print(f"   MAE:      {metrics['translation_mae']:.4f} m")
    print(f"   Median:   {metrics['translation_median']:.4f} m")
    print(f"   Max:      {metrics['translation_max']:.4f} m")
    print(f"   Min:      {metrics['translation_min']:.4f} m")
    print(f"   Std Dev:  {metrics['translation_std']:.4f} m")
    
    print(f"\n🔄 ROTATION ERROR ANALYSIS (degrees):")
    print(f"   RMSE:     {metrics['rotation_rmse']:.4f}°")
    print(f"   MAE:      {metrics['rotation_mae']:.4f}°")
    print(f"   Median:   {metrics['rotation_median']:.4f}°")
    print(f"   Max:      {metrics['rotation_max']:.4f}°")
    print(f"   Min:      {metrics['rotation_min']:.4f}°")
    print(f"   Std Dev:  {metrics['rotation_std']:.4f}°")
    
    print(f"\n🎛️ PER-AXIS TRANSLATION ERRORS (RMSE):")
    print(f"   X-axis:   {metrics['tx_rmse']:.4f} m")
    print(f"   Y-axis:   {metrics['ty_rmse']:.4f} m")
    print(f"   Z-axis:   {metrics['tz_rmse']:.4f} m")
    
    print(f"\n📈 ACCURACY THRESHOLDS:")
    print(f"   Translation < 1cm:   {metrics['trans_1cm_count']:3d}/{metrics['total_samples']} ({metrics['trans_1cm_percent']:5.1f}%)")
    print(f"   Translation < 5cm:   {metrics['trans_5cm_count']:3d}/{metrics['total_samples']} ({metrics['trans_5cm_percent']:5.1f}%)")
    print(f"   Translation < 10cm:  {metrics['trans_10cm_count']:3d}/{metrics['total_samples']} ({metrics['trans_10cm_percent']:5.1f}%)")
    print(f"   Rotation < 1°:       {metrics['rot_1deg_count']:3d}/{metrics['total_samples']} ({metrics['rot_1deg_percent']:5.1f}%)")
    print(f"   Rotation < 5°:       {metrics['rot_5deg_count']:3d}/{metrics['total_samples']} ({metrics['rot_5deg_percent']:5.1f}%)")
    print(f"   Rotation < 10°:      {metrics['rot_10deg_count']:3d}/{metrics['total_samples']} ({metrics['rot_10deg_percent']:5.1f}%)")
    
    print(f"\n🔍 PERFORMANCE ASSESSMENT:")
    
    # Translation assessment
    if metrics['translation_mae'] < 0.1:
        trans_status = "✅ EXCELLENT - Mean error < 10cm"
    elif metrics['translation_mae'] < 0.5:
        trans_status = "⚠️  MODERATE - Mean error 10-50cm"
    else:
        trans_status = "❌ POOR - Mean error > 50cm"
    print(f"   Translation: {trans_status}")
    
    # Rotation assessment
    if metrics['rotation_mae'] < 5.0:
        rot_status = "✅ EXCELLENT - Mean error < 5°"
    elif metrics['rotation_mae'] < 15.0:
        rot_status = "⚠️  MODERATE - Mean error 5-15°"
    else:
        rot_status = "❌ POOR - Mean error > 15°"
    print(f"   Rotation:    {rot_status}")
    
    print("\n" + "="*75)
    print("🎉 UNBIASED EVALUATION COMPLETE!")
    print("="*75)

def save_results(metrics, test_loss, checkpoint_path, output_file):
    """Save evaluation results to JSON file"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'checkpoint_path': str(checkpoint_path),
        'test_loss': test_loss,
        'metrics': metrics,
        'test_sequences': [20, 22, 24],
        'test_description': 'Unbiased evaluation on AriaEveryday sequences not used during training'
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run unbiased evaluation on VIFT-AEA model")
    parser.add_argument("--checkpoint", type=str, 
                       default="logs/train/runs/2025-05-27_11-26-57/checkpoints/epoch_000.ckpt",
                       help="Path to trained model checkpoint")
    parser.add_argument("--test_data", type=str,
                       default="aria_latent_data/test_3",
                       help="Path to unbiased test data (sequences 20,22,24)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, mps, cuda)")
    parser.add_argument("--output", type=str, default="unbiased_evaluation_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print("🎯 VIFT-AEA Unbiased Test Evaluation")
    print("="*50)
    print(f"🖥️  Device: {device}")
    
    # Verify checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Verify test data exists
    test_data_path = Path(args.test_data)
    if not test_data_path.exists():
        print(f"❌ Test data not found: {test_data_path}")
        print("   Run this first: python data/latent_caching_aria.py --data_dir data/aria_real_test --save_dir aria_latent_data/test_3 --mode test")
        return
    
    # Load test dataset
    print(f"📁 Loading unbiased test data from: {test_data_path}")
    test_dataset = LatentVectorDataset(str(test_data_path))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    print(f"📊 Test dataset: {len(test_dataset)} samples from sequences 20, 22, 24")
    
    # Load trained model
    model = load_trained_model(checkpoint_path, device)
    
    # Run unbiased evaluation
    predictions, ground_truth, test_loss = run_unbiased_evaluation(model, test_loader, device)
    
    # Compute comprehensive metrics
    print("\n🧮 Computing comprehensive metrics...")
    metrics = compute_comprehensive_metrics(predictions, ground_truth)
    
    # Print evaluation report
    print_evaluation_report(metrics, test_loss, checkpoint_path)
    
    # Save results
    save_results(metrics, test_loss, checkpoint_path, args.output)
    
    print(f"\n✨ Evaluation complete! Check {args.output} for detailed results.")

if __name__ == "__main__":
    main()
