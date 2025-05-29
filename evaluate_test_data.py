#!/usr/bin/env python3
"""
Robust evaluation script for VIFT-AEA trained model on test data
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# Import the model and dataset
from src.models.vio_module import VIOLitModule
from src.data.components.latent_kitti_dataset import LatentVectorDataset

def evaluate_model(checkpoint_path, test_data_path, batch_size=32, device='cuda'):
    """
    Evaluate the trained VIFT model on test data
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        test_data_path: Path to the cached test data directory
        batch_size: Batch size for evaluation
        device: Device to run evaluation on ('cuda', 'mps', or 'cpu')
    """
    
    # Set up device
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"🖥️ Using device: {device}")
    
    # Load the trained model
    print(f"📂 Loading trained model from: {checkpoint_path}")
    try:
        model = VIOLitModule.load_from_checkpoint(checkpoint_path)
        model = model.to(device)
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Load test dataset
    print(f"📁 Loading test dataset from: {test_data_path}")
    try:
        test_dataset = LatentVectorDataset(str(test_data_path))
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=False
        )
        print(f"✅ Test dataset loaded: {len(test_dataset)} samples, {len(test_loader)} batches")
    except Exception as e:
        print(f"❌ Error loading test dataset: {e}")
        return None
    
    # Run evaluation
    print(f"🚀 Starting evaluation...")
    
    total_loss = 0.0
    total_samples = 0
    batch_losses = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            try:
                # Get batch data
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                
                # Forward pass
                predictions = model(x)
                
                # Calculate loss
                loss = model.criterion(predictions, y)
                
                # Accumulate statistics
                batch_loss = loss.item()
                batch_size_actual = x.size(0)
                
                total_loss += batch_loss * batch_size_actual
                total_samples += batch_size_actual
                batch_losses.append(batch_loss)
                
                # Print progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    avg_loss_so_far = total_loss / total_samples
                    print(f"  Batch {batch_idx + 1}/{len(test_loader)}: Loss = {batch_loss:.4f}, Avg = {avg_loss_so_far:.4f}")
                    
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                continue
    
    # Calculate final statistics
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        min_loss = min(batch_losses)
        max_loss = max(batch_losses)
        std_loss = np.std(batch_losses)
        
        print(f"\n🎉 Evaluation Complete!")
        print(f"=" * 50)
        print(f"📊 Test Results:")
        print(f"  Total samples processed: {total_samples}")
        print(f"  Total batches: {len(batch_losses)}")
        print(f"  Average test loss: {avg_loss:.4f}")
        print(f"  Min batch loss: {min_loss:.4f}")
        print(f"  Max batch loss: {max_loss:.4f}")
        print(f"  Std deviation: {std_loss:.4f}")
        print(f"=" * 50)
        
        return {
            'avg_loss': avg_loss,
            'min_loss': min_loss,
            'max_loss': max_loss,
            'std_loss': std_loss,
            'total_samples': total_samples,
            'total_batches': len(batch_losses)
        }
    else:
        print(f"❌ No samples were successfully processed")
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate VIFT-AEA model on test data")
    parser.add_argument("--checkpoint", type=str, 
                       default="logs/aria_vio/runs/2025-05-29_13-44-07/checkpoints/epoch_000.ckpt",
                       help="Path to model checkpoint")
    parser.add_argument("--test_data", type=str,
                       default="aria_latent_data/test",
                       help="Path to test data directory")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "mps", "cpu"],
                       help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    # Check if paths exist
    checkpoint_path = Path(args.checkpoint)
    test_data_path = Path(args.test_data)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
        
    if not test_data_path.exists():
        print(f"❌ Test data directory not found: {test_data_path}")
        return
    
    # Run evaluation
    results = evaluate_model(
        checkpoint_path=checkpoint_path,
        test_data_path=test_data_path,
        batch_size=args.batch_size,
        device=args.device
    )
    
    if results:
        print(f"\n✅ Evaluation completed successfully!")
    else:
        print(f"\n❌ Evaluation failed!")

if __name__ == "__main__":
    main()
