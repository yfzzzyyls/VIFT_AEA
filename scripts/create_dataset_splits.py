#!/usr/bin/env python3
"""
Create custom dataset splits for Aria data
"""
import argparse
from pathlib import Path
import json
import random
import shutil

def create_splits(data_dir: Path, output_dir: Path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Create train/val/test splits from available sequences"""
    
    # Find all sequence directories
    sequences = []
    for seq_dir in sorted(data_dir.glob("[0-9]*")):
        if seq_dir.is_dir() and seq_dir.name.isdigit():
            sequences.append(seq_dir.name)
    
    print(f"Found {len(sequences)} sequences")
    
    # Shuffle sequences
    random.seed(seed)
    random.shuffle(sequences)
    
    # Calculate split sizes
    n_total = len(sequences)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    # Create splits
    train_seqs = sequences[:n_train]
    val_seqs = sequences[n_train:n_train + n_val]
    test_seqs = sequences[n_train + n_val:]
    
    splits = {
        "train": sorted(train_seqs),
        "val": sorted(val_seqs),
        "test": sorted(test_seqs),
        "total_sequences": n_total,
        "split_sizes": {
            "train": n_train,
            "val": n_val,
            "test": n_test
        }
    }
    
    # Create output directories
    for split_name in ["train", "val", "test"]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy sequences to appropriate directories
    print("\nCopying sequences to split directories...")
    for seq in train_seqs:
        src = data_dir / seq
        dst = output_dir / "train" / seq
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"  Copied {seq} to train/")
    
    for seq in val_seqs:
        src = data_dir / seq
        dst = output_dir / "val" / seq
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"  Copied {seq} to val/")
    
    for seq in test_seqs:
        src = data_dir / seq
        dst = output_dir / "test" / seq
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"  Copied {seq} to test/")
    
    return splits

def main():
    parser = argparse.ArgumentParser(description="Create dataset splits for Aria data")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing processed sequences")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for split datasets")
    parser.add_argument("--json_output", type=str, default=None, help="Optional: Save splits info to JSON file")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        return
    
    # Create splits and copy data
    splits = create_splits(
        data_dir,
        output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Optionally save splits info to JSON
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"\nSplits info saved to {args.json_output}")
    
    print(f"\nDataset splits created in {output_dir}")
    print(f"Train: {splits['split_sizes']['train']} sequences in {output_dir}/train/")
    print(f"Val: {splits['split_sizes']['val']} sequences in {output_dir}/val/")
    print(f"Test: {splits['split_sizes']['test']} sequences in {output_dir}/test/")
    
    # Print example commands
    print("\nExample commands to cache these splits:")
    print(f"\n# Training set")
    print(f"python data/latent_caching_aria.py \\")
    print(f"  --data_dir {output_dir}/train \\")
    print(f"  --save_dir aria_latent_data/train \\")
    print(f"  --mode train \\")
    print(f"  --device mps")
    
    print(f"\n# Validation set")
    print(f"python data/latent_caching_aria.py \\")
    print(f"  --data_dir {output_dir}/val \\")
    print(f"  --save_dir aria_latent_data/val \\")
    print(f"  --mode val \\")
    print(f"  --device mps")
    
    print(f"\n# Test set")
    print(f"python data/latent_caching_aria.py \\")
    print(f"  --data_dir {output_dir}/test \\")
    print(f"  --save_dir aria_latent_data/test \\")
    print(f"  --mode test \\")
    print(f"  --device mps")

if __name__ == "__main__":
    main()