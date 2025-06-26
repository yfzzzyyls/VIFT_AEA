#!/usr/bin/env python3
"""
Organize processed Aria data into train/val/test splits.
Default split: 60% train, 20% val, 20% test
"""

import os
import shutil
from pathlib import Path
import json
import argparse


def organize_splits(data_dir, train_ratio=0.6, val_ratio=0.2):
    """Organize sequences into train/val/test splits."""
    data_path = Path(data_dir)
    
    # Get all sequence directories
    seq_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.isdigit()])
    num_sequences = len(seq_dirs)
    
    if num_sequences == 0:
        print("❌ No sequence directories found!")
        return
    
    print(f"📁 Found {num_sequences} sequences")
    
    # Calculate splits
    train_size = int(num_sequences * train_ratio)
    val_size = int(num_sequences * val_ratio)
    test_size = num_sequences - train_size - val_size
    
    print(f"📊 Split sizes: train={train_size}, val={val_size}, test={test_size}")
    
    # Create split directories
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Move sequences to appropriate splits
    print("\n🚀 Moving sequences to splits...")
    
    # Train sequences
    for i in range(train_size):
        src = seq_dirs[i]
        dst = train_dir / src.name
        if not dst.exists():
            shutil.move(str(src), str(dst))
            print(f"  Train: {src.name}")
    
    # Validation sequences
    for i in range(train_size, train_size + val_size):
        src = seq_dirs[i]
        dst = val_dir / src.name
        if not dst.exists():
            shutil.move(str(src), str(dst))
            print(f"  Val: {src.name}")
    
    # Test sequences
    for i in range(train_size + val_size, num_sequences):
        src = seq_dirs[i]
        dst = test_dir / src.name
        if not dst.exists():
            shutil.move(str(src), str(dst))
            print(f"  Test: {src.name}")
    
    # Create split info file
    split_info = {
        "total_sequences": num_sequences,
        "train_sequences": [seq_dirs[i].name for i in range(train_size)],
        "val_sequences": [seq_dirs[i].name for i in range(train_size, train_size + val_size)],
        "test_sequences": [seq_dirs[i].name for i in range(train_size + val_size, num_sequences)],
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size
    }
    
    with open(data_path / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✅ Data organized into splits!")
    print(f"📂 Train: {train_size} sequences in {train_dir}")
    print(f"📂 Val: {val_size} sequences in {val_dir}")
    print(f"📂 Test: {test_size} sequences in {test_dir}")


def main():
    parser = argparse.ArgumentParser(description='Organize Aria data into train/val/test splits')
    parser.add_argument('--data-dir', type=str, default='aria_processed',
                        help='Directory with processed Aria data')
    parser.add_argument('--train-ratio', type=float, default=0.6,
                        help='Ratio of data for training (default: 0.6)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Ratio of data for validation (default: 0.2)')
    
    args = parser.parse_args()
    
    # Validate ratios
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0 or test_ratio > 1:
        print("❌ Invalid ratios! train_ratio + val_ratio must be < 1.0")
        return
    
    print(f"📊 Data split ratios: train={args.train_ratio:.1%}, val={args.val_ratio:.1%}, test={test_ratio:.1%}")
    
    organize_splits(args.data_dir, args.train_ratio, args.val_ratio)


if __name__ == "__main__":
    main()