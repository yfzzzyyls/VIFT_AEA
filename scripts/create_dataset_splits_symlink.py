#!/usr/bin/env python3
"""
Create custom dataset splits for Aria data using symlinks (faster than copying)
"""
import argparse
from pathlib import Path
import json
import random
import os

def create_splits(data_dir: Path, output_dir: Path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Create train/val/test splits from available sequences using symlinks"""
    
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
    
    # Create symlinks to sequences in appropriate directories
    print("\nCreating symlinks to sequences in split directories...")
    for seq in train_seqs:
        src = data_dir / seq
        dst = output_dir / "train" / seq
        if src.exists():
            if dst.exists() or dst.is_symlink():
                if dst.is_dir() and not dst.is_symlink():
                    import shutil
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            dst.symlink_to(src.absolute())
            print(f"  Linked {seq} to train/")
    
    for seq in val_seqs:
        src = data_dir / seq
        dst = output_dir / "val" / seq
        if src.exists():
            if dst.exists() or dst.is_symlink():
                if dst.is_dir() and not dst.is_symlink():
                    import shutil
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            dst.symlink_to(src.absolute())
            print(f"  Linked {seq} to val/")
    
    for seq in test_seqs:
        src = data_dir / seq
        dst = output_dir / "test" / seq
        if src.exists():
            if dst.exists() or dst.is_symlink():
                if dst.is_dir() and not dst.is_symlink():
                    import shutil
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            dst.symlink_to(src.absolute())
            print(f"  Linked {seq} to test/")
    
    print(f"\nCreated splits:")
    print(f"  Train: {n_train} sequences ({n_train/n_total*100:.1f}%)")
    print(f"  Val: {n_val} sequences ({n_val/n_total*100:.1f}%)")
    print(f"  Test: {n_test} sequences ({n_test/n_total*100:.1f}%)")
    
    return splits

def main():
    parser = argparse.ArgumentParser(description="Create dataset splits for Aria data using symlinks")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing processed sequences")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for split datasets")
    parser.add_argument("--json_output", type=str, default=None, help="Optional: Save splits info to JSON file")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        return
    
    # Create splits and create symlinks
    splits = create_splits(
        data_dir, output_dir, 
        args.train_ratio, args.val_ratio, args.test_ratio, 
        args.seed
    )
    
    # Save splits info to JSON if requested
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"\nSaved splits info to {args.json_output}")
    
    # Create list files for each split
    for split_name, seq_list in [("train", splits["train"]), 
                                  ("val", splits["val"]), 
                                  ("test", splits["test"])]:
        list_file = output_dir / f"{split_name}_list.txt"
        with open(list_file, 'w') as f:
            for seq in seq_list:
                f.write(f"{seq}\n")
        print(f"Created {list_file}")

if __name__ == "__main__":
    main()