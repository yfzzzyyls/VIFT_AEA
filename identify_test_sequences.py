#!/usr/bin/env python3
"""
Identify and save test sequence indices for consistent evaluation
"""

import os
import json
from pathlib import Path


def identify_test_sequences(processed_dir, output_dir, split_ratios=(0.7, 0.1, 0.2)):
    """
    Identify which sequences belong to train/val/test splits and save the information
    """
    # Get all sequence directories
    seq_dirs = sorted([d for d in Path(processed_dir).iterdir() 
                      if d.is_dir() and d.name.isdigit()])
    
    num_sequences = len(seq_dirs)
    print(f"Found {num_sequences} sequences")
    
    # Calculate split sizes
    train_size = int(num_sequences * split_ratios[0])
    val_size = int(num_sequences * split_ratios[1])
    test_size = num_sequences - train_size - val_size
    
    # Split sequences
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, num_sequences))
    
    # Get sequence names
    train_seqs = [seq_dirs[i].name for i in train_indices]
    val_seqs = [seq_dirs[i].name for i in val_indices]
    test_seqs = [seq_dirs[i].name for i in test_indices]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_seqs)} sequences ({train_indices[0]}-{train_indices[-1]})")
    print(f"  Val: {len(val_seqs)} sequences ({val_indices[0]}-{val_indices[-1]})")
    print(f"  Test: {len(test_seqs)} sequences ({test_indices[0]}-{test_indices[-1]})")
    
    # Save split information
    split_info = {
        'total_sequences': num_sequences,
        'split_ratios': split_ratios,
        'splits': {
            'train': {
                'indices': train_indices,
                'sequences': train_seqs,
                'count': len(train_seqs)
            },
            'val': {
                'indices': val_indices,
                'sequences': val_seqs,
                'count': len(val_seqs)
            },
            'test': {
                'indices': test_indices,
                'sequences': test_seqs,
                'count': len(test_seqs)
            }
        }
    }
    
    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    split_file = os.path.join(output_dir, 'dataset_splits.json')
    
    with open(split_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\nSplit information saved to: {split_file}")
    
    # Also save just the test sequence list for easy access
    test_file = os.path.join(output_dir, 'test_sequences.txt')
    with open(test_file, 'w') as f:
        f.write('\n'.join(test_seqs))
    
    print(f"Test sequence list saved to: {test_file}")
    
    # Print some test sequences as examples
    print(f"\nExample test sequences:")
    for i in range(min(5, len(test_seqs))):
        print(f"  {test_seqs[i]}")
    if len(test_seqs) > 5:
        print(f"  ... and {len(test_seqs) - 5} more")
    
    return split_info


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Identify dataset splits')
    parser.add_argument('--processed-dir', type=str, default='data/aria_processed',
                        help='Directory with processed sequences')
    parser.add_argument('--output-dir', type=str, default='aria_latent_data_pretrained',
                        help='Output directory for split information')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation set ratio')
    
    args = parser.parse_args()
    
    # Calculate test ratio
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    split_ratios = (args.train_ratio, args.val_ratio, test_ratio)
    
    print(f"Dataset Split Identification")
    print(f"{'='*40}")
    print(f"Split ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={test_ratio:.1f}")
    
    identify_test_sequences(args.processed_dir, args.output_dir, split_ratios)