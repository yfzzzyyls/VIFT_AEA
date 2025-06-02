#!/usr/bin/env python3
"""
Identify which sequences from aria_processed are in the test split.
"""

import os
import json
from pathlib import Path
import pickle


def identify_test_sequences():
    """Map test samples back to original sequences."""
    
    # Load metadata to understand the split
    metadata_path = "aria_latent_data_pretrained/metadata.pkl"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print("Metadata loaded:")
        print(f"  Sample counts: {metadata['sample_counts']}")
    
    # Check processed sequences
    processed_dir = Path("data/aria_processed")
    if not processed_dir.exists():
        print(f"Error: {processed_dir} not found!")
        return
    
    # Get all processed sequences
    all_sequences = sorted([d for d in processed_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    print(f"\nTotal processed sequences: {len(all_sequences)}")
    
    # Load sequence names from metadata files
    sequence_info = {}
    for seq_dir in all_sequences:
        metadata_file = seq_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                meta = json.load(f)
                sequence_info[seq_dir.name] = meta['sequence_name']
    
    # Based on the split: 70/10/20 of 143 sequences
    num_sequences = len(all_sequences)
    if num_sequences != 143:
        print(f"WARNING: Expected 143 sequences, found {num_sequences}")
    
    # Fixed split sizes based on 143 total sequences
    train_size = 100  # 70% of 143 ≈ 100
    val_size = 14     # 10% of 143 ≈ 14
    test_size = 29    # 20% of 143 ≈ 29 (remaining)
    
    print(f"\nSplit sizes (70/10/20 of 143 sequences):")
    print(f"  Train: {train_size} sequences (000-099)")
    print(f"  Val: {val_size} sequences (100-113)")
    print(f"  Test: {test_size} sequences (114-142)")
    
    # Identify test sequences
    test_start_idx = train_size + val_size
    test_sequences = all_sequences[test_start_idx:]
    
    print(f"\nTest sequences:")
    for seq_dir in test_sequences:
        seq_id = seq_dir.name
        seq_name = sequence_info.get(seq_id, "Unknown")
        
        # Check if visual data exists
        visual_path = seq_dir / "visual_data.pt"
        poses_path = seq_dir / "poses.json"
        
        if visual_path.exists() and poses_path.exists():
            # Get number of frames
            import torch
            visual_data = torch.load(visual_path)
            num_frames = visual_data.shape[0]
            
            print(f"  Sequence {seq_id}: {seq_name}")
            print(f"    - Frames: {num_frames}")
            print(f"    - Path: {seq_dir}")
        else:
            print(f"  Sequence {seq_id}: Missing data files!")
    
    # Save mapping
    test_mapping = {
        "test_sequences": [
            {
                "id": seq.name,
                "name": sequence_info.get(seq.name, "Unknown"),
                "path": str(seq)
            }
            for seq in test_sequences
        ]
    }
    
    with open("test_sequences_mapping.json", 'w') as f:
        json.dump(test_mapping, f, indent=2)
    
    print(f"\nTest sequence mapping saved to test_sequences_mapping.json")
    
    return test_sequences


if __name__ == "__main__":
    identify_test_sequences()