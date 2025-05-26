#!/usr/bin/env python3
"""
Generate train/test sequence lists for AriaEveryday VIFT training
"""

import json
import argparse
from pathlib import Path

def create_sequence_lists(processed_data_dir: str, train_ratio: float = 0.6):
    """Create train/test sequence lists from processed AriaEveryday data"""
    
    data_dir = Path(processed_data_dir)
    summary_file = data_dir / "dataset_summary.json"
    
    if not summary_file.exists():
        print(f"❌ Dataset summary not found: {summary_file}")
        return
    
    # Load dataset summary
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    sequences = summary['sequences']
    total_sequences = len(sequences)
    
    # For 10 sequences: 6 training, 4 testing (0.6 ratio)
    train_count = int(total_sequences * train_ratio)
    test_count = total_sequences - train_count
    
    # Split sequences (first 6 for train, last 4 for test)
    train_sequences = sequences[:train_count]
    test_sequences = sequences[train_count:]
    
    # Create train list
    train_file = data_dir / "train_sequences.txt"
    with open(train_file, 'w') as f:
        for seq in train_sequences:
            f.write(f"{seq['sequence_id']}\n")
    
    # Create test list  
    test_file = data_dir / "test_sequences.txt"
    with open(test_file, 'w') as f:
        for seq in test_sequences:
            f.write(f"{seq['sequence_id']}\n")
    
    print(f"📊 Created sequence lists for 10-sequence subset:")
    print(f"   🏋️ Train: {train_count} sequences → {train_file}")
    print(f"   🧪 Test:  {test_count} sequences → {test_file}")
    
    # Create sequence mapping for reference
    mapping = {
        'train_sequences': [{'id': seq['sequence_id'], 'name': seq['sequence_name']} for seq in train_sequences],
        'test_sequences': [{'id': seq['sequence_id'], 'name': seq['sequence_name']} for seq in test_sequences],
        'train_count': train_count,
        'test_count': test_count,
        'train_ratio': train_ratio,
        'total_sequences': total_sequences
    }
    
    mapping_file = data_dir / "sequence_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"📄 Sequence mapping saved: {mapping_file}")
    return train_sequences, test_sequences

def main():
    parser = argparse.ArgumentParser(description='Create train/test sequence lists')
    parser.add_argument('--processed-data-dir', type=str,
                      default='/vast/fy2243/VIFT_AEA/data/aria_processed',
                      help='Directory with processed AriaEveryday data')
    parser.add_argument('--train-ratio', type=float, default=0.6,
                      help='Ratio of sequences for training (default: 0.6 for 6/10 split)')
    
    args = parser.parse_args()
    
    create_sequence_lists(args.processed_data_dir, args.train_ratio)

if __name__ == "__main__":
    main()