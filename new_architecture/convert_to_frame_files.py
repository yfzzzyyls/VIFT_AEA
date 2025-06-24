#!/usr/bin/env python3
"""
Convert large visual_data.pt files to individual frame files for efficient loading.
"""

import torch
from pathlib import Path
from tqdm import tqdm
import shutil
import argparse


def convert_sequence(seq_path: Path):
    """Convert a single sequence to frame files."""
    visual_data_path = seq_path / 'visual_data.pt'
    frames_dir = seq_path / 'frames'
    
    if not visual_data_path.exists():
        print(f"Skipping {seq_path.name}: No visual_data.pt found")
        return False
    
    if frames_dir.exists():
        print(f"Skipping {seq_path.name}: frames/ directory already exists")
        return False
    
    print(f"\nProcessing {seq_path.name}...")
    
    # Create frames directory
    frames_dir.mkdir(exist_ok=True)
    
    # Load visual data
    print("Loading visual_data.pt...")
    visual_data = torch.load(visual_data_path, map_location='cpu')
    num_frames = visual_data.shape[0]
    
    print(f"Converting {num_frames} frames...")
    # Save each frame individually
    for i in tqdm(range(num_frames), desc="Saving frames"):
        frame = visual_data[i]  # [3, H, W]
        frame_path = frames_dir / f'frame_{i:06d}.pt'
        torch.save(frame, frame_path)
    
    print(f"✅ Saved {num_frames} frame files")
    
    # Optionally remove the large file
    # visual_data_path.unlink()
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert visual_data.pt to individual frame files')
    parser.add_argument('--data-dir', type=str, default='../aria_processed',
                       help='Path to processed Aria data')
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
                       help='Specific sequences to convert (e.g., 000 001). If not specified, converts all.')
    parser.add_argument('--remove-original', action='store_true',
                       help='Remove original visual_data.pt after conversion')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Find sequences to convert
    if args.sequences:
        sequences = [data_dir / seq for seq in args.sequences]
    else:
        sequences = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    print(f"Found {len(sequences)} sequences to process")
    
    converted = 0
    for seq_path in sequences:
        if convert_sequence(seq_path):
            converted += 1
            
            if args.remove_original:
                visual_data_path = seq_path / 'visual_data.pt'
                if visual_data_path.exists():
                    print(f"Removing {visual_data_path.name}...")
                    visual_data_path.unlink()
    
    print(f"\n✅ Converted {converted} sequences")
    
    # Calculate space saved
    total_frame_size = 0
    total_original_size = 0
    
    for seq_path in sequences:
        frames_dir = seq_path / 'frames'
        if frames_dir.exists():
            frame_files = list(frames_dir.glob('frame_*.pt'))
            if frame_files:
                # Sample size from first frame
                frame_size = frame_files[0].stat().st_size
                total_frame_size += frame_size * len(frame_files)
                
                # Original size
                total_original_size += 12 * 1024**3  # 12GB per sequence
    
    if total_frame_size > 0:
        print(f"\nStorage comparison:")
        print(f"Original: {total_original_size / 1024**3:.1f} GB")
        print(f"Frames:   {total_frame_size / 1024**3:.1f} GB")
        print(f"Overhead: {(total_frame_size - total_original_size) / 1024**3:.1f} GB")


if __name__ == "__main__":
    main()