#!/usr/bin/env python3
"""
Pre-process TUM VI images to avoid resizing during training.
This will significantly speed up data loading.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial

def process_image(img_path, output_dir):
    """Process a single image - keep original resolution."""
    try:
        # Read image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read {img_path}")
            return False
        
        # Keep original resolution - no resizing for 512x512
        # For 1024x1024, use 2x2 binning for best quality
        if img.shape[0] == 1024 and img.shape[1] == 1024:
            # 2x2 binning: average every 2x2 block
            img = img.reshape(512, 2, 512, 2).mean(axis=(1, 3))
        
        # Create output path - fix the relative path calculation
        # img_path is like: /path/dataset-room1_512_16/mav0/cam0/data/image.png
        # We want: mav0/cam0/data/image.npy
        rel_path = img_path.relative_to(img_path.parents[3])  # relative to mav0 parent
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy array for fast loading (no resizing, just normalization)
        np.save(out_path.with_suffix('.npy'), img.astype(np.float32) / 255.0)
        
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def process_sequence(sequence_path, output_base_dir, num_workers=8):
    """Process all images in a sequence."""
    sequence_name = sequence_path.name
    print(f"\nProcessing {sequence_name}...")
    
    # Find all images
    img_dir = sequence_path / 'mav0' / 'cam0' / 'data'
    if not img_dir.exists():
        print(f"Image directory not found: {img_dir}")
        return
    
    img_paths = sorted(list(img_dir.glob('*.png')))
    print(f"Found {len(img_paths)} images")
    
    # Create output directory
    output_dir = output_base_dir / sequence_name
    
    # Process in parallel
    process_func = partial(process_image, output_dir=output_dir)
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, img_paths),
            total=len(img_paths),
            desc=f"Processing {sequence_name}"
        ))
    
    success_count = sum(results)
    print(f"Successfully processed {success_count}/{len(img_paths)} images")

def copy_other_data(sequence_path, output_base_dir):
    """Copy IMU and ground truth data."""
    sequence_name = sequence_path.name
    output_dir = output_base_dir / sequence_name
    
    # Copy IMU data
    imu_src = sequence_path / 'mav0' / 'imu0' / 'data.csv'
    if imu_src.exists():
        imu_dst = output_dir / 'mav0' / 'imu0'
        imu_dst.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(imu_src, imu_dst / 'data.csv')
    
    # Copy ground truth
    gt_src = sequence_path / 'mav0' / 'mocap0' / 'data.csv'
    if gt_src.exists():
        gt_dst = output_dir / 'mav0' / 'mocap0'
        gt_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gt_src, gt_dst / 'data.csv')
    
    # Copy cam0 data.csv (timestamps)
    cam_src = sequence_path / 'mav0' / 'cam0' / 'data.csv'
    if cam_src.exists():
        cam_dst = output_dir / 'mav0' / 'cam0'
        cam_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cam_src, cam_dst / 'data.csv')

def main():
    parser = argparse.ArgumentParser(description='Pre-process TUM VI images')
    parser.add_argument('--input-dir', type=str, 
                        default='/mnt/ssd_ext/incSeg-data/tumvi',
                        help='Input directory with TUM VI sequences')
    parser.add_argument('--output-dir', type=str,
                        default='/mnt/ssd_ext/incSeg-data/tumvi_preprocessed',
                        help='Output directory for preprocessed data')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='Number of parallel workers')
    parser.add_argument('--sequences', nargs='+', default=None,
                        help='Specific sequences to process (default: all)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find sequences
    if args.sequences:
        sequences = [input_dir / f"dataset-{seq}_512_16" for seq in args.sequences]
    else:
        # Process all sequences
        sequences = []
        for pattern in ['dataset-room*_512_16', 'dataset-corridor*_512_16']:
            sequences.extend(sorted(input_dir.glob(pattern)))
    
    print(f"Found {len(sequences)} sequences to process")
    print(f"Output directory: {output_dir}")
    
    # Process each sequence
    for seq_path in sequences:
        if seq_path.is_dir():
            # Process images
            process_sequence(seq_path, output_dir, args.num_workers)
            
            # Copy other data
            copy_other_data(seq_path, output_dir)
    
    print("\n\nPreprocessing complete!")
    print(f"Preprocessed data saved to: {output_dir}")
    print("\nTo use preprocessed data, update your training command:")
    print(f"  --data-dir {output_dir}")

if __name__ == "__main__":
    main()