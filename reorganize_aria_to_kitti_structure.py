#!/usr/bin/env python3
"""
Reorganize Aria latent features to match KITTI's directory structure.
This allows using the original KITTI tester without modifications.

KITTI structure:
- kitti_latent_data/
  - val_10/
    - 0.npy, 0_gt.npy, 0_rot.npy, 0_w.npy
    - 1.npy, 1_gt.npy, 1_rot.npy, 1_w.npy
    - ...

We'll map:
- Aria sequence 016 → renamed as sequence 05
- Aria sequence 017 → renamed as sequence 07  
- Aria sequence 018 → renamed as sequence 10
"""

import os
import shutil
from pathlib import Path
import numpy as np


def reorganize_aria_to_kitti_format(source_dir, target_dir):
    """
    Reorganize Aria latent features to match KITTI structure.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Create target directory structure
    val_dir = target_dir / "val_10"
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapping from Aria sequences to KITTI sequence names
    sequence_mapping = {
        '016': '05',
        '017': '07', 
        '018': '10',
        '019': '10'  # We'll combine 018 and 019 as sequence 10
    }
    
    # Track global sample index
    global_idx = 0
    
    # Process each Aria sequence
    for aria_seq, kitti_seq in sequence_mapping.items():
        print(f"\nProcessing Aria sequence {aria_seq} (mapped to KITTI {kitti_seq})...")
        
        seq_dir = source_dir / aria_seq
        if not seq_dir.exists():
            print(f"Warning: Sequence directory {seq_dir} not found, skipping...")
            continue
            
        # Find all feature files in this sequence
        feature_files = sorted(seq_dir.glob("*[0-9].npy"))
        
        for feat_file in feature_files:
            # Extract sample index from filename
            local_idx = int(feat_file.stem)
            
            # Construct paths for all associated files
            gt_file = seq_dir / f"{local_idx}_gt.npy"
            rot_file = seq_dir / f"{local_idx}_rot.npy"
            w_file = seq_dir / f"{local_idx}_w.npy"
            
            # Check all files exist
            if not all(f.exists() for f in [gt_file, rot_file, w_file]):
                print(f"Warning: Missing files for sample {local_idx} in sequence {aria_seq}")
                continue
            
            # Copy files with new global index
            shutil.copy2(feat_file, val_dir / f"{global_idx}.npy")
            shutil.copy2(gt_file, val_dir / f"{global_idx}_gt.npy")
            shutil.copy2(rot_file, val_dir / f"{global_idx}_rot.npy")
            shutil.copy2(w_file, val_dir / f"{global_idx}_w.npy")
            
            global_idx += 1
        
        print(f"Processed {len(feature_files)} samples from sequence {aria_seq}")
    
    print(f"\nTotal samples reorganized: {global_idx}")
    print(f"Output directory: {val_dir}")
    
    # Create metadata file to track the mapping
    metadata = {
        'total_samples': global_idx,
        'sequence_mapping': sequence_mapping,
        'source_dir': str(source_dir),
        'note': 'Aria sequences reorganized to match KITTI structure'
    }
    
    import json
    with open(target_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nReorganization complete!")
    return val_dir


def main():
    # Paths
    source_dir = "/home/external/VIFT_AEA/aria_latent_kitti_format"
    target_dir = "/home/external/VIFT_AEA/data/aria_latent_as_kitti"
    
    print(f"Reorganizing Aria latent features from: {source_dir}")
    print(f"To KITTI-compatible structure at: {target_dir}")
    
    # Reorganize
    reorganize_aria_to_kitti_format(source_dir, target_dir)
    
    print("\nYou can now evaluate using:")
    print("python src/eval.py \\")
    print("    ckpt_path=/path/to/checkpoint.ckpt \\")
    print("    model=weighted_latent_vio_tf \\")
    print("    data=latent_kitti_vio \\")
    print(f"    data.test_loader.root_dir={target_dir}/val_10 \\")
    print("    trainer=gpu trainer.devices=1 logger=csv")


if __name__ == "__main__":
    main()