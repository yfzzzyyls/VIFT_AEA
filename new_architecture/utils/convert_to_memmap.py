"""
Convert PyTorch tensors to memory-mapped numpy arrays for efficient loading.
This allows loading only the needed portions of large files.
"""

import torch
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm


def convert_visual_data_to_memmap(pt_path: Path, output_dir: Path):
    """Convert visual_data.pt to memory-mapped format."""
    print(f"Converting {pt_path}...")
    
    # Load tensor to get shape and dtype
    data = torch.load(pt_path, map_location='cpu')
    shape = data.shape
    dtype = data.numpy().dtype
    
    # Create output paths
    mmap_path = output_dir / 'visual_data.dat'
    meta_path = output_dir / 'visual_data_meta.json'
    
    # Create memory-mapped array
    mmap_array = np.memmap(mmap_path, dtype=dtype, mode='w+', shape=shape)
    
    # Copy data
    mmap_array[:] = data.numpy()
    mmap_array.flush()
    
    # Save metadata
    metadata = {
        'shape': list(shape),
        'dtype': str(dtype),
        'original_file': str(pt_path)
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    del data
    print(f"Saved to {mmap_path} ({shape}, {dtype})")
    
    return mmap_path, meta_path


def convert_imu_data_to_memmap(pt_path: Path, output_dir: Path):
    """Convert imu_data.pt to memory-mapped format."""
    print(f"Converting {pt_path}...")
    
    # Load data
    data = torch.load(pt_path, map_location='cpu')
    
    # IMU data is a list of variable-length tensors
    # We'll save each as a separate file with an index
    imu_dir = output_dir / 'imu_sequences'
    imu_dir.mkdir(exist_ok=True)
    
    index = []
    for i, imu_seq in enumerate(data):
        seq_path = imu_dir / f'seq_{i:06d}.npy'
        np.save(seq_path, imu_seq.numpy())
        index.append({
            'index': i,
            'shape': list(imu_seq.shape),
            'dtype': str(imu_seq.numpy().dtype)
        })
    
    # Save index
    index_path = output_dir / 'imu_index.json'
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"Saved {len(data)} IMU sequences to {imu_dir}")
    
    return imu_dir, index_path


def convert_sequence(seq_dir: Path, output_base: Path):
    """Convert a single sequence to memory-mapped format."""
    seq_name = seq_dir.name
    output_dir = output_base / seq_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert visual data
    visual_pt = seq_dir / 'visual_data.pt'
    if visual_pt.exists():
        convert_visual_data_to_memmap(visual_pt, output_dir)
    
    # Convert IMU data
    imu_pt = seq_dir / 'imu_data.pt'
    if imu_pt.exists():
        convert_imu_data_to_memmap(imu_pt, output_dir)
    
    # Copy other files
    for file_name in ['poses_quaternion.json', 'metadata.json']:
        src = seq_dir / file_name
        if src.exists():
            dst = output_dir / file_name
            import shutil
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert Aria data to memory-mapped format')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory with PT files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for memory-mapped files')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy splits.json if exists
    splits_file = input_dir / 'splits.json'
    if splits_file.exists():
        import shutil
        shutil.copy2(splits_file, output_dir / 'splits.json')
    
    # Convert all sequences
    sequences = [d for d in input_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    
    for seq_dir in tqdm(sequences, desc='Converting sequences'):
        convert_sequence(seq_dir, output_dir)
    
    print(f"Conversion complete! Output saved to {output_dir}")


if __name__ == '__main__':
    main()