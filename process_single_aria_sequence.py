#!/usr/bin/env python3
"""
Process a single Aria sequence with ALL frames
Wrapper around process_aria_to_vift_quaternion.py for single sequence processing
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

def process_single_sequence(sequence_name, aria_path, output_dir, output_idx, max_frames=-1):
    """Process a single sequence by name"""
    
    # Create temporary directory with just this sequence
    temp_dir = Path(f"/tmp/aria_single_{output_idx}")
    temp_dir.mkdir(exist_ok=True)
    
    # Create symlink to the sequence
    sequence_path = Path(aria_path) / sequence_name
    if not sequence_path.exists():
        print(f"ERROR: Sequence {sequence_name} not found at {sequence_path}")
        return False
        
    temp_sequence = temp_dir / sequence_name
    if temp_sequence.exists():
        os.unlink(temp_sequence)
    os.symlink(sequence_path, temp_sequence)
    
    # Run the processor
    cmd = [
        "python", "scripts/process_aria_to_vift_quaternion.py",
        "--input-dir", str(temp_dir),
        "--output-dir", output_dir,
        "--max-frames", str(max_frames),
        "--start-index", "0",
        "--max-sequences", "1",
        "--folder-offset", str(int(output_idx)),
        "--device", "cpu"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        # Clean up
        os.unlink(temp_sequence)
        temp_dir.rmdir()
        
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-name', type=str, required=True)
    parser.add_argument('--aria-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--output-idx', type=str, required=True)
    parser.add_argument('--max-frames', type=int, default=-1)
    
    args = parser.parse_args()
    
    success = process_single_sequence(
        args.sequence_name,
        args.aria_path,
        args.output_dir,
        args.output_idx,
        args.max_frames
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()