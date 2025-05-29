#!/usr/bin/env python3
"""
Check AriaEveryday dataset for corrupted or broken sequences
"""
import argparse
from pathlib import Path
import cv2
import json
import shutil

def check_video_file(video_path):
    """Check if a video file can be opened and read"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, "Cannot open video"
        
        # Try to read first frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return False, "Cannot read frames"
        
        return True, "OK"
    except Exception as e:
        return False, f"Exception: {str(e)}"

def check_sequence(seq_dir):
    """Check if a sequence has all required files and they're valid"""
    issues = []
    
    # Check for RGB video
    rgb_pattern = "*_preview_rgb.mp4"
    rgb_files = list(seq_dir.glob(rgb_pattern))
    
    if not rgb_files:
        issues.append("Missing RGB video file")
    else:
        for rgb_file in rgb_files:
            valid, msg = check_video_file(rgb_file)
            if not valid:
                issues.append(f"Corrupted RGB video: {msg}")
    
    # Check for SLAM trajectory
    slam_pattern = "*_mps_slam_trajectories.zip"
    slam_files = list(seq_dir.glob(slam_pattern))
    
    if not slam_files:
        issues.append("Missing SLAM trajectory file")
    
    # Check for summary
    summary_pattern = "*_mps_slam_summary.zip"
    summary_files = list(seq_dir.glob(summary_pattern))
    
    if not summary_files:
        issues.append("Missing SLAM summary file")
    
    return len(issues) == 0, issues

def main():
    parser = argparse.ArgumentParser(description="Check AriaEveryday dataset for corrupted sequences")
    parser.add_argument("--data_dir", type=str, default="data/aria_everyday", 
                       help="Directory containing AriaEveryday sequences")
    parser.add_argument("--remove", action="store_true", 
                       help="Remove corrupted sequences")
    parser.add_argument("--backup_dir", type=str, default="data/aria_corrupted",
                       help="Directory to move corrupted sequences (if --remove)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        return
    
    # Find all sequence directories
    sequences = []
    for item in sorted(data_dir.iterdir()):
        if item.is_dir() and item.name.startswith("loc"):
            sequences.append(item)
    
    print(f"Found {len(sequences)} sequences to check\n")
    
    corrupted_sequences = []
    
    # Check each sequence
    for seq_dir in sequences:
        print(f"Checking {seq_dir.name}...", end=" ")
        valid, issues = check_sequence(seq_dir)
        
        if valid:
            print("✅ OK")
        else:
            print(f"❌ CORRUPTED")
            for issue in issues:
                print(f"  - {issue}")
            corrupted_sequences.append(seq_dir)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Total sequences: {len(sequences)}")
    print(f"Valid sequences: {len(sequences) - len(corrupted_sequences)}")
    print(f"Corrupted sequences: {len(corrupted_sequences)}")
    
    if corrupted_sequences:
        print("\nCorrupted sequences:")
        for seq in corrupted_sequences:
            print(f"  - {seq.name}")
    
    # Remove corrupted sequences if requested
    if args.remove and corrupted_sequences:
        print(f"\n{'='*60}")
        print(f"Moving corrupted sequences to {args.backup_dir}")
        
        backup_dir = Path(args.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for seq_dir in corrupted_sequences:
            dest = backup_dir / seq_dir.name
            print(f"Moving {seq_dir.name} to backup...", end=" ")
            try:
                shutil.move(str(seq_dir), str(dest))
                print("✅")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\nCorrupted sequences moved to {backup_dir}")
        print(f"Clean dataset now has {len(sequences) - len(corrupted_sequences)} sequences")

if __name__ == "__main__":
    main()