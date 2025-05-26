#!/usr/bin/env python3
"""
Download AriaEveryday Dataset using the official download URLs JSON file
"""

import os
import json
import requests
import hashlib
from pathlib import Path
from tqdm import tqdm
import argparse
import time

def download_file(url, filepath, expected_size=None, expected_sha1=None, max_retries=3):
    """Download a file with progress bar and verification."""
    for attempt in range(max_retries):
        try:
            # Create directory if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if file already exists and is valid
            if filepath.exists():
                if expected_sha1 and verify_file(filepath, expected_sha1):
                    print(f"✓ File already exists and verified: {filepath.name}")
                    return True
                else:
                    print(f"⚠ File exists but verification failed, re-downloading: {filepath.name}")
            
            # Download with progress bar
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as file, tqdm(
                desc=filepath.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
            
            # Verify download
            if expected_sha1:
                if verify_file(filepath, expected_sha1):
                    print(f"✓ Downloaded and verified: {filepath.name}")
                    return True
                else:
                    print(f"✗ SHA1 verification failed for {filepath.name}")
                    os.remove(filepath)
                    if attempt < max_retries - 1:
                        print(f"  Retrying... (attempt {attempt + 2}/{max_retries})")
                        time.sleep(2)
            else:
                print(f"✓ Downloaded: {filepath.name}")
                return True
                
        except Exception as e:
            print(f"✗ Error downloading {filepath.name}: {e}")
            if filepath.exists():
                os.remove(filepath)
            if attempt < max_retries - 1:
                print(f"  Retrying... (attempt {attempt + 2}/{max_retries})")
                time.sleep(2)
    
    return False

def verify_file(filepath, expected_sha1):
    """Verify file SHA1 checksum."""
    sha1 = hashlib.sha1()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            sha1.update(chunk)
    return sha1.hexdigest() == expected_sha1

def parse_json_and_download(json_path, output_dir, sequences_filter=None, file_types=None):
    """Parse the JSON file and download the dataset."""
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get sequences
    sequences = data.get('sequences', {})
    
    # Filter sequences if specified
    if sequences_filter:
        sequences = {k: v for k, v in sequences.items() if k in sequences_filter}
    
    print(f"Found {len(sequences)} sequences to download")
    
    # Count total files
    total_files = 0
    for seq_name, seq_data in sequences.items():
        for file_type, file_info in seq_data.items():
            if isinstance(file_info, dict) and 'download_url' in file_info:
                if not file_types or file_type in file_types:
                    total_files += 1
    
    print(f"Total files to download: {total_files}")
    
    # Download each sequence
    downloaded = 0
    failed = 0
    
    for seq_name, seq_data in sequences.items():
        print(f"\n📁 Processing sequence: {seq_name}")
        seq_dir = output_path / seq_name
        
        for file_type, file_info in seq_data.items():
            if isinstance(file_info, dict) and 'download_url' in file_info:
                # Skip if filtering by file type
                if file_types and file_type not in file_types:
                    continue
                
                filename = file_info.get('filename', f"{seq_name}_{file_type}.bin")
                filepath = seq_dir / filename
                url = file_info['download_url']
                expected_size = file_info.get('file_size_bytes')
                expected_sha1 = file_info.get('sha1sum')
                
                print(f"\n📥 Downloading {file_type}: {filename}")
                if expected_size:
                    print(f"   Size: {expected_size / 1024 / 1024:.1f} MB")
                
                success = download_file(url, filepath, expected_size, expected_sha1)
                
                if success:
                    downloaded += 1
                else:
                    failed += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Download Summary:")
    print(f"  ✓ Successfully downloaded: {downloaded} files")
    if failed > 0:
        print(f"  ✗ Failed downloads: {failed} files")
    print(f"  📁 Output directory: {output_path}")
    print(f"{'='*50}")

def main():
    parser = argparse.ArgumentParser(
        description="Download AriaEveryday dataset using the official JSON file"
    )
    parser.add_argument(
        "json_file",
        help="Path to AriaEverydayActivities_download_urls.json"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./data/aria_everyday",
        help="Directory to save downloaded files (default: ./data/aria_everyday)"
    )
    parser.add_argument(
        "--sequences", "-s",
        nargs="+",
        help="Specific sequences to download (e.g., loc5_script4_seq6_rec1)"
    )
    parser.add_argument(
        "--file-types", "-t",
        nargs="+",
        help="Specific file types to download (e.g., video_main_rgb mps_slam_trajectories)"
    )
    parser.add_argument(
        "--list-sequences", "-l",
        action="store_true",
        help="List available sequences without downloading"
    )
    parser.add_argument(
        "--list-file-types", "-lt",
        action="store_true",
        help="List available file types without downloading"
    )
    
    args = parser.parse_args()
    
    # Load JSON to list sequences/file types
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    sequences = data.get('sequences', {})
    
    if args.list_sequences:
        print("Available sequences:")
        for seq_name in sequences.keys():
            print(f"  - {seq_name}")
        return
    
    if args.list_file_types:
        print("Available file types:")
        file_types = set()
        for seq_data in sequences.values():
            for file_type in seq_data.keys():
                if isinstance(seq_data[file_type], dict):
                    file_types.add(file_type)
        for ft in sorted(file_types):
            print(f"  - {ft}")
        return
    
    # Download dataset
    parse_json_and_download(
        args.json_file,
        args.output_dir,
        args.sequences,
        args.file_types
    )

if __name__ == "__main__":
    main()