#!/usr/bin/env python3
"""
Download the Visual-Selective-VIO pretrained model from the official repository.
"""

import os
import sys
import subprocess
import shutil

def verify_model_file(filepath):
    """Verify the downloaded model file."""
    if not os.path.exists(filepath):
        return False, "File does not exist"
    
    size = os.path.getsize(filepath)
    if size < 100_000_000:  # Less than 100MB
        return False, f"File too small: {size} bytes (expected ~185MB)"
    
    # Check if it's an HTML file (redirect page)
    with open(filepath, 'rb') as f:
        header = f.read(100)
        if b'<!DOCTYPE' in header or b'<html' in header:
            return False, "File is HTML (redirect page), not a model file"
    
    return True, f"File appears valid: {size / 1024 / 1024:.1f} MB"

def main():
    model_dir = "pretrained_models"
    model_filename = "vf_512_if_256_3e-05.model"
    model_path = os.path.join(model_dir, model_filename)
    
    # Create directory if needed
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if valid model already exists
    if os.path.exists(model_path):
        valid, message = verify_model_file(model_path)
        if valid:
            print(f"✓ Model already exists and is valid: {message}")
            return
        else:
            print(f"✗ Existing model is invalid: {message}")
            os.remove(model_path)
    
    print("Visual-Selective-VIO Model Download")
    print("=" * 50)
    
    # Download from the official Visual-Selective-VIO repository
    folder_id = "1KrxpvUV9Bn5SwUlrDKe76T2dqF1ooZyk"
    
    print("\nDownloading from Visual-Selective-VIO official repository...")
    print("This will download all pretrained models (~700MB total)")
    
    try:
        # Check if gdown is installed
        try:
            import gdown
        except ImportError:
            print("\n✗ Error: gdown is not installed.")
            print("Please install it with: pip install gdown")
            return
        
        # Download the folder
        temp_dir = "VIO-models-temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        print("\nDownloading models...")
        cmd = [sys.executable, "-m", "gdown", "--folder", "--id", folder_id, "-O", temp_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"\n✗ Download failed: {result.stderr}")
            return
        
        # Find and copy the specific model
        source_path = os.path.join(temp_dir, "VIO-models", model_filename)
        if not os.path.exists(source_path):
            # Try without VIO-models subdirectory
            source_path = os.path.join(temp_dir, model_filename)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, model_path)
            print(f"\n✓ Model copied to {model_path}")
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            # Verify
            valid, message = verify_model_file(model_path)
            if valid:
                print(f"✓ Download successful: {message}")
            else:
                print(f"✗ Downloaded file is invalid: {message}")
        else:
            print(f"\n✗ Could not find {model_filename} in downloaded files")
            print(f"Downloaded files are in: {temp_dir}")
            
    except Exception as e:
        print(f"\n✗ Error during download: {e}")
        print("\nIf automatic download fails, you can manually download from:")
        print(f"https://drive.google.com/drive/folders/{folder_id}")
        print(f"Then save '{model_filename}' to '{model_path}'")

def check_gdown_installation():
    """Check if gdown is installed and provide installation instructions if not."""
    try:
        import gdown
        return True
    except ImportError:
        print("✗ gdown is not installed.")
        print("\nPlease install it with:")
        print("  pip install gdown")
        print("\nOr if using a virtual environment:")
        print("  source ~/venv/py39/bin/activate")
        print("  pip install gdown")
        return False

if __name__ == "__main__":
    if check_gdown_installation():
        main()