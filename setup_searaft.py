#!/usr/bin/env python3
"""
Setup script to properly install SEA-RAFT for use with VIFT.
Run this before training with SEA-RAFT encoder.
"""

import os
import sys
import subprocess

def fix_imports_in_file(filepath, repo_dir):
    """Fix relative imports in a Python file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace relative imports with absolute imports
    replacements = [
        ('from update import', 'from core.update import'),
        ('from corr import', 'from core.corr import'),
        ('from extractor import', 'from core.extractor import'),
        ('from layer import', 'from core.layer import'),
        ('from utils.utils import', 'from core.utils.utils import'),
        ('from utils import', 'from core.utils import'),
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    # Special handling for raft.py - make HuggingFace optional
    if 'raft.py' in filepath:
        # Comment out HuggingFace import
        content = content.replace('from huggingface_hub import PyTorchModelHubMixin', 
                                  '# from huggingface_hub import PyTorchModelHubMixin')
        # Modify class definition to not inherit from PyTorchModelHubMixin
        content = content.replace('class RAFT(\n    nn.Module,\n    PyTorchModelHubMixin,', 
                                  'class RAFT(nn.Module):  # Modified for VIFT\n    """')
        # Close the docstring properly
        content = content.replace('    license="bsd-3-clause",\n):', 
                                  '    Original: https://github.com/princeton-vl/SEA-RAFT\n    """')
    
    # Write back
    with open(filepath, 'w') as f:
        f.write(content)

def setup_searaft():
    # Clone repository if needed
    repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'third_party', 'SEA-RAFT')
    
    # Create third_party directory if it doesn't exist
    os.makedirs(os.path.dirname(repo_dir), exist_ok=True)
    
    if not os.path.exists(repo_dir):
        print("Cloning SEA-RAFT repository...")
        subprocess.run(['git', 'clone', 'https://github.com/princeton-vl/SEA-RAFT.git', repo_dir], check=True)
    else:
        # Reset to clean state if already exists
        print("Resetting SEA-RAFT to clean state...")
        subprocess.run(['git', 'checkout', '.'], cwd=repo_dir, capture_output=True)
    
    # Fix imports in core files
    print("Fixing imports for compatibility...")
    core_files = ['raft.py', 'update.py', 'extractor.py', 'corr.py', 'layer.py']
    for filename in core_files:
        filepath = os.path.join(repo_dir, 'core', filename)
        if os.path.exists(filepath):
            fix_imports_in_file(filepath, repo_dir)
    
    # Add to Python path
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    
    # Test import
    try:
        from core.raft import RAFT
        print("✓ SEA-RAFT successfully installed and importable")
        
        # Check for weights
        weights_path = os.path.join(repo_dir, 'SEA-RAFT-Sintel.pth')
        if not os.path.exists(weights_path):
            print("\n⚠ IMPORTANT: Pretrained weights are REQUIRED for SEA-RAFT to work properly!")
            print("\n  SEA-RAFT was trained for weeks on large optical flow datasets.")
            print("  Without pretrained weights, it won't extract meaningful motion features.")
            print("\n  Please download from:")
            print("  https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW")
            print("\n  1. Download 'Tartan-C-T-TSKH-sintel368x768-S.pth'")
            print(f"  2. Save as: {weights_path}")
            print("\n  Then run this script again to verify the weights.")
            # Still return True since installation succeeded
            return True
        else:
            # Check if it's a valid PyTorch file
            try:
                import torch
                torch.load(weights_path, map_location='cpu', weights_only=False)
                print("✓ Pretrained weights already present")
            except Exception as e:
                print(f"✗ Invalid weights file. Please re-download from:")
                print("  https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW")
                return False
            
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    if setup_searaft():
        weights_exist = os.path.exists(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'third_party', 'SEA-RAFT', 'SEA-RAFT-Sintel.pth'
        ))
        
        if weights_exist:
            print("\n✓ SEA-RAFT setup complete with pretrained weights!")
            print("\nNext steps:")
            print("1. Test integration: python test_searaft_integration.py")
            print("2. Train with SEA-RAFT: python train_aria_from_scratch.py --use-searaft")
        else:
            print("\n✓ SEA-RAFT installation complete!")
            print("\n⚠ IMPORTANT: You MUST download pretrained weights before training!")
            print("\nNext steps:")
            print("1. Download pretrained weights (see instructions above)")
            print("2. Run this script again to verify weights")
            print("3. Test integration: python test_searaft_integration.py")
            print("4. Train with SEA-RAFT: python train_aria_from_scratch.py --use-searaft")
    else:
        print("\n✗ SEA-RAFT setup failed. Please check the error messages above.")
        sys.exit(1)