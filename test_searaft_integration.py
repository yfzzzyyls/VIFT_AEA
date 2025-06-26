#!/usr/bin/env python3
"""
Test SEA-RAFT integration with VIFT.
Verifies shapes, memory usage, and speed.
"""

import torch
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.models.components.vsvio import TransformerVIO
from train_aria_from_scratch import VIFTFromScratch


def test_searaft_encoder():
    """Test the SEA-RAFT encoder standalone."""
    print("=" * 60)
    print("Testing SEA-RAFT Feature Encoder")
    print("=" * 60)
    
    # Create config
    class Config:
        seq_len = 11
        img_w = 704
        img_h = 704
        v_f_len = 512
        i_f_len = 256
        imu_dropout = 0.2
        use_searaft = True
    
    opt = Config()
    
    try:
        from src.models.components.searaft_encoder import SEARAFTFeatureEncoder
        encoder = SEARAFTFeatureEncoder(opt).cuda()
        print("✓ SEA-RAFT encoder created successfully")
        
        # Test with dummy input
        dummy_input = torch.randn(2, 6, 704, 704).cuda()  # 2 frame pairs
        
        # Memory before
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1e9
        
        # Forward pass
        t0 = time.time()
        with torch.cuda.amp.autocast():
            output = encoder.encode_image(dummy_input)
        torch.cuda.synchronize()
        t1 = time.time()
        
        # Memory after
        mem_after = torch.cuda.memory_allocated() / 1e9
        
        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Expected shape: [2, {opt.v_f_len}]")
        print(f"  - Time: {(t1-t0)*1000:.1f}ms")
        print(f"  - Memory used: {mem_after - mem_before:.2f}GB")
        
        assert output.shape == (2, opt.v_f_len), f"Wrong output shape: {output.shape}"
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True


def test_full_model():
    """Test the full VIFT model with SEA-RAFT."""
    print("\n" + "=" * 60)
    print("Testing Full VIFT Model with SEA-RAFT")
    print("=" * 60)
    
    try:
        # Create model with SEA-RAFT
        model = VIFTFromScratch(use_searaft=True).cuda()
        print("✓ VIFT model created with SEA-RAFT encoder")
        
        # Dummy batch
        batch_size = 2
        seq_len = 11
        images = torch.randn(batch_size, seq_len, 3, 704, 704).cuda()
        imu = torch.randn(batch_size, seq_len-1, 50, 6).cuda()  # Variable IMU
        
        # Forward pass
        torch.cuda.synchronize()
        t0 = time.time()
        
        batch = {
            'images': images,
            'imu': imu,
            'gt_poses': torch.randn(batch_size, seq_len-1, 7).cuda()  # Dummy GT
        }
        
        with torch.cuda.amp.autocast():
            output = model(batch)
        
        torch.cuda.synchronize()
        t1 = time.time()
        
        print(f"✓ Full forward pass successful")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Sequence length: {seq_len}")
        print(f"  - Output shape: {output['poses'].shape}")
        print(f"  - Time: {(t1-t0)*1000:.1f}ms")
        print(f"  - Time per frame pair: {(t1-t0)*1000/(batch_size*(seq_len-1)):.1f}ms")
        
        # Test gradient flow
        loss = output['poses'].mean()
        loss.backward()
        print("✓ Backward pass successful")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def compare_encoders():
    """Compare CNN vs SEA-RAFT encoders."""
    print("\n" + "=" * 60)
    print("Comparing CNN vs SEA-RAFT Encoders")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 11
    
    # Test data
    images = torch.randn(batch_size, seq_len, 3, 704, 704).cuda()
    imu = torch.randn(batch_size, seq_len-1, 50, 6).cuda()
    batch = {
        'images': images,
        'imu': imu,
        'gt_poses': torch.randn(batch_size, seq_len-1, 7).cuda()
    }
    
    # Test CNN
    model_cnn = VIFTFromScratch(use_searaft=False).cuda()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.cuda.amp.autocast():
        _ = model_cnn(batch)
    torch.cuda.synchronize()
    t_cnn = time.time() - t0
    
    # Test SEA-RAFT
    model_raft = VIFTFromScratch(use_searaft=True).cuda()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.cuda.amp.autocast():
        _ = model_raft(batch)
    torch.cuda.synchronize()
    t_raft = time.time() - t0
    
    print(f"CNN Encoder: {t_cnn*1000:.1f}ms")
    print(f"SEA-RAFT Encoder: {t_raft*1000:.1f}ms")
    print(f"Slowdown: {t_raft/t_cnn:.1f}x")
    
    return True


if __name__ == "__main__":
    print("Testing SEA-RAFT Integration\n")
    
    # Test 1: Encoder only
    if not test_searaft_encoder():
        print("\nEncoder test failed!")
        sys.exit(1)
    
    # Test 2: Full model
    if not test_full_model():
        print("\nFull model test failed!")
        sys.exit(1)
    
    # Test 3: Speed comparison
    if not compare_encoders():
        print("\nComparison test failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)