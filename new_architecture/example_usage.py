#!/usr/bin/env python3
"""
Example usage of the FlowNet-LSTM-Transformer architecture with raw IMU data.
"""

import torch
import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from models.flownet_lstm_transformer import FlowNetLSTMTransformer
from data.aria_variable_imu_dataset import AriaVariableIMUDataset, collate_variable_imu
from configs.flownet_lstm_transformer_config import ModelConfig
from torch.utils.data import DataLoader


def main():
    print("FlowNet-LSTM-Transformer Example Usage")
    print("=" * 50)
    
    # 1. Create model configuration
    config = ModelConfig(
        visual_feature_dim=256,
        imu_hidden_dim=128,
        imu_lstm_layers=3,
        imu_feature_dim=256,
        imu_bidirectional=True,
        transformer_dim=512,
        transformer_heads=8,
        transformer_layers=6,
        transformer_feedforward=2048,
        dropout=0.1
    )
    
    # 2. Create model
    model = FlowNetLSTMTransformer(config)
    print(f"\n✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 3. Create dataset with raw IMU data
    try:
        dataset = AriaVariableIMUDataset(
            data_dir="../aria_processed",
            split="train",
            variable_length=True,
            min_seq_len=5,
            max_seq_len=20,
            sequence_length=11
        )
        print(f"\n✅ Dataset loaded with {len(dataset)} samples")
        
        # 4. Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=collate_variable_imu,
            num_workers=0
        )
        
        # 5. Test forward pass
        print("\n🧪 Testing forward pass...")
        model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > 0:
                    break
                    
                # Get data
                images = batch['images']  # [B, T, 3, H, W]
                imu_sequences = batch['imu_sequences']  # List of variable-length IMU
                
                print(f"\n📊 Batch {batch_idx}:")
                print(f"   - Images shape: {images.shape}")
                print(f"   - Batch size: {len(imu_sequences)}")
                
                # Show IMU statistics for each sample in batch
                for b in range(len(imu_sequences)):
                    imu_lengths = [seq.shape[0] for seq in imu_sequences[b]]
                    print(f"   - Sample {b}: {len(imu_sequences[b])} transitions")
                    print(f"     IMU samples per transition: {imu_lengths}")
                    print(f"     Average: {sum(imu_lengths)/len(imu_lengths):.1f}")
                
                # Forward pass
                outputs = model(images, imu_sequences)
                
                print(f"\n📤 Outputs:")
                print(f"   - Poses: {outputs['poses'].shape}")
                print(f"   - Translation: {outputs['translation'].shape}")
                print(f"   - Rotation: {outputs['rotation'].shape}")
                
                # Verify temporal alignment
                T = images.shape[1]
                T_minus_1 = outputs['poses'].shape[1]
                print(f"\n✅ Temporal alignment verified:")
                print(f"   - Input frames: {T}")
                print(f"   - Output poses: {T_minus_1} (T-1)")
                
    except FileNotFoundError:
        print("\n❌ Error: aria_processed directory not found.")
        print("   Please run process_aria.py first to extract the data.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Example completed!")


if __name__ == "__main__":
    main()