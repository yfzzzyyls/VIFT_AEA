#!/usr/bin/env python3
"""
Export trained model for production deployment.
Supports TorchScript and ONNX formats.
"""

import torch
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from models.flownet_lstm_transformer import FlowNetLSTMTransformer
from configs.flownet_lstm_transformer_config import ModelConfig


def export_torchscript(model, output_path, example_inputs):
    """Export model to TorchScript format."""
    print("Exporting to TorchScript...")
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_inputs)
    
    # Save
    traced_model.save(output_path)
    print(f"TorchScript model saved to {output_path}")
    
    # Verify
    loaded_model = torch.jit.load(output_path)
    with torch.no_grad():
        output_original = model(*example_inputs)
        output_loaded = loaded_model(*example_inputs)
        
        diff = torch.abs(output_original['poses'] - output_loaded['poses']).max()
        print(f"Verification: Max difference = {diff.item():.6f}")


def export_onnx(model, output_path, example_inputs, image_size, sequence_length):
    """Export model to ONNX format."""
    print("Exporting to ONNX...")
    
    # ONNX export requires fixed-size inputs
    images, imu_sequences = example_inputs
    
    # Flatten IMU sequences for ONNX
    # Convert list of lists to tensor [B, T-1, max_imu_len, 6]
    max_imu_len = 100  # Maximum IMU samples per interval
    B = len(imu_sequences)
    T_minus_1 = len(imu_sequences[0])
    
    imu_tensor = torch.zeros(B, T_minus_1, max_imu_len, 6)
    imu_lengths = torch.zeros(B, T_minus_1, dtype=torch.long)
    
    for b in range(B):
        for t in range(T_minus_1):
            imu_data = imu_sequences[b][t]
            length = min(len(imu_data), max_imu_len)
            imu_tensor[b, t, :length] = imu_data[:length]
            imu_lengths[b, t] = length
    
    # Create wrapper for ONNX export
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, images, imu_tensor, imu_lengths):
            # Reconstruct IMU sequences
            B, T_minus_1 = imu_tensor.shape[:2]
            imu_sequences = []
            
            for b in range(B):
                batch_sequences = []
                for t in range(T_minus_1):
                    length = imu_lengths[b, t]
                    batch_sequences.append(imu_tensor[b, t, :length])
                imu_sequences.append(batch_sequences)
            
            # Run model
            outputs = self.model(images, imu_sequences)
            return outputs['poses']
    
    wrapped_model = ONNXWrapper(model)
    wrapped_model.eval()
    
    # Export
    torch.onnx.export(
        wrapped_model,
        (images, imu_tensor, imu_lengths),
        output_path,
        input_names=['images', 'imu_data', 'imu_lengths'],
        output_names=['poses'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'imu_data': {0: 'batch_size'},
            'imu_lengths': {0: 'batch_size'},
            'poses': {0: 'batch_size'}
        },
        opset_version=11,
        verbose=True
    )
    
    print(f"ONNX model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export FlowNet-LSTM-Transformer model")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for exported model')
    parser.add_argument('--format', type=str, default='torchscript',
                       choices=['torchscript', 'onnx'],
                       help='Export format')
    parser.add_argument('--sequence-length', type=int, default=31,
                       help='Sequence length for export')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for export')
    parser.add_argument('--image-height', type=int, default=704,
                       help='Image height')
    parser.add_argument('--image-width', type=int, default=704,
                       help='Image width')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Extract config
    if 'config' in checkpoint and hasattr(checkpoint['config'], 'model'):
        model_config = checkpoint['config'].model
    else:
        model_config = ModelConfig()
    
    # Create model
    model = FlowNetLSTMTransformer(model_config)
    
    # Load weights
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Create example inputs
    print("Creating example inputs...")
    device = next(model.parameters()).device
    
    # Images
    images = torch.randn(
        args.batch_size, 
        args.sequence_length, 
        3, 
        args.image_height, 
        args.image_width,
        device=device
    )
    
    # IMU sequences (variable length)
    imu_sequences = []
    for b in range(args.batch_size):
        batch_sequences = []
        for t in range(args.sequence_length - 1):
            # Random number of IMU samples (40-60)
            num_samples = torch.randint(40, 60, (1,)).item()
            imu_data = torch.randn(num_samples, 6, device=device)
            batch_sequences.append(imu_data)
        imu_sequences.append(batch_sequences)
    
    # Export
    if args.format == 'torchscript':
        export_torchscript(model, args.output, (images, imu_sequences))
    else:
        export_onnx(model, args.output, (images, imu_sequences),
                   (args.image_height, args.image_width), args.sequence_length)
    
    print("\nExport completed successfully!")
    
    # Print usage example
    if args.format == 'torchscript':
        print("\nTo load in C++:")
        print("  torch::jit::script::Module module = torch::jit::load(\"model.pt\");")
        print("  module.eval();")
    else:
        print("\nTo load in ONNX Runtime:")
        print("  import onnxruntime as ort")
        print("  session = ort.InferenceSession('model.onnx')")


if __name__ == "__main__":
    main()