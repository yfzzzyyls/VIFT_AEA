#!/usr/bin/env python3
"""
Real-time inference with sliding window for FlowNet-LSTM-Transformer.
Simulates online VIO processing.
"""

import torch
import numpy as np
from pathlib import Path
import collections
import time
from typing import Optional, List, Tuple
import cv2

# Add project paths
import sys
sys.path.append(str(Path(__file__).parent))

from models.flownet_lstm_transformer import FlowNetLSTMTransformer
from configs.flownet_lstm_transformer_config import ModelConfig
from utils.pose_utils import integrate_poses, quaternion_to_rotation_matrix


class RealTimeVIO:
    """Real-time Visual-Inertial Odometry with sliding window."""
    
    def __init__(self, model_path: str, window_size: int = 31, 
                 image_size: Tuple[int, int] = (704, 704),
                 device: str = 'cuda'):
        """
        Initialize real-time VIO system.
        
        Args:
            model_path: Path to trained model checkpoint
            window_size: Sliding window size (must match training)
            image_size: Expected image size
            device: Computation device
        """
        self.window_size = window_size
        self.image_size = image_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Buffers for sliding window
        self.image_buffer = collections.deque(maxlen=window_size)
        self.imu_buffer = collections.deque(maxlen=window_size-1)
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Trajectory state
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.trajectory = [self.current_pose.copy()]
        
        # Timing statistics
        self.inference_times = collections.deque(maxlen=100)
        
        print(f"RealTimeVIO initialized:")
        print(f"  Window size: {window_size}")
        print(f"  Image size: {image_size}")
        print(f"  Device: {self.device}")
    
    def _load_model(self, model_path: str):
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
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
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def process_frame(self, image: np.ndarray, imu_data: List[np.ndarray]) -> Optional[dict]:
        """
        Process new frame with associated IMU data.
        
        Args:
            image: RGB image [H, W, 3] as numpy array
            imu_data: List of IMU measurements between this and previous frame
                     Each measurement is [ax, ay, az, gx, gy, gz]
        
        Returns:
            Dictionary with pose information if window is full, None otherwise
        """
        # Preprocess image
        image_tensor = self._preprocess_image(image)
        self.image_buffer.append(image_tensor)
        
        # Add IMU data (only if we have previous frame)
        if len(self.image_buffer) > 1:
            imu_tensor = torch.tensor(imu_data, dtype=torch.float32)
            self.imu_buffer.append(imu_tensor)
        
        # Run inference when buffer is full
        if len(self.image_buffer) == self.window_size:
            return self._run_inference()
        
        return None
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize if needed
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        return image_tensor
    
    def _run_inference(self) -> dict:
        """Run model inference on current window."""
        start_time = time.time()
        
        # Prepare inputs
        images = torch.stack(list(self.image_buffer)).unsqueeze(0).to(self.device)
        imu_sequences = [[imu.to(self.device) for imu in self.imu_buffer]]
        
        # Run model
        with torch.no_grad():
            outputs = self.model(images, imu_sequences)
            poses = outputs['poses'][0].cpu().numpy()  # [T-1, 7]
        
        # Update timing
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Get latest relative pose
        latest_pose = poses[-1]  # Most recent relative transformation
        
        # Update absolute pose
        self._update_pose(latest_pose)
        
        # Prepare output
        result = {
            'relative_pose': latest_pose,
            'absolute_pose': self.current_pose.copy(),
            'position': self.current_pose[:3, 3],
            'rotation_matrix': self.current_pose[:3, :3],
            'inference_time': inference_time,
            'avg_inference_time': np.mean(self.inference_times),
            'window_poses': poses  # All poses in window
        }
        
        return result
    
    def _update_pose(self, relative_pose: np.ndarray):
        """Update current absolute pose with relative transformation."""
        # Extract translation and rotation
        translation = relative_pose[:3]
        quaternion = relative_pose[3:]
        
        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_to_rotation_matrix(quaternion)
        
        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation
        
        # Update absolute pose
        self.current_pose = self.current_pose @ transform
        self.trajectory.append(self.current_pose.copy())
    
    def get_trajectory(self) -> np.ndarray:
        """Get full trajectory as array of positions."""
        positions = np.array([pose[:3, 3] for pose in self.trajectory])
        return positions
    
    def reset(self):
        """Reset VIO system to initial state."""
        self.image_buffer.clear()
        self.imu_buffer.clear()
        self.current_pose = np.eye(4)
        self.trajectory = [self.current_pose.copy()]
        self.inference_times.clear()


def simulate_realtime_processing():
    """Simulate real-time processing with example data."""
    import matplotlib.pyplot as plt
    from data.aria_variable_imu_dataset import AriaVariableIMUDataset
    
    # Initialize VIO system
    vio = RealTimeVIO(
        model_path='checkpoints/exp_704_4gpu/best_model.pt',
        window_size=31,
        image_size=(704, 704)
    )
    
    # Load test data
    dataset = AriaVariableIMUDataset(
        data_dir='../aria_processed',
        split='test',
        variable_length=False,
        sequence_length=100,  # Longer sequence for simulation
        image_size=(704, 704)
    )
    
    if len(dataset) == 0:
        print("No test data found!")
        return
    
    # Get first sequence
    sample = dataset[0]
    images = sample['images']  # [T, 3, H, W]
    imu_sequences = sample['imu_sequences']  # List of T-1 IMU tensors
    
    print(f"\nSimulating real-time processing on {len(images)} frames...")
    
    # Process frames sequentially
    results = []
    for t in range(len(images)):
        # Convert image to numpy
        image = images[t].permute(1, 2, 0).numpy() * 255.0
        image = image.astype(np.uint8)
        
        # Get IMU data for this interval
        if t > 0:
            imu_data = imu_sequences[t-1].numpy()
        else:
            imu_data = []
        
        # Process frame
        result = vio.process_frame(image, imu_data)
        
        if result is not None:
            results.append(result)
            print(f"Frame {t}: Pose estimated in {result['inference_time']*1000:.1f}ms")
    
    # Print statistics
    print(f"\nProcessing complete:")
    print(f"  Total frames: {len(images)}")
    print(f"  Poses estimated: {len(results)}")
    print(f"  Average inference time: {vio.avg_inference_time*1000:.1f}ms")
    print(f"  FPS: {1.0/vio.avg_inference_time:.1f}")
    
    # Plot trajectory
    trajectory = vio.get_trajectory()
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(trajectory[:, 0], trajectory[:, 1])
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory (Top View)')
    plt.axis('equal')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(trajectory[:, 0], trajectory[:, 2])
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.title('Trajectory (Side View)')
    plt.axis('equal')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    inference_times = list(vio.inference_times)
    plt.plot(np.array(inference_times) * 1000)
    plt.xlabel('Frame')
    plt.ylabel('Inference Time (ms)')
    plt.title('Inference Time per Frame')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.5, f"Average FPS: {1.0/np.mean(inference_times):.1f}\n"
                       f"Min time: {np.min(inference_times)*1000:.1f}ms\n"
                       f"Max time: {np.max(inference_times)*1000:.1f}ms",
             transform=plt.gca().transAxes, fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('realtime_inference_results.png')
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-time VIO inference")
    parser.add_argument('--simulate', action='store_true',
                       help='Run simulation with test data')
    
    args = parser.parse_args()
    
    if args.simulate:
        simulate_realtime_processing()
    else:
        print("Real-time VIO system ready for integration.")
        print("Example usage:")
        print("  vio = RealTimeVIO('model.pt')")
        print("  result = vio.process_frame(image, imu_data)")