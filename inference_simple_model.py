#!/usr/bin/env python3
"""Simple inference script for the direct training model."""

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D

class DirectVIOModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_proj = torch.nn.Linear(512, 256)
        self.imu_proj = torch.nn.Linear(256, 256)
        
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 256)
        )
        
        # Transformer
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Output heads
        self.translation_head = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 3)
        )
        
        self.rotation_head = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 4)  # quaternion
        )
        
    def forward(self, batch):
        # Extract features
        visual_features = batch['visual_features']  # [batch, seq, 512]
        imu_features = batch['imu_features']        # [batch, seq, 256]
        
        # Project inputs
        visual_emb = self.visual_proj(visual_features)
        imu_emb = self.imu_proj(imu_features)
        
        # Fuse modalities
        fused = torch.cat([visual_emb, imu_emb], dim=-1)
        fused = self.fusion(fused)
        
        # Apply transformer
        output = self.transformer(fused)
        
        # Get predictions
        translation = self.translation_head(output)
        rotation = self.rotation_head(output)
        
        # Normalize quaternions
        rotation = torch.nn.functional.normalize(rotation, p=2, dim=-1)
        
        return {
            'translation': translation,
            'rotation': rotation
        }

def load_test_data(sequence_id):
    """Load test data for a specific sequence."""
    # First, load the latent features from the model's training data
    latent_dir = Path("aria_latent_data_pretrained")
    
    # Try to find the sequence in test/val/train directories
    for split in ['test', 'val', 'train']:
        visual_path = latent_dir / split / f"{sequence_id}_visual.npy"
        if visual_path.exists():
            visual_features = np.load(visual_path)  # [seq_len, 512]
            imu_features = np.load(latent_dir / split / f"{sequence_id}_imu.npy")  # [seq_len, 256]
            gt_data = np.load(latent_dir / split / f"{sequence_id}_gt.npy")  # [seq_len, 7]
            print(f"Found sequence {sequence_id} in {split} split")
            return visual_features, imu_features, gt_data
    
    raise FileNotFoundError(f"Sequence {sequence_id} not found in any split")

def run_inference(model, visual_features, imu_features, window_size=200, stride=1, device='cuda'):
    """Run sliding window inference."""
    model.eval()
    
    seq_len = len(visual_features)
    predictions_trans = []
    predictions_rot = []
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, seq_len - window_size + 1, stride)):
            end_idx = start_idx + window_size
            
            # Prepare batch
            batch = {
                'visual_features': torch.FloatTensor(visual_features[start_idx:end_idx]).unsqueeze(0).to(device),
                'imu_features': torch.FloatTensor(imu_features[start_idx:end_idx]).unsqueeze(0).to(device)
            }
            
            # Run model
            output = model(batch)
            
            # Get predictions
            trans_pred = output['translation'][0].cpu().numpy()  # [seq_len, 3]
            rot_pred = output['rotation'][0].cpu().numpy()       # [seq_len, 4]
            
            # Store predictions
            predictions_trans.append(trans_pred)
            predictions_rot.append(rot_pred)
    
    # Stack predictions
    predictions_trans = np.array(predictions_trans)  # [n_windows, window_size, 3]
    predictions_rot = np.array(predictions_rot)      # [n_windows, window_size, 4]
    
    return predictions_trans, predictions_rot

def integrate_trajectory(predictions_trans, predictions_rot, stride=1):
    """Integrate predictions to get absolute trajectory."""
    # For simplicity, just use the center prediction from each window
    center_idx = len(predictions_trans[0]) // 2
    
    trans_deltas = []
    rot_deltas = []
    
    for i in range(len(predictions_trans)):
        trans_deltas.append(predictions_trans[i, center_idx])
        rot_deltas.append(predictions_rot[i, center_idx])
    
    trans_deltas = np.array(trans_deltas)
    rot_deltas = np.array(rot_deltas)
    
    # Integrate translations
    positions = np.zeros((len(trans_deltas) + 1, 3))
    for i in range(len(trans_deltas)):
        positions[i+1] = positions[i] + trans_deltas[i]
    
    # Integrate rotations
    rotations = [R.identity()]
    for q in rot_deltas:
        delta_rot = R.from_quat(q)
        rotations.append(rotations[-1] * delta_rot)
    
    return positions, rotations

def plot_trajectory_3d(pred_positions, gt_positions, output_path, duration_s=5):
    """Plot 3D trajectory comparison."""
    fps = 30
    num_frames = min(duration_s * fps, len(pred_positions))
    
    pred_pos = pred_positions[:num_frames]
    gt_pos = gt_positions[:num_frames]
    
    # Calculate error
    errors = np.linalg.norm(pred_pos - gt_pos, axis=1)
    mean_error_m = np.mean(errors)
    mean_error_cm = mean_error_m * 100
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 'b-', linewidth=2, label='Ground Truth')
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 'r--', linewidth=2, label='Prediction')
    
    # Add start/end markers
    ax.scatter(*gt_pos[0], c='green', s=100, marker='o', label='Start')
    ax.scatter(*gt_pos[-1], c='red', s=100, marker='s', label='End')
    
    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Trajectory Comparison - First {duration_s}s\nMean Error: {mean_error_cm:.2f} cm')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([gt_pos[:, 0].max() - gt_pos[:, 0].min(),
                         gt_pos[:, 1].max() - gt_pos[:, 1].min(),
                         gt_pos[:, 2].max() - gt_pos[:, 2].min()]).max() / 2.0
    
    mid_x = (gt_pos[:, 0].max() + gt_pos[:, 0].min()) * 0.5
    mid_y = (gt_pos[:, 1].max() + gt_pos[:, 1].min()) * 0.5
    mid_z = (gt_pos[:, 2].max() + gt_pos[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return mean_error_cm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-id', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--window-size', type=int, default=200)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default='simple_inference_results')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model with correct architecture
    print(f"Loading model from {args.checkpoint}")
    model = DirectVIOModel().to(args.device)
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load test data
    print(f"Loading test data for sequence {args.sequence_id}")
    visual_features, imu_features, gt_data = load_test_data(args.sequence_id)
    
    # Run inference
    print("Running inference...")
    predictions_trans, predictions_rot = run_inference(
        model, visual_features, imu_features, 
        window_size=args.window_size, 
        stride=args.stride,
        device=args.device
    )
    
    # Integrate trajectory
    print("Integrating trajectory...")
    pred_positions, pred_rotations = integrate_trajectory(predictions_trans, predictions_rot, stride=args.stride)
    
    # Get ground truth trajectory
    gt_positions = np.cumsum(gt_data[:, :3], axis=0)
    gt_positions = np.vstack([np.zeros(3), gt_positions])
    
    # Plot results
    print("Plotting results...")
    output_path = output_dir / f"trajectory_3d_seq_{args.sequence_id}.png"
    mean_error = plot_trajectory_3d(pred_positions, gt_positions, output_path)
    
    print(f"Results saved to {output_path}")
    print(f"Mean translation error: {mean_error:.2f} cm")
    
    # Save results
    np.savez(
        output_dir / f"results_seq_{args.sequence_id}.npz",
        pred_positions=pred_positions,
        gt_positions=gt_positions,
        predictions_trans=predictions_trans,
        predictions_rot=predictions_rot
    )

if __name__ == "__main__":
    main()