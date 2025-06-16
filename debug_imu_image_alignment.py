#!/usr/bin/env python3
"""
Single-sample overfit test to verify IMU-image stream alignment.
Tests whether the two sensors are properly synchronized by overfitting
on a high-motion sample and analyzing cross-correlation.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import correlate, correlation_lags
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from train_aria_from_scratch import VIFTFromScratch
from src.data.components.aria_raw_dataset import AriaRawDataModule


def compute_optical_flow_magnitude(img1, img2):
    """Compute optical flow magnitude between two images."""
    # Convert PyTorch tensors (CHW) to NumPy arrays (HWC) for OpenCV
    if torch.is_tensor(img1):
        img1 = img1.permute(1, 2, 0).cpu().numpy()
        img1 = (img1 * 255).astype(np.uint8)
    if torch.is_tensor(img2):
        img2 = img2.permute(1, 2, 0).cpu().numpy()
        img2 = (img2 * 255).astype(np.uint8)
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag.mean()


def find_high_motion_window(dataloader, model, sequence_mapping=None, num_samples=10, target_sequence="005"):
    """Find windows with high motion in both image and IMU."""
    motion_scores = []
    target_samples = []
    
    print(f"Searching for sequence {target_sequence} (loc1_script2_seq1_rec1 - standing from couch and turning right)...")
    
    # Iterate through more samples to find the target sequence
    max_search = 500  # Search through more samples to find our target
    for idx, batch in enumerate(dataloader):
        if idx >= max_search:
            break
            
        # Handle dictionary format from dataset
        if isinstance(batch, dict):
            images = batch['images']
            imus = batch['imu']
            gt_poses = batch['gt_poses']
            seq_name = batch['seq_name'][0] if 'seq_name' in batch else 'unknown'
            start_idx = batch['start_idx'].item() if 'start_idx' in batch else -1
        else:
            # Try unpacking as tuple
            if len(batch) == 3:
                images, imus, gt_poses = batch
                seq_name = 'unknown'
                start_idx = -1
            else:
                print(f"Unexpected batch format")
                continue
        
        # Compute IMU motion magnitude
        acc_norm = torch.norm(imus[..., :3], dim=-1).mean().item()  # m/s²
        gyro_norm = torch.norm(imus[..., 3:], dim=-1).mean().item()  # rad/s
        
        # Compute image motion (simplified - using pose change as proxy)
        if gt_poses.shape[1] > 1:
            trans_change = torch.norm(gt_poses[:, -1, :3] - gt_poses[:, 0, :3]).item()
        else:
            trans_change = 0
            
        score = acc_norm + gyro_norm * 0.1 + trans_change * 10
        
        # Collect samples from target sequence
        if seq_name == target_sequence:
            target_samples.append((idx, score, acc_norm, gyro_norm, images, imus, gt_poses, seq_name, start_idx))
            # The standing up motion should be in early frames (0-200)
            if start_idx < 200:
                print(f"  Found target seq {seq_name} at idx {idx}: frame={start_idx}, acc={acc_norm:.3f}m/s², gyro={gyro_norm:.3f}rad/s, score={score:.3f}")
        
    
    # Use target sequence samples if found, otherwise report error
    if target_samples:
        print(f"\nFound {len(target_samples)} samples from target sequence {target_sequence}")
        # Sort target samples by motion score and pick the highest
        target_samples.sort(key=lambda x: x[1], reverse=True)
        best_idx, best_score, acc_norm, gyro_norm, images, imus, gt_poses, seq_name, start_idx = target_samples[0]
    else:
        print(f"\n❌ ERROR: Could not find target sequence {target_sequence} (loc1_script2_seq1_rec1)")
        print(f"Searched through {idx+1} samples but sequence was not found.")
        print("Please ensure sequence 005 is in the training set.")
        sys.exit(1)
    
    print(f"\nSelected sample {best_idx} with score={best_score:.3f}")
    print(f"  Sequence ID: {seq_name}")
    if sequence_mapping and seq_name in sequence_mapping:
        print(f"  Aria sequence: {sequence_mapping[seq_name]}")
    print(f"  Start frame: {start_idx}")
    print(f"  Accelerometer: {acc_norm:.3f}m/s² ({acc_norm/9.81:.3f}g)")
    print(f"  Gyroscope: {gyro_norm:.3f}rad/s")
    
    # Add finite assertions
    assert torch.isfinite(imus).all(), "IMU tensor contains inf/nan"
    assert torch.isfinite(images).all(), "Image tensor contains inf/nan"
    
    return images, imus, gt_poses


def compute_cross_correlation(pred_signal, gt_signal):
    """Compute cross-correlation between predicted and ground truth signals."""
    # Normalize signals
    pred_norm = (pred_signal - pred_signal.mean()) / (pred_signal.std() + 1e-8)
    gt_norm = (gt_signal - gt_signal.mean()) / (gt_signal.std() + 1e-8)
    
    # Convert to numpy
    pred_np = pred_norm.cpu().numpy()
    gt_np = gt_norm.cpu().numpy()
    
    # Compute correlation
    correlation = correlate(pred_np, gt_np, mode='same')
    lags = correlation_lags(len(pred_np), len(gt_np), mode='same')
    
    return correlation, lags


def overfit_single_sample(model, window_img, window_imu, gt_pose, max_steps=800, eval_mode=False):
    """Overfit model on a single sample and monitor convergence."""
    if eval_mode:
        model.eval()  # Disable BatchNorm & Dropout
        # Freeze BatchNorm completely for stable single-sample training
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                m.eval()  # Use pre-trained stats
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
        print("Model in eval mode with BatchNorm frozen")
    else:
        model.train()
    
    # Enable anomaly detection for precise nan back-trace
    torch.autograd.set_detect_anomaly(True)
    
    # Ensure all model parameters are finite at startup
    for p in model.parameters():
        assert torch.isfinite(p).all(), "NaN/inf in model weights"
    
    # Moderate learning rate for stable convergence
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)  # 5x lower than baseline, stable for single sample
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    window_img = window_img.to(device)
    window_imu = window_imu.to(device)
    gt_pose = gt_pose.to(device)
    
    # Bias-correct accelerometer (remove mean gravity component)
    window_imu_corrected = window_imu.clone()
    # Reshape to [B, 10, 11, 6] for proper mean calculation
    B, L, C = window_imu.shape
    window_imu_reshaped = window_imu.reshape(B, -1, 11, 6)  # [B, 10, 11, 6]
    # Subtract mean acceleration for each window
    acc_mean = window_imu_reshaped[..., :3].mean(dim=2, keepdim=True)  # Mean over 11 samples per interval
    window_imu_reshaped[..., :3] -= acc_mean
    window_imu_corrected = window_imu_reshaped.reshape(B, L, C)
    print(f"Removed mean gravity vector: {acc_mean[0, 0].cpu().numpy()} m/s²")
    
    # Check IMU values after bias correction
    acc_norm_before = torch.norm(window_imu[..., :3], dim=-1).mean().item()
    acc_norm_after = torch.norm(window_imu_corrected[..., :3], dim=-1).mean().item()  # Fixed: now using corrected tensor
    print(f"Accelerometer magnitude: before={acc_norm_before:.3f} m/s², after={acc_norm_after:.3f} m/s²")
    
    # Debug: Check a few IMU samples after correction
    print(f"First few IMU samples after bias correction (accel only):")
    print(window_imu_corrected[0, :5, :3].cpu().numpy())
    
    losses = []
    trans_correlations = []
    rot_correlations = []
    
    # Setup live plotting
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for step in range(max_steps):
        # Forward pass - model expects dictionary input with GPU tensors
        batch = {
            'images': window_img,    # Already on GPU
            'imu': window_imu_corrected,  # Use bias-corrected IMU
            'gt_poses': gt_pose      # Already on GPU
        }
        predictions = model(batch)
        pred_pose = predictions['poses']
        
        # Check for NaN in predictions
        if torch.isnan(pred_pose).any():
            print(f"\nNaN in predictions at step {step}!")
            print(f"  pred_pose has NaN: {torch.isnan(pred_pose).sum().item()} values")
            break
        
        # Compute loss (translation + rotation)
        trans_loss = torch.norm(pred_pose[..., :3] - gt_pose[..., :3], dim=-1).mean()
        
        # Quaternion loss (geodesic distance)
        pred_quat = pred_pose[..., 3:7]
        gt_quat = gt_pose[..., 3:7]
        
        # Normalize quaternions before computing loss
        pred_quat = pred_quat / (torch.norm(pred_quat, dim=-1, keepdim=True) + 1e-8)
        gt_quat = gt_quat / (torch.norm(gt_quat, dim=-1, keepdim=True) + 1e-8)
        
        # Compute quaternion geodesic distance using atan2 for numerical stability
        dot_product = torch.sum(pred_quat * gt_quat, dim=-1)
        
        # Use atan2(sqrt(1-x²), x) which has finite slope at |x|→1
        # This avoids the derivative blow-up of acos near ±1
        abs_dot = dot_product.abs()
        # Clamp to avoid numerical issues in sqrt
        abs_dot_clamped = abs_dot.clamp(max=1.0 - 1e-7)
        angle = 2.0 * torch.atan2(torch.sqrt(1.0 - abs_dot_clamped**2), abs_dot_clamped)
        
        rot_loss = angle.mean()
        
        # Balance translation and rotation with alpha scaling
        alpha = 3.0  # Room corner to corner ≈ 3m (better balance for indoor scenes)
        loss = trans_loss * (1.0 / alpha) + rot_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Warm clip schedule: tighter clipping early, then gradually increase headroom
        if step < 300:
            clip_val = 1.0
        elif step < 500:
            clip_val = 2.0  # Intermediate value
        else:
            clip_val = 3.0  # More conservative than 5.0
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
        
        optimizer.step()
        
        losses.append(loss.item())
        
        # Check for NaN immediately
        if torch.isnan(loss):
            print(f"\nNaN detected at step {step}!")
            print(f"  trans_loss: {trans_loss.item() if not torch.isnan(trans_loss) else 'NaN'}")
            print(f"  rot_loss: {rot_loss.item() if not torch.isnan(rot_loss) else 'NaN'}")
            print(f"  dot_product range: [{dot_product.min().item():.6f}, {dot_product.max().item():.6f}]")
            print(f"  angle range: [{angle.min().item() if not torch.isnan(angle).any() else 'NaN'}, {angle.max().item() if not torch.isnan(angle).any() else 'NaN'}]")
            break
        
        # Every 50 steps, compute correlations and update plots
        if step % 50 == 0 and step > 0:  # Skip step 0 to avoid initial plotting overhead
            # Print gradient norm for monitoring
            print(f"Step {step:03d}: loss={loss.item():.4e}, "
                  f"trans_loss={trans_loss.item():.4e}, "
                  f"rot_loss={rot_loss.item():.4e}, "
                  f"grad_norm={grad_norm:.4e}")
            
            # Check for gradient explosion
            if grad_norm > 100:
                print(f"⚠️  WARNING: Large gradient norm detected: {grad_norm:.2e}")
                print("    → Possible numerical instability")
            
            with torch.no_grad():
                # Translation correlation
                pred_trans = pred_pose[0, :, :3]  # [seq_len, 3]
                gt_trans = gt_pose[0, :, :3]
                
                # Compute correlation for each axis
                for axis in range(3):
                    corr, lags = compute_cross_correlation(
                        pred_trans[:, axis], 
                        gt_trans[:, axis]
                    )
                    if axis == 0:  # Use X-axis as representative
                        trans_correlations.append((step, corr, lags))
                
                # Rotation correlation using actual rotation angles between poses
                # For each pose transition, compute the angle
                pred_quat_norm = pred_quat[0] / (torch.norm(pred_quat[0], dim=-1, keepdim=True) + 1e-8)
                gt_quat_norm = gt_quat[0] / (torch.norm(gt_quat[0], dim=-1, keepdim=True) + 1e-8)
                
                # Compute rotation angles for each transition
                pred_angles = []
                gt_angles = []
                for i in range(len(pred_quat_norm)):
                    # Angle between predicted and ground truth quaternions
                    dot_prod = torch.sum(pred_quat_norm[i] * gt_quat_norm[i]).clamp(-1, 1)
                    angle = 2.0 * torch.acos(dot_prod.abs())
                    pred_angles.append(angle)
                    
                    # Ground truth angle is 0 (perfect alignment)
                    gt_angles.append(torch.tensor(0.0))
                
                pred_angles = torch.stack(pred_angles)
                gt_angles = torch.stack(gt_angles)
                
                # Correlate the rotation error signal
                rot_corr, rot_lags = compute_cross_correlation(pred_angles, gt_angles)
                rot_correlations.append((step, rot_corr, rot_lags))
            
            # Update plots
            axes[0, 0].clear()
            axes[0, 0].semilogy(losses)
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
            
            # Translation correlation
            if trans_correlations:
                _, corr, lags = trans_correlations[-1]
                axes[0, 1].clear()
                axes[0, 1].plot(lags, corr)
                peak_idx = np.argmax(corr)
                axes[0, 1].axvline(lags[peak_idx], color='r', linestyle='--', 
                                 label=f'Peak at lag={lags[peak_idx]}')
                axes[0, 1].set_title('Translation Cross-Correlation')
                axes[0, 1].set_xlabel('Lag')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Rotation correlation
            if rot_correlations:
                _, corr, lags = rot_correlations[-1]
                axes[1, 0].clear()
                axes[1, 0].plot(lags, corr)
                peak_idx = np.argmax(corr)
                axes[1, 0].axvline(lags[peak_idx], color='r', linestyle='--',
                                 label=f'Peak at lag={lags[peak_idx]}')
                axes[1, 0].set_title('Rotation Cross-Correlation')
                axes[1, 0].set_xlabel('Lag')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # IMU magnitudes
            axes[1, 1].clear()
            # window_imu has shape [B, 110, 6]
            # Reshape to [10, 11, 6] to get per-frame averages
            imu_reshaped = window_imu[0].reshape(10, 11, 6)  # [10, 11, 6]
            acc_norm = torch.norm(imu_reshaped[:, :, :3], dim=-1).mean(dim=1).cpu()  # Average over 11 IMU samples per frame
            gyro_norm = torch.norm(imu_reshaped[:, :, 3:], dim=-1).mean(dim=1).cpu()
            axes[1, 1].plot(acc_norm, label='Acc norm (m/s²)')
            axes[1, 1].plot(gyro_norm, label='Gyro norm (rad/s)')
            axes[1, 1].set_title('IMU Signal Magnitudes (per frame)')
            axes[1, 1].set_xlabel('Frame transition')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.pause(0.001)  # Minimal pause to avoid slowing down training
    
    plt.ioff()
    
    # Final analysis
    final_loss = losses[-1]
    print(f"\nFinal loss after {max_steps} steps: {final_loss:.4e}")
    
    # Print detailed metrics
    if 'trans_loss' in locals() and 'rot_loss' in locals():
        print(f"Final trans_loss: {trans_loss.item():.4e} ({trans_loss.item()*100:.2f} cm)")
        print(f"Final rot_loss: {rot_loss.item():.4e} ({np.degrees(rot_loss.item()):.2f}°)")
    
    # Check for NaN
    if np.isnan(final_loss):
        print("\n❌ FAILURE: Loss became NaN!")
        print("Possible issues:")
        print("  - Learning rate too high")
        print("  - Numerical instability in quaternion operations")
        print("  - Check IMU data units and scaling")
        return losses, trans_correlations, rot_correlations
    
    if final_loss > 1e-3:
        print("\n⚠️  WARNING: Loss did not converge to near-zero!")
        print("Possible issues:")
        
        # Check correlation peaks
        if trans_correlations:
            _, corr, lags = trans_correlations[-1]
            peak_idx = np.argmax(corr)
            peak_lag = lags[peak_idx]
            if abs(peak_lag) > 0:
                print(f"  - Timestamp misalignment: peak at lag={peak_lag}")
                print(f"    → Try shifting IMU window by {-peak_lag} frames")
        
        # Check IMU magnitudes
        acc_mean = torch.norm(window_imu[..., :3], dim=-1).mean().item()
        gyro_mean = torch.norm(window_imu[..., 3:], dim=-1).mean().item()
        if acc_mean < 5.0:  # Less than 0.5g
            print(f"  - IMU scale issue: acc magnitude too small ({acc_mean:.3f}m/s² = {acc_mean/9.81:.3f}g)")
            print(f"    → Check units (should be m/s²)")
        
        if trans_loss.item() < rot_loss.item() * 0.1:
            print(f"  - Spatial misalignment: rotation loss dominates")
            print(f"    → Check camera-IMU extrinsics calibration")
    else:
        print("\n✅ SUCCESS: Model converged to near-zero loss!")
        print("The IMU and image streams appear to be properly aligned.")
    
    return losses, trans_correlations, rot_correlations


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Debug IMU-image alignment')
    parser.add_argument('--eval-mode', action='store_true', 
                       help='Run model in eval mode to disable BatchNorm/Dropout noise')
    args = parser.parse_args()
    
    # Initialize model
    model = VIFTFromScratch()
    
    # Load sequence mapping
    sequence_mapping = {}
    mapping_file = Path("./aria_processed/sequence_mapping.json")
    if mapping_file.exists():
        import json
        with open(mapping_file, 'r') as f:
            sequence_mapping = json.load(f)
        print(f"Loaded sequence mapping with {len(sequence_mapping)} entries")
    
    # Initialize data
    data_module = AriaRawDataModule(
        data_dir="./aria_processed",
        batch_size=1,
        num_workers=1
    )
    data_module.setup()
    
    # Get a single batch from training dataloader
    train_loader = data_module.train_dataloader()
    
    # Print first batch details
    print("\nGetting first batch from dataloader...")
    for batch in train_loader:
        print(f"Batch type: {type(batch)}")
        
        # Handle dictionary format from dataset
        if isinstance(batch, dict):
            images = batch['images']
            imus = batch['imu']
            gt_poses = batch['gt_poses']
            print(f"Batch keys: {list(batch.keys())}")
        else:
            # Try unpacking as tuple
            print(f"Batch length: {len(batch)}")
            if len(batch) == 3:
                images, imus, gt_poses = batch
            else:
                print(f"Unexpected batch format")
                break
            
        print(f"\nBatch details:")
        print(f"  Images shape: {images.shape}")
        print(f"  IMU shape: {imus.shape}")
        print(f"  GT poses shape: {gt_poses.shape}")
        print(f"\nImage stats:")
        print(f"  Min: {images.min():.3f}, Max: {images.max():.3f}, Mean: {images.mean():.3f}")
        print(f"\nIMU stats:")
        print(f"  Accelerometer (first 3 channels) - Min: {imus[..., :3].min():.3f}, Max: {imus[..., :3].max():.3f}, Mean: {imus[..., :3].mean():.3f}")
        print(f"  Gyroscope (last 3 channels) - Min: {imus[..., 3:].min():.3f}, Max: {imus[..., 3:].max():.3f}, Mean: {imus[..., 3:].mean():.3f}")
        print(f"\nPose stats:")
        print(f"  Translation (first 3) - Min: {gt_poses[..., :3].min():.3f}, Max: {gt_poses[..., :3].max():.3f}")
        print(f"  Rotation (last 4) - Min: {gt_poses[..., 3:].min():.3f}, Max: {gt_poses[..., 3:].max():.3f}")
        break
    
    # Find high-motion window
    window_img, window_imu, gt_pose = find_high_motion_window(train_loader, model, sequence_mapping)
    
    print(f"\nWindow shapes:")
    print(f"  Images: {window_img.shape}")
    print(f"  IMU: {window_imu.shape}")
    print(f"  GT poses: {gt_pose.shape}")
    
    # Reshape IMU if needed - model expects [B, 11, 10, 6] but we have [B, 110, 6]
    if window_imu.shape[1] == 110:
        # Reshape from [B, 110, 6] to [B, 10, 11, 6]
        window_imu_reshaped = window_imu.reshape(window_imu.shape[0], 10, 11, 6)
        # Transpose to [B, 11, 10, 6] if that's what model expects
        # Actually, let's keep it as [B, 110, 6] as that seems to be what the model expects
        print(f"  IMU reshaped would be: {window_imu_reshaped.shape}")
    
    # Run overfit test
    print("\nStarting single-sample overfit test...")
    losses, trans_corr, rot_corr = overfit_single_sample(
        model, window_img, window_imu, gt_pose, eval_mode=args.eval_mode
    )
    
    # Keep plot open
    plt.show()


if __name__ == "__main__":
    main()