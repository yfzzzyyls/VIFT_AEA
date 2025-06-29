#!/usr/bin/env python3
"""
Process first 20 sequences from AriaEveryday dataset with real IMU data.
Combines shell script logic with processing script and adds IMU order assertions.
"""

import argparse
import json
import math
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
from functools import partial

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# Import Aria tools (required)
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

# Try alternative VRS reader
try:
    from aria_everyday_vision.aria_everyday_vrs import AriaVrsDataLoader
    ARIA_VRS_LOADER_AVAILABLE = True
except ImportError:
    ARIA_VRS_LOADER_AVAILABLE = False


class AriaProcessor:
    def __init__(self, input_dir: str, output_dir: str, max_frames: int = 1000, seq_id: str = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_frames = max_frames
        self.seq_id = seq_id  # For logging with sequence prefix
        self.output_dir.mkdir(exist_ok=True)
        
        # Always use CPU for extraction to avoid GPU memory issues in multiprocessing
        self.device = torch.device("cpu")
        if self.seq_id:
            print(f"[{self.seq_id}] 💻 Using CPU for extraction")
        else:
            print("💻 Using CPU for extraction")
    
    def extract_slam_poses(self, sequence_path: Path) -> Optional[List[Dict]]:
        """Extract SLAM trajectory from MPS results and resample to 20Hz."""
        prefix = f"[{self.seq_id}] " if self.seq_id else ""
        print(f"{prefix}📍 Extracting and resampling SLAM trajectory...")
        
        # Look for SLAM trajectories zip
        trajectory_zips = list(sequence_path.glob("*mps_slam_trajectories.zip"))
        if not trajectory_zips:
            print(f"{prefix}❌ No SLAM trajectories found")
            return None
        
        trajectory_zip = trajectory_zips[0]
        temp_dir = sequence_path / "temp_slam"
        temp_dir.mkdir(exist_ok=True)
        
        raw_poses = []
        try:
            with zipfile.ZipFile(trajectory_zip, 'r') as z:
                # Extract closed loop trajectory CSV
                csv_name = 'closed_loop_trajectory.csv'
                if csv_name in z.namelist():
                    z.extract(csv_name, temp_dir)
                    csv_file = temp_dir / csv_name
                    
                    # Read CSV file
                    import csv
                    with open(csv_file, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            try:
                                # Extract pose data from CSV
                                ts = float(row['tracking_timestamp_us']) / 1e6
                                tx = float(row['tx_world_device'])
                                ty = float(row['ty_world_device'])
                                tz = float(row['tz_world_device'])
                                qx = float(row['qx_world_device'])
                                qy = float(row['qy_world_device'])
                                qz = float(row['qz_world_device'])
                                qw = float(row['qw_world_device'])
                                
                                # Normalize quaternion
                                q_norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
                                if q_norm > 0:
                                    qx, qy, qz, qw = qx/q_norm, qy/q_norm, qz/q_norm, qw/q_norm
                                
                                raw_poses.append({
                                    'timestamp': ts,
                                    'translation': [tx, ty, tz],
                                    'quaternion': [qx, qy, qz, qw]
                                })
                            except (KeyError, ValueError) as e:
                                continue
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        if not raw_poses:
            print(f"{prefix}❌ No poses extracted from SLAM trajectory")
            return None
            
        print(f"{prefix}📊 Extracted {len(raw_poses)} raw SLAM poses")
        
        # Calculate raw SLAM statistics
        raw_positions = np.array([p['translation'] for p in raw_poses])
        raw_duration = raw_poses[-1]['timestamp'] - raw_poses[0]['timestamp']
        raw_distance = np.sum(np.linalg.norm(np.diff(raw_positions, axis=0), axis=1))
        print(f"{prefix}📊 Raw SLAM: {raw_duration:.1f}s duration, {raw_distance:.2f}m total movement")
        
        # Proper time-based resampling at 20Hz
        t_start = raw_poses[0]['timestamp']
        t_end = raw_poses[-1]['timestamp']
        
        # Create target timestamps at exactly 20Hz intervals
        target_timestamps = np.arange(t_start, t_end, 0.05)  # 50ms = 20Hz
        
        # Limit to max_frames if specified
        if self.max_frames > 0 and len(target_timestamps) > self.max_frames:
            # Take first max_frames consecutive frames for dense temporal sampling
            target_timestamps = target_timestamps[:self.max_frames]
        
        print(f"{prefix}📊 Taking first {len(target_timestamps)} consecutive frames at 20Hz...")
        
        # Extract all raw timestamps for efficient search
        raw_timestamps = np.array([p['timestamp'] for p in raw_poses])
        
        # Find closest raw pose for each target timestamp
        # Track used indices to avoid duplicates
        resampled_poses = []
        used_indices = set()
        
        for target_ts in target_timestamps:
            closest_idx = np.argmin(np.abs(raw_timestamps - target_ts))
            
            # If this index was already used, find the next closest
            if closest_idx in used_indices and len(raw_poses) > len(used_indices):
                # Get distances to all poses
                distances = np.abs(raw_timestamps - target_ts)
                # Sort indices by distance
                sorted_indices = np.argsort(distances)
                # Find first unused index
                for idx in sorted_indices:
                    if idx not in used_indices:
                        closest_idx = idx
                        break
            
            used_indices.add(closest_idx)
            resampled_poses.append(raw_poses[closest_idx])
        
        if resampled_poses:
            # Verify timestamps are strictly monotonic
            resampled_timestamps = [p['timestamp'] for p in resampled_poses]
            time_diffs = np.diff(resampled_timestamps)
            if np.any(time_diffs <= 0):
                print(f"{prefix}⚠️ WARNING: Non-monotonic timestamps detected after resampling")
                print(f"{prefix}   Found {np.sum(time_diffs <= 0)} non-increasing intervals")
                # Could raise an error here if strict monotonicity is required
                # raise ValueError("Resampled poses have non-monotonic timestamps")
            # Calculate resampled statistics
            resampled_positions = np.array([p['translation'] for p in resampled_poses])
            resampled_duration = resampled_poses[-1]['timestamp'] - resampled_poses[0]['timestamp']
            resampled_distance = np.sum(np.linalg.norm(np.diff(resampled_positions, axis=0), axis=1))
            
            print(f"{prefix}✅ Sampled {len(resampled_poses)} poses")
            print(f"{prefix}📊 Duration: {resampled_duration:.1f}s, Movement: {resampled_distance:.2f}m")
            
        return resampled_poses if resampled_poses else None
    
    def extract_real_imu_data(self, vrs_path: Path, poses: List[Dict]) -> Optional[Tuple[torch.Tensor, str]]:
        """Extract real IMU data from VRS file with channel order assertions.
        Returns: (imu_data, stream_id) or None
        """
        prefix = f"[{self.seq_id}] " if self.seq_id else ""
        print(f"{prefix}📊 Extracting real IMU data from VRS...")
        
        # Try AriaVrsDataLoader first if available
        if ARIA_VRS_LOADER_AVAILABLE:
            try:
                result = self.extract_real_imu_data_alternative(vrs_path, poses)
                if result is not None:
                    return result
            except Exception as e:
                print(f"{prefix}❌ AriaVrsDataLoader failed: {e}")
        
        # Try original method
        try:
            return self.extract_real_imu_data_original(vrs_path, poses)
        except Exception as e:
            print(f"{prefix}❌ Real IMU extraction failed: {e}")
            raise RuntimeError(f"Failed to extract real IMU data: {e}")
    
    def assert_imu_channel_order(self, imu_data: torch.Tensor, source: str):
        """Assert that IMU data has correct channel order [accel, gyro]."""
        prefix = f"[{self.seq_id}] " if self.seq_id else ""
        
        # Calculate magnitudes
        acc_norm = torch.norm(imu_data[..., :3], dim=-1).mean().item()
        gyro_norm = torch.norm(imu_data[..., 3:], dim=-1).mean().item()
        
        print(f"{prefix}📊 IMU sanity check ({source}): |acc|={acc_norm:.2f} m/s² ({acc_norm/9.81:.2f}g), |gyro|={gyro_norm:.2f} rad/s")
        
        # More refined checks to handle edge cases
        if acc_norm < 5.0:
            print(f"{prefix}❌ ERROR: Accelerometer magnitude too low: {acc_norm:.2f} m/s²")
            print(f"{prefix}   Expected ~9.8 m/s² (gravity). Channel order is likely wrong!")
            raise ValueError(f"IMU channel order error: |acc|={acc_norm:.2f} m/s², expected ~9.8")
        
        if gyro_norm > 10.0:
            print(f"{prefix}❌ ERROR: Gyroscope magnitude too high: {gyro_norm:.2f} rad/s")
            print(f"{prefix}   Expected <5 rad/s for indoor motion. Channel order is likely wrong!")
            raise ValueError(f"IMU channel order error: |gyro|={gyro_norm:.2f} rad/s, expected <5")
        
        # Warnings for unusual but potentially valid cases
        # Relaxed threshold for stationary/tilted scenarios
        if acc_norm < 7.0:
            print(f"{prefix}⚠️ WARNING: Low accelerometer magnitude: {acc_norm:.2f} m/s²")
            print(f"{prefix}   Device may be stationary on a tilted surface")
        elif acc_norm > 15.0:
            print(f"{prefix}⚠️ WARNING: High accelerometer magnitude: {acc_norm:.2f} m/s²")
            print(f"{prefix}   This might indicate rapid motion or impacts")
            
        if gyro_norm > 5.0:
            print(f"{prefix}⚠️ WARNING: High gyroscope magnitude: {gyro_norm:.2f} rad/s")
            print(f"{prefix}   This might indicate rapid rotation")
    
    def extract_real_imu_data_alternative(self, vrs_path: Path, poses: List[Dict]) -> Optional[Tuple[torch.Tensor, str]]:
        """Extract real IMU data using AriaVrsDataLoader."""
        prefix = f"[{self.seq_id}] " if self.seq_id else ""
        
        try:
            # Use AriaVrsDataLoader for simpler IMU extraction
            dl = AriaVrsDataLoader(str(vrs_path))
            
            # Try to load right IMU first (≈1 kHz)
            imu_timestamps_ns = None
            accel_data = None
            gyro_data = None
            stream_id = None
            
            try:
                imu_timestamps_ns, accel_data, gyro_data = dl.load_imu_stream("imu-right")
                stream_id = "1202-1"  # Right IMU
                print(f"{prefix}✅ Using right IMU stream (1202-1)")
            except:
                try:
                    # Fall back to left IMU (≈800 Hz)
                    imu_timestamps_ns, accel_data, gyro_data = dl.load_imu_stream("imu-left")
                    stream_id = "1202-2"  # Left IMU
                    print(f"{prefix}✅ Using left IMU stream (1202-2)")
                except Exception as e:
                    print(f"{prefix}❌ No IMU streams found: {e}")
                    return None
            
            if imu_timestamps_ns is None or len(imu_timestamps_ns) == 0:
                print(f"{prefix}❌ No IMU data found")
                return None
            
            print(f"{prefix}📊 Found {len(imu_timestamps_ns)} IMU samples")
            
            # Convert timestamps to seconds
            imu_timestamps_s = imu_timestamps_ns / 1e9
            
            # Extract IMU data BETWEEN consecutive frames (proper VIO approach)
            # Save ALL IMU samples between frames
            imu_data = []
            imu_sample_counts = []  # Track number of samples per interval
            
            # Process each interval between consecutive frames
            for i in range(len(poses) - 1):
                if self.max_frames > 0 and i >= self.max_frames - 1:
                    break
                
                # Get timestamps for consecutive frames
                t_start = poses[i]['timestamp']
                t_end = poses[i + 1]['timestamp']
                
                # Find all IMU samples in this interval [t_start, t_end)
                mask = (imu_timestamps_s >= t_start) & (imu_timestamps_s < t_end)
                interval_indices = np.where(mask)[0]
                
                if len(interval_indices) == 0:
                    print(f"{prefix}⚠️ No IMU data between frames {i} and {i+1}")
                    # Create empty tensor for this interval
                    frame_samples = torch.zeros((0, 6), dtype=torch.float32)
                else:
                    # NEW: Extract ALL IMU samples in this interval
                    frame_samples = []
                    for idx in interval_indices:
                        # CORRECT ORDER: [accel, gyro]
                        sample_data = torch.tensor([
                            accel_data[idx, 0], accel_data[idx, 1], accel_data[idx, 2],
                            gyro_data[idx, 0], gyro_data[idx, 1], gyro_data[idx, 2]
                        ], dtype=torch.float32)
                        frame_samples.append(sample_data)
                    
                    frame_samples = torch.stack(frame_samples) if frame_samples else torch.zeros((0, 6), dtype=torch.float32)
                
                imu_data.append(frame_samples)
                imu_sample_counts.append(len(frame_samples))
            
            if imu_data:
                # NEW: Can't stack variable-length sequences, return as list
                # Print statistics about IMU samples per interval
                avg_samples = np.mean(imu_sample_counts)
                min_samples = np.min(imu_sample_counts)
                max_samples = np.max(imu_sample_counts)
                print(f"{prefix}✅ Extracted real IMU data for {len(imu_data)} intervals")
                print(f"{prefix}📊 IMU samples per interval: avg={avg_samples:.1f}, min={min_samples}, max={max_samples}")
                
                # Check channel order on first non-empty interval
                for data in imu_data:
                    if len(data) > 0:
                        self.assert_imu_channel_order(data.unsqueeze(0), "AriaVrsDataLoader")
                        break
                
                return imu_data, stream_id
            else:
                print(f"{prefix}❌ No IMU data extracted")
            
        except Exception as e:
            print(f"{prefix}❌ Error with AriaVrsDataLoader: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def extract_real_imu_data_original(self, vrs_path: Path, poses: List[Dict]) -> Optional[Tuple[torch.Tensor, str]]:
        """Extract real IMU data from VRS file using original API."""
        prefix = f"[{self.seq_id}] " if self.seq_id else ""
        print(f"{prefix}📊 Using original projectaria_tools method...")
        try:
            provider = data_provider.create_vrs_data_provider(str(vrs_path))
            if not provider:
                print(f"{prefix}❌ Failed to open VRS file")
                return None
            
            # Get IMU stream - try right IMU first, then left
            imu_stream = None
            used_stream_id = None
            stream_ids = ["1202-1", "1202-2"]  # Right and left IMU
            
            for stream_id_str in stream_ids:
                try:
                    imu_stream_id = StreamId(stream_id_str)
                    # Try to get configuration to check if stream exists
                    config = provider.get_imu_configuration(imu_stream_id)
                    if config:
                        imu_stream = imu_stream_id
                        used_stream_id = stream_id_str
                        print(f"{prefix}✅ Using IMU stream: {stream_id_str}")
                        print(f"{prefix}📊 IMU rate: {config.nominal_rate_hz} Hz")
                        break
                except:
                    continue
            
            if not imu_stream:
                print(f"{prefix}❌ No IMU streams found")
                return None
            
            # Extract IMU data BETWEEN consecutive frames (proper VIO approach)
            # Extract ALL IMU samples between frames
            imu_data = []
            imu_sample_counts = []
            
            print(f"{prefix}📊 Extracting ALL IMU samples between {len(poses)-1} frame pairs...")
            
            # First, get all IMU data in the time range
            start_time_ns = int(poses[0]['timestamp'] * 1e9)
            end_time_ns = int(poses[-1]['timestamp'] * 1e9)
            
            # Collect all IMU samples in the sequence timespan
            all_imu_samples = []
            all_imu_timestamps = []
            
            # Query IMU data in chunks to avoid memory issues
            chunk_duration_ns = int(1e9)  # 1 second chunks
            current_time_ns = start_time_ns
            
            while current_time_ns < end_time_ns:
                try:
                    imu_sample = provider.get_imu_data_by_time_ns(
                        imu_stream,
                        current_time_ns,
                        TimeDomain.DEVICE_TIME,
                        TimeQueryOptions.CLOSEST
                    )
                    if imu_sample:
                        all_imu_samples.append(imu_sample)
                        all_imu_timestamps.append(current_time_ns / 1e9)  # Convert to seconds
                except:
                    pass
                current_time_ns += int(1e6)  # 1ms steps for ~1000Hz IMU
            
            print(f"{prefix}📊 Collected {len(all_imu_samples)} total IMU samples")
            
            # Now extract samples for each frame interval
            for i in range(len(poses) - 1):
                if self.max_frames > 0 and i >= self.max_frames - 1:
                    break
                
                if i % 100 == 0:
                    print(f"{prefix}  Progress: {i}/{len(poses)-1} intervals...")
                
                # Get timestamps for consecutive frames
                t_start = poses[i]['timestamp']
                t_end = poses[i + 1]['timestamp']
                
                # Find all IMU samples in this interval [t_start, t_end)
                frame_samples = []
                for j, ts in enumerate(all_imu_timestamps):
                    if t_start <= ts < t_end:
                        imu_sample = all_imu_samples[j]
                        # CORRECT FORMAT: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
                        sample_data = torch.tensor([
                            imu_sample.accel_msec2[0],
                            imu_sample.accel_msec2[1],
                            imu_sample.accel_msec2[2],
                            imu_sample.gyro_radsec[0],
                            imu_sample.gyro_radsec[1],
                            imu_sample.gyro_radsec[2]
                        ], dtype=torch.float32)
                        frame_samples.append(sample_data)
                
                if len(frame_samples) == 0:
                    # No samples in this interval, create empty tensor
                    frame_samples_tensor = torch.zeros((0, 6), dtype=torch.float32)
                else:
                    frame_samples_tensor = torch.stack(frame_samples)
                
                imu_data.append(frame_samples_tensor)
                imu_sample_counts.append(len(frame_samples))
            
            if imu_data:
                # NEW: Return as list of variable-length tensors
                avg_samples = np.mean(imu_sample_counts) if imu_sample_counts else 0
                min_samples = np.min(imu_sample_counts) if imu_sample_counts else 0
                max_samples = np.max(imu_sample_counts) if imu_sample_counts else 0
                print(f"{prefix}✅ Extracted real IMU data for {len(imu_data)} intervals")
                print(f"{prefix}📊 IMU samples per interval: avg={avg_samples:.1f}, min={min_samples}, max={max_samples}")
                
                # Check channel order on first non-empty interval
                for data in imu_data:
                    if len(data) > 0:
                        self.assert_imu_channel_order(data.unsqueeze(0), "projectaria_tools")
                        break
                
                return imu_data, used_stream_id
            
        except Exception as e:
            print(f"{prefix}❌ Error extracting IMU: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def extract_rgb_frames(self, vrs_path: Path, poses: List[Dict], output_video_path: Path) -> Optional[torch.Tensor]:
        """Extract RGB frames from VRS file."""
        prefix = f"[{self.seq_id}] " if self.seq_id else ""
        print(f"{prefix}🎥 Extracting RGB frames from VRS...")
        
        try:
            provider = data_provider.create_vrs_data_provider(str(vrs_path))
            if not provider:
                print(f"{prefix}❌ Failed to open VRS file")
                return None
            
            # Get RGB camera stream
            rgb_stream_id = StreamId("214-1")
            config = provider.get_image_configuration(rgb_stream_id)
            
            if not config:
                print(f"{prefix}❌ No RGB camera stream found")
                return None
            
            print(f"{prefix}📷 RGB camera: {config.image_width}x{config.image_height}")
            
            # Calculate target resolution maintaining aspect ratio
            # Original Aria resolution is typically 1408x1408 (square)
            # We want to downsample while maintaining aspect ratio
            original_width = config.image_width
            original_height = config.image_height
            
            # Fixed target resolution 704x704
            target_width = 704
            target_height = 704
            
            aspect_ratio = original_width / original_height
            print(f"{prefix}📐 Resizing to {target_width}x{target_height} (original aspect ratio: {aspect_ratio:.2f})")
            
            # Extract frames aligned with poses
            frames = []
            
            print(f"{prefix}📊 Extracting {len(poses)} RGB frames...")
            for i, pose in enumerate(poses):
                if self.max_frames > 0 and i >= self.max_frames:
                    break
                
                if i % 100 == 0:
                    print(f"{prefix}  Progress: {i}/{len(poses)} frames...")
                
                frame_timestamp_ns = int(pose['timestamp'] * 1e9)
                
                # Get frame data
                image_data = provider.get_image_data_by_time_ns(
                    rgb_stream_id,
                    frame_timestamp_ns,
                    TimeDomain.DEVICE_TIME,
                    TimeQueryOptions.CLOSEST
                )
                
                if image_data and len(image_data) > 0:
                    # Get the first image if multiple
                    img_array = image_data[0].to_numpy_array()
                    
                    # Resize to target resolution with INTER_AREA for better quality
                    img_resized = cv2.resize(img_array, (target_width, target_height), interpolation=cv2.INTER_AREA)
                    
                    # Keep as uint8 to save memory, convert to float later if needed
                    # This follows PyTorch's recommendation for preprocessing pipelines
                    img_tensor = torch.from_numpy(img_resized)
                    
                    # Add to list
                    frames.append(img_tensor)
                else:
                    # Pad with zeros if no frame (uint8)
                    frames.append(torch.zeros(target_height, target_width, 3, dtype=torch.uint8))
            
            if frames:
                print(f"{prefix}✅ Extracted {len(frames)} RGB frames")
                
                # Stack and permute to [N, C, H, W]
                # Convert to float and normalize here for efficiency
                visual_data = torch.stack(frames).permute(0, 3, 1, 2).float() / 255.0
                
                # Save as video (optional)
                if output_video_path:
                    self.save_video(visual_data, output_video_path)
                
                return visual_data.to(self.device)
            
        except Exception as e:
            print(f"{prefix}❌ Error extracting RGB frames: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def save_video(self, frames: torch.Tensor, output_path: Path, fps: int = 20):
        """Save frames as video."""
        prefix = f"[{self.seq_id}] " if self.seq_id else ""
        try:
            # Convert to numpy (already normalized float data)
            # Denormalize back to uint8 for video
            frames_np = (frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            
            # Setup video writer
            height, width = frames_np.shape[1:3]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            # Write frames
            for frame in frames_np:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"{prefix}✅ Saved video to {output_path}")
            
        except Exception as e:
            print(f"{prefix}⚠️ Failed to save video: {e}")
    
    def process_sequence(self, sequence_path: Path, sequence_id: str) -> bool:
        """Process a single sequence with real IMU data."""
        prefix = f"[{self.seq_id}] " if self.seq_id else ""
        print(f"\n{prefix}🔄 Processing sequence: {sequence_path.name}")
        
        # Check for VRS file - look for the main recording VRS file
        vrs_files = list(sequence_path.glob("*main_recording.vrs"))
        if not vrs_files:
            print(f"{prefix}❌ No main_recording.vrs file found in {sequence_path}")
            return False
        vrs_path = vrs_files[0]  # Use the first VRS file found
        print(f"{prefix}📹 Found VRS file: {vrs_path.name}")
        
        # Extract SLAM poses (now properly resampled)
        poses = self.extract_slam_poses(sequence_path)
        if not poses:
            return False
        
        # Extract real IMU data (with assertions)
        imu_result = self.extract_real_imu_data(vrs_path, poses)
        if imu_result is None:
            return False
        imu_data, imu_stream_id = imu_result
        
        # Extract RGB frames
        seq_output_dir = self.output_dir / sequence_id
        seq_output_dir.mkdir(exist_ok=True)
        
        video_path = seq_output_dir / "rgb_video.mp4"
        visual_data = self.extract_rgb_frames(vrs_path, poses, video_path)
        if visual_data is None:
            return False
        
        # Ensure matching lengths
        # Note: IMU data has N-1 intervals for N frames
        num_frames = min(len(poses), visual_data.shape[0])
        num_imu_intervals = min(num_frames - 1, len(imu_data))  # imu_data is now a list
        
        # Adjust to have consistent data
        poses = poses[:num_imu_intervals + 1]
        visual_data = visual_data[:num_imu_intervals + 1]
        imu_data = imu_data[:num_imu_intervals]  # Slice the list
        
        min_frames = num_imu_intervals + 1  # For metadata
        
        # Save data
        poses_file = seq_output_dir / "poses_quaternion.json"
        with open(poses_file, 'w') as f:
            json.dump(poses, f, indent=2)
        
        torch.save(visual_data.cpu(), seq_output_dir / "visual_data.pt")
        # Save as list of tensors for variable-length IMU data
        torch.save(imu_data, seq_output_dir / "imu_data.pt")
        
        # Save metadata
        metadata = {
            'sequence_name': sequence_path.name,
            'sequence_id': sequence_id,
            'num_frames': min_frames,
            'num_imu_intervals': num_imu_intervals,
            'visual_shape': list(visual_data.shape),
            'imu_shape': f"List of {len(imu_data)} variable-length tensors",
            'imu_samples_per_interval': [len(interval) for interval in imu_data],
            'slam_source': 'mps_slam_time_based_20hz',
            'imu_source': 'real_vrs_data_between_frames',
            'imu_stream_id': imu_stream_id,
            'imu_stream_name': 'right IMU' if imu_stream_id == '1202-1' else 'left IMU',
            'imu_frequency': 1000,
            'camera_frequency': 20,
            'rotation_format': 'quaternion_xyzw',
            'imu_format': 'variable_length_between_frames',
            'imu_channel_order': 'ax,ay,az,gx,gy,gz',
            'imu_units': 'm/s²,m/s²,m/s²,rad/s,rad/s,rad/s',
            'imu_note': 'IMU data contains ALL samples between consecutive frames (variable length per interval)',
            'imu_sampling': 'All IMU samples in [t_start, t_end) interval are preserved',
            'frame_sampling': 'First 1000 consecutive frames at 20Hz for dense temporal sampling'
        }
        
        with open(seq_output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"{prefix}✅ Successfully processed {sequence_path.name}: {min_frames} frames")
        return True


def process_sequence_wrapper(args):
    """Wrapper for multiprocessing."""
    seq_path, seq_id, processor_args = args
    # Disable CUDA for worker processes to avoid GPU memory issues
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    processor = AriaProcessor(**processor_args, seq_id=seq_id)
    return processor.process_sequence(seq_path, seq_id)


def main():
    parser = argparse.ArgumentParser(description='Process first 20 AriaEveryday sequences with real IMU data')
    parser.add_argument('--input-dir', type=str, default='/mnt/ssd_ext/incSeg-data/aria_everyday',
                       help='Path to raw AriaEveryday dataset')
    parser.add_argument('--output-dir', type=str, default='./aria_processed',
                       help='Output directory')
    parser.add_argument('--max-frames', type=int, default=3000,
                       help='Max frames per sequence (default: 1000, first consecutive frames)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Get all sequences in the input directory
    input_path = Path(args.input_dir)
    all_sequences = []
    for d in sorted(input_path.iterdir()):
        if d.is_dir():
            # Look for any .vrs file ending with "main_recording.vrs"
            vrs_files = list(d.glob("*main_recording.vrs"))
            if vrs_files:
                all_sequences.append(d)
    
    # Take first 20 sequences
    sequences_to_process = all_sequences[:20]
    
    print(f"🎯 Processing First 20 AriaEveryday Sequences")
    print(f"📁 Input: {args.input_dir}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🔢 Found {len(all_sequences)} total sequences")
    print(f"🎬 Processing first 20 sequences")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create sequence mapping
    sequence_mapping = {}
    print("\n📋 Sequences to process:")
    for i, seq_path in enumerate(sequences_to_process):
        seq_id = f"{i:03d}"
        sequence_mapping[seq_id] = seq_path.name
        if seq_path.name == "loc2_script3_seq3_rec1":
            print(f"  {seq_id}: {seq_path.name} ⭐ (standing from couch)")
        else:
            print(f"  {seq_id}: {seq_path.name}")
    
    # Save sequence mapping
    with open(output_path / "sequence_mapping.json", 'w') as f:
        json.dump(sequence_mapping, f, indent=2)
    print(f"\n✅ Saved sequence mapping to {output_path / 'sequence_mapping.json'}")
    
    # Process sequences
    if args.num_workers > 1:
        print(f"\n🚀 Processing with {args.num_workers} workers...")
        
        # Ensure no sequences are dropped due to integer division
        # Use ceiling division to distribute work evenly
        num_sequences = len(sequences_to_process)
        chunk_size = math.ceil(num_sequences / args.num_workers)
        
        print(f"📊 Distributing {num_sequences} sequences across {args.num_workers} workers")
        print(f"   Each worker will process up to {chunk_size} sequences")
        
        # Prepare arguments for multiprocessing
        processor_args = {
            'input_dir': args.input_dir,
            'output_dir': args.output_dir,
            'max_frames': args.max_frames
        }
        
        process_args = [(sequences_to_process[i], f"{i:03d}", processor_args) 
                       for i in range(len(sequences_to_process))]
        
        # Process in parallel
        with mp.Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_sequence_wrapper, process_args),
                total=len(process_args),
                desc="Processing sequences"
            ))
        
        processed_count = sum(results)
    else:
        print(f"\n🚀 Processing sequentially...")
        processed_count = 0
        
        for i, seq_path in enumerate(tqdm(sequences_to_process, desc="Processing")):
            seq_id = f"{i:03d}"
            processor = AriaProcessor(args.input_dir, args.output_dir, args.max_frames, seq_id=seq_id)
            if processor.process_sequence(seq_path, seq_id):
                processed_count += 1
    
    print(f"\n🎉 Processing Complete!")
    print(f"✅ Successfully processed: {processed_count}/{len(sequences_to_process)} sequences")
    
    # Save summary
    summary = {
        'dataset': 'AriaEveryday_Raw',
        'total_sequences': len(all_sequences),
        'processed_sequences': processed_count,
        'sequences_processed': list(sequence_mapping.values()),
        'imu_type': 'real_sensor_data',
        'imu_channel_order': 'ax,ay,az,gx,gy,gz',
        'slam_resampling': 'time_based_20hz',
        'max_frames_per_sequence': args.max_frames,
        'frame_sampling_strategy': 'first_consecutive_frames'
    }
    
    with open(output_path / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Summary:")
    print(f"  - Processed {processed_count} sequences")
    print(f"  - Each with up to {args.max_frames} consecutive frames (50 seconds at 20Hz)")
    print(f"  - IMU data format: [ax,ay,az,gx,gy,gz]")
    print(f"  - IMU extracted between consecutive frames")
    print(f"\n📁 Output saved to: {output_path}")
    
    # Verify sequence 005 is included
    if "005" in sequence_mapping:
        print(f"\n✅ Sequence 005 ({sequence_mapping['005']}) is included in the processed data")
    else:
        raise ValueError("Sequence 005 is missing from the processed data!")
    
    print("\n🎯 Next steps:")
    print("1. Run the debug script to verify IMU-image alignment:")
    print("   python debug_imu_image_alignment.py")
    print("2. Generate latent features:")
    print("   python generate_all_pretrained_latents_between_frames.py")
    print("3. Train the model with properly aligned data")


if __name__ == "__main__":
    main()