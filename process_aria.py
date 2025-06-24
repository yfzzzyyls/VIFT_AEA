#!/usr/bin/env python3
"""
Process first 20 sequences from AriaEveryday dataset with real IMU data.
Extracts ALL raw IMU samples between consecutive frames for proper VIO processing.
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

# No fallback imports - we only use projectaria_tools


class AriaProcessor:
    def __init__(self, input_dir: str, output_dir: str, max_frames: int = -1, seq_id: str = None):
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
            # Evenly sample from the 20Hz grid
            indices = np.linspace(0, len(target_timestamps) - 1, self.max_frames, dtype=int)
            target_timestamps = target_timestamps[indices]
        
        print(f"{prefix}📊 Resampling to {len(target_timestamps)} frames at 20Hz...")
        
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
    
    def extract_real_imu_data(self, vrs_path: Path, poses: List[Dict]) -> Tuple[List[torch.Tensor], str]:
        """Extract ALL raw IMU data from VRS file between consecutive frames.
        Returns: (imu_data_raw, stream_id) where imu_data_raw is a list of variable-length tensors
        """
        prefix = f"[{self.seq_id}] " if self.seq_id else ""
        print(f"{prefix}📊 Extracting ALL raw IMU data from VRS...")
        
        # Use projectaria_tools to extract ALL IMU samples
        provider = data_provider.create_vrs_data_provider(str(vrs_path))
        if not provider:
            raise RuntimeError(f"Failed to create VRS data provider for {vrs_path}")
        
        # Get IMU stream
        imu_stream_id = provider.get_stream_id_from_label("imu-right")
        if imu_stream_id.is_valid():
            stream_id = "1202-1"
            print(f"{prefix}✅ Using right IMU stream (1202-1)")
        else:
            imu_stream_id = provider.get_stream_id_from_label("imu-left")
            if imu_stream_id.is_valid():
                stream_id = "1202-2"
                print(f"{prefix}✅ Using left IMU stream (1202-2)")
            else:
                raise RuntimeError("No valid IMU stream found")
        
        # Get number of IMU samples
        num_imu_data = provider.get_num_data(imu_stream_id)
        print(f"{prefix}📊 Found {num_imu_data} total IMU samples in VRS file")
        
        # Extract ALL IMU data by iterating through all indices
        all_accel_data = []
        all_gyro_data = []
        all_timestamps_ns = []
        
        print(f"{prefix}📊 Loading all IMU data...")
        for idx in range(num_imu_data):
            imu_data = provider.get_imu_data_by_index(imu_stream_id, idx)
            if imu_data:
                all_timestamps_ns.append(imu_data.capture_timestamp_ns)
                all_accel_data.append([imu_data.accel_msec2[0], imu_data.accel_msec2[1], imu_data.accel_msec2[2]])
                all_gyro_data.append([imu_data.gyro_radsec[0], imu_data.gyro_radsec[1], imu_data.gyro_radsec[2]])
        
        # Convert to numpy arrays
        timestamps_ns = np.array(all_timestamps_ns)
        accel_data = np.array(all_accel_data) / 1000.0  # Convert to m/s²
        gyro_data = np.array(all_gyro_data)
        
        # Convert timestamps to seconds
        timestamps_s = timestamps_ns / 1e9
        
        print(f"{prefix}📊 IMU data time range: {timestamps_s[0]:.3f}s to {timestamps_s[-1]:.3f}s")
        print(f"{prefix}📊 IMU sampling rate: ~{1.0/np.mean(np.diff(timestamps_s)):.0f} Hz")
        
        # Extract IMU data between consecutive frames
        imu_data_raw = []
        imu_counts = []
        
        for i in range(len(poses) - 1):
            if self.max_frames > 0 and i >= self.max_frames - 1:
                break
            
            t_start = poses[i]['timestamp']
            t_end = poses[i + 1]['timestamp']
            
            # Find ALL IMU samples in [t_start, t_end)
            mask = (timestamps_s >= t_start) & (timestamps_s < t_end)
            indices = np.where(mask)[0]
            
            if len(indices) == 0:
                # No IMU data in this interval - use single zero sample
                samples = torch.zeros((1, 6), dtype=torch.float32)
                print(f"{prefix}⚠️ No IMU data found in interval [{t_start:.3f}, {t_end:.3f})")
            else:
                # Extract ALL samples
                samples = torch.tensor(
                    np.hstack([accel_data[indices], gyro_data[indices]]),
                    dtype=torch.float32
                )
            
            imu_data_raw.append(samples)
            imu_counts.append(len(indices))
        
        # Print detailed statistics
        if imu_counts:
            avg_samples = np.mean(imu_counts)
            min_samples = np.min(imu_counts)
            max_samples = np.max(imu_counts)
            print(f"{prefix}✅ Extracted raw IMU data for {len(imu_data_raw)} frame intervals")
            print(f"{prefix}   IMU samples per interval: avg={avg_samples:.1f}, min={min_samples}, max={max_samples}")
            
            # Check if we're getting the expected ~50 samples
            if avg_samples < 40:
                print(f"{prefix}⚠️ WARNING: Expected ~50 IMU samples per frame interval, but got avg={avg_samples:.1f}")
                print(f"{prefix}   This might indicate the IMU data is being downsampled or filtered")
        
        return imu_data_raw, stream_id
    
    
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
            
            # Target 704x704 (2x2 binning from 1408x1408)
            target_area = 704 * 704
            aspect_ratio = original_width / original_height
            
            # Calculate dimensions that maintain aspect ratio
            target_height = int(np.sqrt(target_area / aspect_ratio))
            target_width = int(target_height * aspect_ratio)
            
            print(f"{prefix}📐 Resizing to {target_width}x{target_height} (aspect ratio: {aspect_ratio:.2f})")
            
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
                    
                    # Keep full resolution - no downsampling
                    if target_width == original_width and target_height == original_height:
                        # No resizing needed, keep original
                        img_resized = img_array
                    else:
                        # Only resize if target size is different
                        # For 1408x1408 to 512x512: first do 2x2 binning, then resize
                        if img_array.shape[0] == 1408 and img_array.shape[1] == 1408 and target_width == 512:
                            # Step 1: 2x2 binning to 704x704
                            if len(img_array.shape) == 3:  # Color image
                                img_binned = img_array.reshape(704, 2, 704, 2, 3).mean(axis=(1, 3))
                            else:  # Grayscale
                                img_binned = img_array.reshape(704, 2, 704, 2).mean(axis=(1, 3))
                            img_binned = img_binned.astype(np.uint8)
                            # Step 2: Resize to 512x512 using INTER_AREA
                            img_resized = cv2.resize(img_binned, (target_width, target_height), interpolation=cv2.INTER_AREA)
                        else:
                            # For other sizes, use INTER_AREA directly
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
        imu_data_raw, imu_stream_id = imu_result
        
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
        num_imu_intervals = min(num_frames - 1, len(imu_data_raw))
        
        # Adjust to have consistent data
        poses = poses[:num_imu_intervals + 1]
        visual_data = visual_data[:num_imu_intervals + 1]
        imu_data_raw = imu_data_raw[:num_imu_intervals]
        
        min_frames = num_imu_intervals + 1  # For metadata
        
        # Save data
        poses_file = seq_output_dir / "poses_quaternion.json"
        with open(poses_file, 'w') as f:
            json.dump(poses, f, indent=2)
        
        torch.save(visual_data.cpu(), seq_output_dir / "visual_data.pt")
        
        # Save raw IMU data as a list of variable-length tensors
        # Each element contains all IMU samples between consecutive frames
        torch.save(imu_data_raw, seq_output_dir / "imu_data.pt")
        
        # Save metadata
        # Calculate raw IMU statistics
        raw_imu_lengths = [len(samples) for samples in imu_data_raw]
        
        metadata = {
            'sequence_name': sequence_path.name,
            'sequence_id': sequence_id,
            'num_frames': int(min_frames),
            'num_imu_intervals': int(num_imu_intervals),
            'visual_shape': [int(x) for x in visual_data.shape],
            'imu_intervals': int(len(imu_data_raw)),
            'imu_samples_per_interval': {
                'mean': float(np.mean(raw_imu_lengths)),
                'min': int(np.min(raw_imu_lengths)),
                'max': int(np.max(raw_imu_lengths)),
                'std': float(np.std(raw_imu_lengths))
            },
            'slam_source': 'mps_slam_time_based_20hz',
            'imu_source': 'real_vrs_data_between_frames',
            'imu_stream_id': imu_stream_id,
            'imu_stream_name': 'right IMU' if imu_stream_id == '1202-1' else 'left IMU',
            'imu_frequency': 1000,
            'camera_frequency': 20,
            'rotation_format': 'quaternion_xyzw',
            'imu_format': 'raw_variable_length',
            'imu_channel_order': 'ax,ay,az,gx,gy,gz',
            'imu_units': 'm/s²,m/s²,m/s²,rad/s,rad/s,rad/s',
            'imu_note': 'IMU data contains ALL raw samples between consecutive frames (variable length per interval)',
            'imu_sampling': 'All IMU samples with timestamps in [t_start, t_end) interval'
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
    parser.add_argument('--max-frames', type=int, default=2000,
                       help='Max frames per sequence (default: 2000 frames = 100 seconds)')
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
        'max_frames_per_sequence': args.max_frames
    }
    
    with open(output_path / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Summary:")
    print(f"  - Processed {processed_count} sequences")
    if args.max_frames > 0:
        print(f"  - Each with up to {args.max_frames} frames")
    else:
        print(f"  - Using ALL frames (no subsampling)")
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