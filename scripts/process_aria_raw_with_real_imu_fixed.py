#!/usr/bin/env python3
"""
Process raw AriaEveryday dataset with real IMU data and proper SLAM resampling.
Fixed version that:
1. Resamples SLAM poses from ~1000Hz to 20Hz camera rate
2. Extracts real IMU data from VRS files
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


class AriaRawProcessor:
    def __init__(self, input_dir: str, output_dir: str, max_frames: int = 1000):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_frames = max_frames  # Default 1000 frames
        self.output_dir.mkdir(exist_ok=True)
        
        # Check device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("🚀 Using CUDA GPU")
        else:
            self.device = torch.device("cpu")
            print("💻 Using CPU")
    
    def extract_slam_poses(self, sequence_path: Path) -> Optional[List[Dict]]:
        """Extract SLAM trajectory from MPS results and resample to 20Hz."""
        print(f"📍 Extracting and resampling SLAM trajectory...")
        
        # Look for SLAM trajectories zip
        trajectory_zips = list(sequence_path.glob("*mps_slam_trajectories.zip"))
        if not trajectory_zips:
            print("❌ No SLAM trajectories found")
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
            print("❌ No poses extracted from SLAM trajectory")
            return None
            
        print(f"📊 Extracted {len(raw_poses)} raw SLAM poses")
        
        # Calculate raw SLAM statistics
        raw_positions = np.array([p['translation'] for p in raw_poses])
        raw_duration = raw_poses[-1]['timestamp'] - raw_poses[0]['timestamp']
        raw_distance = np.sum(np.linalg.norm(np.diff(raw_positions, axis=0), axis=1))
        print(f"📊 Raw SLAM: {raw_duration:.1f}s duration, {raw_distance:.2f}m total movement")
        
        # Evenly sample frames across the trajectory
        total_frames = len(raw_poses)
        
        if self.max_frames > 0 and self.max_frames < total_frames:
            # Evenly sample max_frames from the trajectory
            print(f"📊 Evenly sampling {self.max_frames} frames from {total_frames} total poses...")
            indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
            resampled_poses = [raw_poses[i] for i in indices]
        else:
            # Use all frames
            resampled_poses = raw_poses
            print(f"📊 Using all {len(resampled_poses)} frames")
        
        if resampled_poses:
            # Calculate resampled statistics
            resampled_positions = np.array([p['translation'] for p in resampled_poses])
            resampled_duration = resampled_poses[-1]['timestamp'] - resampled_poses[0]['timestamp']
            resampled_distance = np.sum(np.linalg.norm(np.diff(resampled_positions, axis=0), axis=1))
            
            print(f"✅ Sampled {len(resampled_poses)} poses")
            print(f"📊 Duration: {resampled_duration:.1f}s, Movement: {resampled_distance:.2f}m")
            
        return resampled_poses if resampled_poses else None
    
    def extract_real_imu_data(self, vrs_path: Path, poses: List[Dict]) -> Optional[torch.Tensor]:
        """Extract real IMU data from VRS file - NO FALLBACKS."""
        print(f"📊 Extracting real IMU data from VRS...")
        
        # Try AriaVrsDataLoader first if available
        if ARIA_VRS_LOADER_AVAILABLE:
            try:
                result = self.extract_real_imu_data_alternative(vrs_path, poses)
                if result is not None:
                    return result
            except Exception as e:
                print(f"❌ AriaVrsDataLoader failed: {e}")
        
        # Try original method
        try:
            return self.extract_real_imu_data_original(vrs_path, poses)
        except Exception as e:
            print(f"❌ Real IMU extraction failed: {e}")
            raise RuntimeError(f"Failed to extract real IMU data: {e}")
    
    def extract_real_imu_data_alternative(self, vrs_path: Path, poses: List[Dict]) -> Optional[torch.Tensor]:
        """Extract real IMU data using AriaVrsDataLoader."""
        
        try:
            # Use AriaVrsDataLoader for simpler IMU extraction
            dl = AriaVrsDataLoader(str(vrs_path))
            
            # Try to load right IMU first (≈1 kHz)
            imu_timestamps_ns = None
            accel_data = None
            gyro_data = None
            
            try:
                imu_timestamps_ns, accel_data, gyro_data = dl.load_imu_stream("imu-right")
                print("✅ Using right IMU stream")
            except:
                try:
                    # Fall back to left IMU (≈800 Hz)
                    imu_timestamps_ns, accel_data, gyro_data = dl.load_imu_stream("imu-left")
                    print("✅ Using left IMU stream")
                except Exception as e:
                    print(f"❌ No IMU streams found: {e}")
                    return None
            
            if imu_timestamps_ns is None or len(imu_timestamps_ns) == 0:
                print("❌ No IMU data found")
                return None
            
            print(f"📊 Found {len(imu_timestamps_ns)} IMU samples")
            
            # Convert timestamps to seconds
            imu_timestamps_s = imu_timestamps_ns / 1e9
            
            # Extract IMU data aligned with camera frames
            imu_data = []
            samples_per_frame = 10  # Target samples per frame
            
            for i, pose in enumerate(poses):
                if self.max_frames > 0 and i >= self.max_frames:
                    break
                
                frame_timestamp = pose['timestamp']
                frame_samples = []
                
                # Get 10 IMU samples around frame timestamp
                # Sample window: -25ms to +25ms around frame
                for j in range(samples_per_frame):
                    # Evenly distribute samples across 50ms window
                    sample_offset = (j - 4.5) * 0.005  # -22.5ms to +22.5ms
                    target_time = frame_timestamp + sample_offset
                    
                    # Find closest IMU sample
                    idx = np.argmin(np.abs(imu_timestamps_s - target_time))
                    
                    if abs(imu_timestamps_s[idx] - target_time) < 0.01:  # Within 10ms
                        sample_data = torch.tensor([
                            gyro_data[idx, 0], gyro_data[idx, 1], gyro_data[idx, 2],
                            accel_data[idx, 0], accel_data[idx, 1], accel_data[idx, 2]
                        ], dtype=torch.float64)
                        frame_samples.append(sample_data)
                    else:
                        # Pad with zeros if no close sample
                        frame_samples.append(torch.zeros(6, dtype=torch.float64))
                
                if len(frame_samples) == samples_per_frame:
                    imu_data.append(torch.stack(frame_samples))
            
            if imu_data:
                print(f"✅ Extracted real IMU data for {len(imu_data)} frames")
                return torch.stack(imu_data).to(self.device)
            else:
                print("❌ No IMU data extracted")
            
        except Exception as e:
            print(f"❌ Error with AriaVrsDataLoader: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def extract_real_imu_data_original(self, vrs_path: Path, poses: List[Dict]) -> Optional[torch.Tensor]:
        """Extract real IMU data from VRS file using original API."""
        print("📊 Using original projectaria_tools method...")
        try:
            provider = data_provider.create_vrs_data_provider(str(vrs_path))
            if not provider:
                print("❌ Failed to open VRS file")
                return None
            
            # Get IMU stream - try right IMU first, then left
            imu_stream = None
            stream_ids = ["1202-1", "1202-2"]  # Right and left IMU
            
            for stream_id_str in stream_ids:
                try:
                    imu_stream_id = StreamId(stream_id_str)
                    # Try to get configuration to check if stream exists
                    config = provider.get_imu_configuration(imu_stream_id)
                    if config:
                        imu_stream = imu_stream_id
                        print(f"✅ Using IMU stream: {stream_id_str}")
                        print(f"📊 IMU rate: {config.nominal_rate_hz} Hz")
                        break
                except:
                    continue
            
            if not imu_stream:
                print("❌ No IMU streams found")
                return None
            
            # Extract IMU data aligned with camera frames
            imu_data = []
            samples_per_frame = 10  # Downsampled from 50 to 10 samples per frame
            
            print(f"📊 Extracting IMU for {len(poses)} poses...")
            for i, pose in enumerate(poses):
                if self.max_frames > 0 and i >= self.max_frames:
                    break
                
                if i % 1000 == 0:
                    print(f"  Progress: {i}/{len(poses)} poses...")
                
                frame_timestamp = pose['timestamp']
                frame_samples = []
                
                # Get 10 IMU samples evenly distributed from 50 samples (every 5th sample)
                # Original: 50 samples over 50ms window
                # Now: 10 samples, taking every 5th sample
                for j in range(samples_per_frame):
                    # Sample indices: 0, 5, 10, 15, 20, 25, 30, 35, 40, 45
                    sample_idx = j * 5
                    # Sample time: 25ms before to 25ms after frame
                    sample_offset = (sample_idx - 25) * 0.001  # 25 is half of 50
                    sample_time = frame_timestamp + sample_offset
                    sample_time_ns = int(sample_time * 1e9)
                    
                    # Get IMU data
                    try:
                        imu_sample = provider.get_imu_data_by_time_ns(
                            imu_stream,
                            sample_time_ns,
                            TimeDomain.DEVICE_TIME,
                            TimeQueryOptions.CLOSEST
                        )
                    except Exception as e:
                        print(f"⚠️ Error getting IMU sample: {e}")
                        imu_sample = None
                    
                    if imu_sample:
                        # Format: [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]
                        sample_data = torch.tensor([
                            imu_sample.gyro_radsec[0],
                            imu_sample.gyro_radsec[1],
                            imu_sample.gyro_radsec[2],
                            imu_sample.accel_msec2[0],
                            imu_sample.accel_msec2[1],
                            imu_sample.accel_msec2[2]
                        ], dtype=torch.float64)
                        frame_samples.append(sample_data)
                    else:
                        # Pad with zeros if needed
                        frame_samples.append(torch.zeros(6, dtype=torch.float64))
                
                if len(frame_samples) == samples_per_frame:
                    imu_data.append(torch.stack(frame_samples))
            
            if imu_data:
                print(f"✅ Extracted real IMU data for {len(imu_data)} frames")
                return torch.stack(imu_data).to(self.device)
            
        except Exception as e:
            print(f"❌ Error extracting IMU: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def extract_rgb_frames(self, vrs_path: Path, poses: List[Dict], output_video_path: Path) -> Optional[torch.Tensor]:
        """Extract RGB frames from VRS file."""
        print(f"🎥 Extracting RGB frames from VRS...")
        
        try:
            provider = data_provider.create_vrs_data_provider(str(vrs_path))
            if not provider:
                print("❌ Failed to open VRS file")
                return None
            
            # Get RGB camera stream
            rgb_stream_id = StreamId("214-1")
            config = provider.get_image_configuration(rgb_stream_id)
            
            if not config:
                print("❌ No RGB camera stream found")
                return None
            
            print(f"📷 RGB camera: {config.image_width}x{config.image_height}")
            
            # Target resolution
            target_width = 336
            target_height = 188
            
            # Extract frames aligned with poses
            frames = []
            
            print(f"📊 Extracting {len(poses)} RGB frames...")
            for i, pose in enumerate(poses):
                if self.max_frames > 0 and i >= self.max_frames:
                    break
                
                if i % 100 == 0:
                    print(f"  Progress: {i}/{len(poses)} frames...")
                
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
                    
                    # Resize to target resolution
                    img_resized = cv2.resize(img_array, (target_width, target_height))
                    
                    # Convert to tensor and normalize
                    img_tensor = torch.from_numpy(img_resized).float() / 255.0
                    
                    # Add to list
                    frames.append(img_tensor)
                else:
                    # Pad with zeros if no frame
                    frames.append(torch.zeros(target_height, target_width, 3))
            
            if frames:
                print(f"✅ Extracted {len(frames)} RGB frames")
                
                # Stack and permute to [N, C, H, W]
                visual_data = torch.stack(frames).permute(0, 3, 1, 2)
                
                # Save as video (optional)
                if output_video_path:
                    self.save_video(visual_data, output_video_path)
                
                return visual_data.to(self.device)
            
        except Exception as e:
            print(f"❌ Error extracting RGB frames: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def save_video(self, frames: torch.Tensor, output_path: Path, fps: int = 20):
        """Save frames as video."""
        try:
            # Convert to numpy and denormalize
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
            print(f"✅ Saved video to {output_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to save video: {e}")
    
    def process_sequence(self, sequence_path: Path, sequence_id: str) -> bool:
        """Process a single sequence with real IMU data."""
        print(f"\n🔄 Processing sequence: {sequence_path.name}")
        
        # Check for VRS file - look for any .vrs file in the directory
        vrs_files = list(sequence_path.glob("*.vrs"))
        if not vrs_files:
            print(f"❌ No .vrs file found in {sequence_path}")
            return False
        vrs_path = vrs_files[0]  # Use the first VRS file found
        print(f"📹 Found VRS file: {vrs_path.name}")
        
        # Extract SLAM poses (now properly resampled)
        poses = self.extract_slam_poses(sequence_path)
        if not poses:
            return False
        
        # Extract real IMU data
        imu_data = self.extract_real_imu_data(vrs_path, poses)
        if imu_data is None:
            return False
        
        # Extract RGB frames
        seq_output_dir = self.output_dir / sequence_id
        seq_output_dir.mkdir(exist_ok=True)
        
        video_path = seq_output_dir / "rgb_video.mp4"
        visual_data = self.extract_rgb_frames(vrs_path, poses, video_path)
        if visual_data is None:
            return False
        
        # Ensure matching lengths
        min_frames = min(len(poses), visual_data.shape[0], imu_data.shape[0])
        poses = poses[:min_frames]
        visual_data = visual_data[:min_frames]
        imu_data = imu_data[:min_frames]
        
        # Save data
        poses_file = seq_output_dir / "poses_quaternion.json"
        with open(poses_file, 'w') as f:
            json.dump(poses, f, indent=2)
        
        torch.save(visual_data.cpu(), seq_output_dir / "visual_data.pt")
        torch.save(imu_data.cpu(), seq_output_dir / "imu_data.pt")
        
        # Save metadata
        metadata = {
            'sequence_name': sequence_path.name,
            'sequence_id': sequence_id,
            'num_frames': min_frames,
            'visual_shape': list(visual_data.shape),
            'imu_shape': list(imu_data.shape),
            'slam_source': 'mps_slam_resampled_20hz',
            'imu_source': 'real_vrs_data',
            'imu_frequency': 1000,
            'camera_frequency': 20,
            'rotation_format': 'quaternion_xyzw'
        }
        
        with open(seq_output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Successfully processed {sequence_path.name}: {min_frames} frames")
        return True
    
    
    def process_dataset(self, sequences: Optional[List[str]] = None, max_sequences: Optional[int] = None):
        """Process multiple sequences."""
        print(f"🎯 Processing Raw AriaEveryday Dataset")
        print(f"📁 Input: {self.input_dir}")
        print(f"📁 Output: {self.output_dir}")
        print("=" * 60)
        
        # Get sequences
        if sequences:
            sequence_paths = [self.input_dir / s for s in sequences]
        else:
            sequence_paths = [d for d in self.input_dir.iterdir() 
                            if d.is_dir() and (d / "recording.vrs").exists()]
        
        if max_sequences:
            sequence_paths = sequence_paths[:max_sequences]
        
        print(f"📊 Found {len(sequence_paths)} sequences to process")
        
        processed_count = 0
        
        for i, seq_path in enumerate(tqdm(sequence_paths, desc="Processing")):
            # If processing single sequence, use sequence name as ID
            if len(sequence_paths) == 1 and sequences:
                sequence_id = seq_path.name
            else:
                sequence_id = f"{i:03d}"
            
            if self.process_sequence(seq_path, sequence_id):
                processed_count += 1
        
        print(f"\n🎉 Processing Complete!")
        print(f"✅ Successfully processed: {processed_count}/{len(sequence_paths)} sequences")
        
        # Save summary
        summary = {
            'dataset': 'AriaEveryday_Raw',
            'total_sequences': len(sequence_paths),
            'processed_sequences': processed_count,
            'imu_type': 'real_sensor_data',
            'slam_resampling': 'resampled_to_20hz',
            'max_frames_per_sequence': self.max_frames
        }
        
        with open(self.output_dir / "dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Process raw AriaEveryday with real IMU data and proper SLAM resampling')
    parser.add_argument('--input-dir', type=str, default='/mnt/ssd_ext/incSeg-data/aria_everyday',
                       help='Path to raw AriaEveryday dataset')
    parser.add_argument('--output-dir', type=str, default='aria_processed_real_imu',
                       help='Output directory')
    parser.add_argument('--max-frames', type=int, default=1000,
                       help='Max frames per sequence (default: 1000, evenly sampled)')
    parser.add_argument('--sequences', nargs='+', help='Specific sequences to process')
    parser.add_argument('--max-sequences', type=int, help='Maximum number of sequences')
    
    args = parser.parse_args()
    
    processor = AriaRawProcessor(args.input_dir, args.output_dir, args.max_frames)
    processor.process_dataset(args.sequences, args.max_sequences)


if __name__ == "__main__":
    main()