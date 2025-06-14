import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from scipy.spatial.transform import Rotation as R


class AriaKITTIFormat(Dataset):
    """
    Aria dataset wrapper that mimics KITTI dataset format for VIFT evaluation.
    This allows testing KITTI-trained models on Aria data.
    """
    
    def __init__(self, root, sequence_length=11, train_seqs=None, transform=None):
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Default to all sequences if none specified
        if train_seqs is None:
            # Get all sequence directories
            seq_dirs = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
            self.train_seqs = seq_dirs
        else:
            self.train_seqs = train_seqs
            
        self.make_dataset()
    
    def make_dataset(self):
        sequence_set = []
        
        for seq_name in self.train_seqs:
            seq_dir = self.root / seq_name
            if not seq_dir.exists():
                print(f"Warning: Sequence {seq_name} not found, skipping...")
                continue
                
            # Load metadata
            with open(seq_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            
            # Load data
            visual_data = torch.load(seq_dir / 'visual_data.pt')  # [N, 3, H, W]
            imu_data = torch.load(seq_dir / 'imu_data.pt')  # [N, imu_freq/cam_freq, 6]
            
            # Load poses
            with open(seq_dir / 'poses_quaternion.json', 'r') as f:
                poses_data = json.load(f)
                
            # Convert poses to numpy array
            poses = []
            for pose in poses_data:
                # Extract translation and quaternion
                t = pose['translation']
                q = pose['quaternion']  # XYZW format
                
                # Convert to 4x4 transformation matrix (KITTI format)
                T = np.eye(4)
                T[:3, 3] = [t[0], t[1], t[2]]
                
                # Convert quaternion to rotation matrix
                rot = R.from_quat([q[0], q[1], q[2], q[3]])
                T[:3, :3] = rot.as_matrix()
                
                poses.append(T)
            
            poses = np.array(poses)
            
            # Compute relative poses between consecutive frames
            poses_rel = []
            for i in range(len(poses) - 1):
                T_rel = np.linalg.inv(poses[i]) @ poses[i + 1]
                poses_rel.append(T_rel)
            poses_rel = np.array(poses_rel)
            
            # Create samples with sliding window
            num_frames = len(visual_data)
            imu_ratio = imu_data.shape[1]  # IMU samples per image frame
            
            for i in range(num_frames - self.sequence_length):
                # Image samples (as tensors, not file paths)
                img_samples = visual_data[i:i+self.sequence_length]
                
                # IMU samples - KITTI expects 10 IMU samples between consecutive frames
                # Aria has 10 IMU samples per frame (matching KITTI's expectation)
                # For a sequence of N images, we need (N-1)*10 + 1 IMU samples
                imu_samples = []
                for j in range(self.sequence_length - 1):
                    # Get IMU data between frame j and j+1
                    imu_frame = imu_data[i + j]  # Shape: [10, 6]
                    imu_samples.append(imu_frame)
                # Add one final IMU sample from the last frame
                imu_samples.append(imu_data[i + self.sequence_length - 1][:1])  # Just first IMU sample
                imu_samples = np.concatenate(imu_samples, axis=0)  # Shape: [(N-1)*10 + 1, 6]
                
                # Pose samples
                pose_samples = poses[i:i+self.sequence_length]
                pose_rel_samples = poses_rel[i:i+self.sequence_length-1]
                
                # Compute segment rotation (for weighting)
                segment_rot = self._rotation_error(pose_samples[0], pose_samples[-1])
                
                sample = {
                    'imgs': img_samples,
                    'imus': imu_samples,
                    'gts': pose_rel_samples,
                    'rot': segment_rot,
                    'seq_name': seq_name,
                    'start_idx': i
                }
                sequence_set.append(sample)
                
        self.samples = sequence_set
        
        # For simplicity, use uniform weights (can be improved later)
        self.weights = [1.0] * len(self.samples)
        
    def _rotation_error(self, pose1, pose2):
        """Compute rotation error between two poses"""
        R1 = pose1[:3, :3]
        R2 = pose2[:3, :3]
        R_rel = R1.T @ R2
        
        # Compute rotation angle from rotation matrix
        trace = np.trace(R_rel)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        return angle
        
    def __getitem__(self, index):
        sample = self.samples[index]
        
        # Images are already tensors
        imgs = sample['imgs']
        
        # Convert to numpy for transform compatibility
        imgs_np = imgs.numpy().transpose(0, 2, 3, 1)  # [N, H, W, C]
        
        if self.transform is not None:
            # Transform expects list of numpy arrays
            imgs_list = [imgs_np[i] for i in range(len(imgs_np))]
            imgs, imus, gts = self.transform(imgs_list, np.copy(sample['imus']), np.copy(sample['gts']))
        else:
            # Convert back to tensor format [N, C, H, W]
            imgs = torch.from_numpy(imgs_np.transpose(0, 3, 1, 2)).float()
            imus = np.copy(sample['imus'])
            gts = np.copy(sample['gts']).astype(np.float32)
            
        rot = sample['rot'].astype(np.float32)
        weight = self.weights[index]
        
        # Convert relative poses to 6DOF format (3 trans + 3 rot euler)
        gts_6dof = []
        for T in gts:
            # Extract translation
            trans = T[:3, 3]
            
            # Extract rotation and convert to Euler angles
            rot_mat = T[:3, :3]
            r = R.from_matrix(rot_mat)
            euler = r.as_euler('xyz')  # XYZ Euler angles
            
            gts_6dof.append(np.concatenate([trans, euler]))
            
        gts_6dof = np.array(gts_6dof, dtype=np.float32)
        
        return (imgs, torch.from_numpy(imus).float(), rot, weight), torch.from_numpy(gts_6dof).float()
        
    def __len__(self):
        return len(self.samples)
        
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Root: {}\n'.format(self.root)
        fmt_str += '    Training sequences: {}\n'.format(self.train_seqs)
        fmt_str += '    Number of segments: {}\n'.format(self.__len__())
        fmt_str += '    Sequence length: {}\n'.format(self.sequence_length)
        return fmt_str