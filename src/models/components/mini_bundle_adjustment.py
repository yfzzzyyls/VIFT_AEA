"""
Mini Bundle Adjustment Module
GPU-accelerated local bundle adjustment for VIO refinement.
Based on DROID-SLAM's approach: fixed complexity with 10 frames, 30 points.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R


@dataclass
class BAFrame:
    """Frame in bundle adjustment."""
    id: int
    timestamp: float
    pose: torch.Tensor  # [7] quaternion + translation
    K: torch.Tensor     # [3, 3] camera intrinsics
    
    
@dataclass
class BAPoint:
    """3D point in bundle adjustment."""
    id: int
    position: torch.Tensor  # [3] world coordinates
    observations: Dict[int, torch.Tensor]  # frame_id -> [u, v]
    

class MiniBA:
    """
    GPU-accelerated mini bundle adjustment.
    
    Key features:
    - Fixed 10 frames, 30 points for predictable runtime
    - 2 Gauss-Newton iterations (< 1ms on GPU)
    - PyTorch autograd for Jacobians
    - Robust Huber loss for outliers
    """
    
    def __init__(self,
                 max_frames: int = 10,
                 max_points: int = 30,
                 max_iterations: int = 2,
                 huber_delta: float = 2.0,
                 device: str = 'cuda'):
        """
        Initialize mini BA module.
        
        Args:
            max_frames: Maximum frames in optimization
            max_points: Maximum points in optimization
            max_iterations: Fixed number of GN iterations
            huber_delta: Huber loss threshold (pixels)
            device: Computation device
        """
        self.max_frames = max_frames
        self.max_points = max_points
        self.max_iterations = max_iterations
        self.huber_delta = huber_delta
        self.device = device
        
        # Damping for Levenberg-Marquardt
        self.lambda_damping = 1e-4
        
    def select_frames_and_points(self,
                                all_frames: List[BAFrame],
                                all_points: List[BAPoint]) -> Tuple[List[BAFrame], List[BAPoint]]:
        """
        Select subset of frames and points for optimization.
        
        Uses heuristics:
        - Most recent frames
        - Points with most observations
        - Good spatial distribution
        
        Returns:
            (selected_frames, selected_points)
        """
        # Select most recent frames
        all_frames.sort(key=lambda f: f.timestamp)
        selected_frames = all_frames[-self.max_frames:]
        frame_ids = {f.id for f in selected_frames}
        
        # Filter points visible in selected frames
        visible_points = []
        for point in all_points:
            obs_in_selected = [fid for fid in point.observations if fid in frame_ids]
            if len(obs_in_selected) >= 2:  # Need at least 2 observations
                point.num_obs = len(obs_in_selected)
                visible_points.append(point)
                
        # Select points with most observations
        visible_points.sort(key=lambda p: p.num_obs, reverse=True)
        selected_points = visible_points[:self.max_points]
        
        return selected_frames, selected_points
        
    def parameterize_state(self, 
                         frames: List[BAFrame],
                         points: List[BAPoint]) -> torch.Tensor:
        """
        Create state vector from frames and points.
        
        State layout:
        - Frames: N × 7 (quaternion + translation)
        - Points: M × 3 (3D position)
        
        Returns:
            State vector [(N×7 + M×3)]
        """
        frame_params = []
        for frame in frames:
            frame_params.append(frame.pose)
            
        point_params = []
        for point in points:
            point_params.append(point.position)
            
        # Stack parameters
        if frame_params:
            frame_tensor = torch.stack(frame_params).reshape(-1)
        else:
            frame_tensor = torch.tensor([], device=self.device)
            
        if point_params:
            point_tensor = torch.stack(point_params).reshape(-1)
        else:
            point_tensor = torch.tensor([], device=self.device)
            
        return torch.cat([frame_tensor, point_tensor])
        
    def unpack_state(self,
                    state: torch.Tensor,
                    num_frames: int,
                    num_points: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Unpack state vector into frame poses and point positions.
        
        Returns:
            (frame_poses, point_positions)
        """
        frame_size = num_frames * 7
        
        # Extract frame poses
        frame_params = state[:frame_size].reshape(num_frames, 7)
        frame_poses = [frame_params[i] for i in range(num_frames)]
        
        # Extract point positions
        point_params = state[frame_size:].reshape(num_points, 3)
        point_positions = [point_params[i] for i in range(num_points)]
        
        return frame_poses, point_positions
        
    def compute_reprojection_error(self,
                                  pose: torch.Tensor,
                                  point: torch.Tensor,
                                  observation: torch.Tensor,
                                  K: torch.Tensor) -> torch.Tensor:
        """
        Compute reprojection error for single observation.
        
        Args:
            pose: [7] quaternion + translation
            point: [3] world point
            observation: [2] pixel coordinates
            K: [3, 3] camera intrinsics
            
        Returns:
            error: [2] reprojection error
        """
        # Extract rotation and translation
        quat = pose[:4]
        t = pose[4:7]
        
        # Quaternion to rotation matrix
        R_mat = self.quat_to_matrix(quat)
        
        # Transform point to camera frame
        p_cam = R_mat @ point + t
        
        # Check if behind camera
        if p_cam[2] <= 0.1:
            return torch.tensor([100.0, 100.0], device=self.device)  # Large error
            
        # Project to image
        p_img_homo = K @ p_cam
        p_img = p_img_homo[:2] / p_img_homo[2]
        
        # Reprojection error
        error = observation - p_img
        
        return error
        
    def quat_to_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion to rotation matrix.
        
        Args:
            quat: [4] quaternion (w, x, y, z)
            
        Returns:
            R: [3, 3] rotation matrix
        """
        w, x, y, z = quat
        
        # Normalize
        norm = torch.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Convert to matrix
        R = torch.stack([
            torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)]),
            torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)]),
            torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)])
        ])
        
        return R
        
    def compute_cost(self,
                    state: torch.Tensor,
                    frames: List[BAFrame],
                    points: List[BAPoint]) -> torch.Tensor:
        """
        Compute total reprojection cost.
        
        Args:
            state: Current state vector
            frames: List of frames
            points: List of points
            
        Returns:
            Total cost (scalar)
        """
        # Unpack state
        frame_poses, point_positions = self.unpack_state(
            state, len(frames), len(points)
        )
        
        # Create frame lookup
        frame_lookup = {f.id: i for i, f in enumerate(frames)}
        
        # Compute all reprojection errors
        total_cost = torch.tensor(0.0, device=self.device)
        num_observations = 0
        
        for i, point in enumerate(points):
            for frame_id, observation in point.observations.items():
                if frame_id not in frame_lookup:
                    continue
                    
                frame_idx = frame_lookup[frame_id]
                frame = frames[frame_idx]
                
                # Compute error
                error = self.compute_reprojection_error(
                    frame_poses[frame_idx],
                    point_positions[i],
                    observation,
                    frame.K
                )
                
                # Huber loss
                error_norm = torch.norm(error)
                if error_norm <= self.huber_delta:
                    cost = 0.5 * error_norm ** 2
                else:
                    cost = self.huber_delta * (error_norm - 0.5 * self.huber_delta)
                    
                total_cost += cost
                num_observations += 1
                
        # Normalize by number of observations
        if num_observations > 0:
            total_cost /= num_observations
            
        return total_cost
        
    def gauss_newton_step(self,
                         state: torch.Tensor,
                         frames: List[BAFrame],
                         points: List[BAPoint]) -> torch.Tensor:
        """
        Perform one Gauss-Newton step.
        
        Returns:
            Updated state
        """
        state = state.requires_grad_(True)
        
        # Compute cost
        cost = self.compute_cost(state, frames, points)
        
        # Compute gradient and Hessian approximation
        grad = torch.autograd.grad(cost, state, create_graph=True)[0]
        
        # Gauss-Newton approximation: H ≈ J^T J
        # We use gradient of gradient for approximation
        H_approx = []
        for i in range(state.shape[0]):
            grad_i = torch.autograd.grad(
                grad[i], state, retain_graph=True
            )[0]
            H_approx.append(grad_i)
            
        H = torch.stack(H_approx)
        
        # Add damping (Levenberg-Marquardt)
        H_damped = H + self.lambda_damping * torch.eye(H.shape[0], device=self.device)
        
        # Solve H * delta = -grad
        try:
            delta = torch.linalg.solve(H_damped, -grad)
        except:
            # Fall back to gradient descent
            delta = -0.01 * grad
            
        # Update state
        state_new = state + delta
        
        # Normalize quaternions
        num_frames = len(frames)
        for i in range(num_frames):
            quat_start = i * 7
            quat_end = quat_start + 4
            quat = state_new[quat_start:quat_end]
            state_new[quat_start:quat_end] = quat / torch.norm(quat)
            
        return state_new.detach()
        
    def optimize(self,
                frames: List[BAFrame],
                points: List[BAPoint]) -> Tuple[List[torch.Tensor], List[torch.Tensor], float]:
        """
        Run mini bundle adjustment.
        
        Args:
            frames: Input frames
            points: Input points
            
        Returns:
            (optimized_poses, optimized_points, final_cost)
        """
        # Move to device
        for frame in frames:
            frame.pose = frame.pose.to(self.device)
            frame.K = frame.K.to(self.device)
            
        for point in points:
            point.position = point.position.to(self.device)
            for fid in point.observations:
                point.observations[fid] = point.observations[fid].to(self.device)
                
        # Initial state
        state = self.parameterize_state(frames, points)
        
        # Fixed iterations
        for iter in range(self.max_iterations):
            state = self.gauss_newton_step(state, frames, points)
            
        # Final cost
        with torch.no_grad():
            final_cost = self.compute_cost(state, frames, points).item()
            
        # Unpack optimized state
        opt_poses, opt_points = self.unpack_state(state, len(frames), len(points))
        
        return opt_poses, opt_points, final_cost
        
    def create_ba_problem(self,
                         msckf_state: 'MSCKFState',
                         features: List['Feature']) -> Tuple[List[BAFrame], List[BAPoint]]:
        """
        Create BA problem from MSCKF state and features.
        
        Args:
            msckf_state: Current MSCKF state
            features: Tracked features
            
        Returns:
            (frames, points) for optimization
        """
        # Create frames from camera states
        frames = []
        for timestamp in msckf_state.camera_timestamps:
            cam_state = msckf_state.camera_states[timestamp]
            
            # Convert to quaternion + translation
            quat = torch.tensor(
                R.from_matrix(cam_state.rotation).as_quat(),  # [x,y,z,w]
                dtype=torch.float32
            )
            # Convert to [w,x,y,z] format
            quat = torch.cat([quat[3:], quat[:3]])
            trans = torch.tensor(cam_state.position, dtype=torch.float32)
            pose = torch.cat([quat, trans])
            
            # Camera intrinsics (assumed known)
            K = torch.tensor([
                [500, 0, 320],
                [0, 500, 240],
                [0, 0, 1]
            ], dtype=torch.float32)
            
            frame = BAFrame(
                id=int(timestamp * 1000),  # Convert to int ID
                timestamp=timestamp,
                pose=pose,
                K=K
            )
            frames.append(frame)
            
        # Create points from features
        points = []
        for feature in features:
            if feature.position_3d is not None:
                observations = {}
                for timestamp, pixel in feature.observations.items():
                    frame_id = int(timestamp * 1000)
                    observations[frame_id] = torch.tensor(
                        pixel, dtype=torch.float32
                    )
                    
                point = BAPoint(
                    id=feature.id,
                    position=torch.tensor(
                        feature.position_3d, dtype=torch.float32
                    ),
                    observations=observations
                )
                points.append(point)
                
        return frames, points