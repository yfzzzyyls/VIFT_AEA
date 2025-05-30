#!/usr/bin/env python3
"""
Simple latent caching for Aria dataset without VIFT dependencies
"""
import argparse
import json
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
from pathlib import Path
from tqdm import tqdm

class SimpleVisualEncoder(nn.Module):
    """Simple visual encoder using ResNet"""
    def __init__(self):
        super().__init__()
        # Use ResNet18 backbone
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(512, 512)  # Project to 512 dims
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 512)),
            transforms.Lambda(lambda x: x - 0.5)  # VIFT normalization
        ])
    
    def forward(self, x):
        # x: [batch, 3, H, W]
        x = torch.stack([self.transform(img) for img in x])
        features = self.backbone(x)  # [batch, 512, 1, 1]
        features = features.flatten(1)  # [batch, 512]
        return self.projection(features)  # [batch, 512]

def process_aria_sequence_simple(seq_dir: Path, visual_encoder, device='cpu'):
    """Simple processing without VIFT dependencies - handles chunked visual data"""
    
    print(f"Processing sequence: {seq_dir.name}")
    
    try:
        # Load IMU data
        imu_data = torch.load(seq_dir / "imu_data.pt", map_location=device)
        
        # Load visual data (check if chunked or single file)
        visual_metadata_path = seq_dir / "visual_metadata.json"
        if visual_metadata_path.exists():
            # Load chunked visual data
            with open(visual_metadata_path, 'r') as f:
                visual_metadata = json.load(f)
            
            visual_chunks = []
            for chunk_file in visual_metadata['chunks']:
                chunk = torch.load(seq_dir / chunk_file, map_location=device)
                visual_chunks.append(chunk)
            
            visual_data = torch.cat(visual_chunks, dim=0)
            print(f"📦 Loaded {len(visual_chunks)} visual chunks, total shape: {visual_data.shape}")
        else:
            # Load single file (legacy format)
            visual_data = torch.load(seq_dir / "visual_data.pt", map_location=device)
        
        # Load poses
        with open(seq_dir / "poses.json", 'r') as f:
            poses = json.load(f)
    except Exception as e:
        print(f"❌ Error loading sequence {seq_dir.name}: {e}")
        return None
    
    num_frames = min(len(visual_data), len(imu_data), len(poses))
    sequence_length = 11
    
    latent_vectors = []
    ground_truths = []
    
    for start_idx in range(0, num_frames - sequence_length + 1, 5):
        end_idx = start_idx + sequence_length
        
        # Extract sequences
        imgs_seq = visual_data[start_idx:end_idx]  # [11, 3, H, W]
        imus_seq = imu_data[start_idx:end_idx]     # [11, 33, 6]
        poses_seq = poses[start_idx:end_idx]
        
        # Process visual features
        with torch.no_grad():
            visual_features = visual_encoder(imgs_seq.to(device))  # [11, 512]
        
        # Process IMU features - average 33 samples to 6
        imu_features = imus_seq.mean(dim=1)  # [11, 6]
        
        # Pad IMU features to 256 dims to match VIFT format
        imu_padded = torch.zeros(sequence_length, 256)
        imu_padded[:, :6] = imu_features
        
        # Concatenate to get 768 dims total
        latent_vector = torch.cat([visual_features.cpu(), imu_padded], dim=1)  # [11, 768]
        latent_vectors.append(latent_vector.numpy())
        
        # Prepare ground truth - compute RELATIVE poses
        gt_poses = []
        for i in range(len(poses_seq)):
            if i == 0:
                # First frame: no motion (identity transformation)
                gt_poses.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                # Compute relative transformation from pose[i-1] to pose[i]
                prev_pose = poses_seq[i-1]
                curr_pose = poses_seq[i]
                
                # Get 6DOF poses
                if 'pose_6dof' in prev_pose:
                    prev_6dof = prev_pose['pose_6dof']
                    curr_6dof = curr_pose['pose_6dof']
                else:
                    # Fallback: construct from translation + rotation_euler
                    prev_6dof = prev_pose['rotation_euler'] + prev_pose['translation']
                    curr_6dof = curr_pose['rotation_euler'] + curr_pose['translation']
                
                # Compute relative pose (simple difference for now)
                # For proper SE3 relative transform, we'd use matrix operations
                rel_rotation = [curr_6dof[j] - prev_6dof[j] for j in range(3)]
                rel_translation = [curr_6dof[j+3] - prev_6dof[j+3] for j in range(3)]
                
                rel_pose = rel_rotation + rel_translation
                gt_poses.append(rel_pose)
        
        ground_truths.append(np.array(gt_poses))
    
    return latent_vectors, ground_truths

def main():
    parser = argparse.ArgumentParser(description="Simple latent caching for Aria dataset")
    parser.add_argument("--data_dir", type=str, default="data/aria_real_train")
    parser.add_argument("--save_dir", type=str, default="aria_latent_data/train")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], default="train",
                       help="Processing mode: train (all sequences), val (specific sequences), or test (test sequences)")
    parser.add_argument("--val_sequences", type=str, default="auto",
                       help="Comma-separated validation/test sequence IDs, or 'auto' to detect all sequences (only used in val/test mode)")
    parser.add_argument("--max_sequences", type=int, default=None,
                       help="Maximum number of sequences to process (e.g., 10 for train, 5 for test)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"🖥️ Using device: {device}")
    
    # Create simple visual encoder
    visual_encoder = SimpleVisualEncoder().to(device)
    visual_encoder.eval()
    
    # Find sequences based on mode
    if args.mode == "train":
        # Find all existing sequence directories for training
        sequence_dirs = []
        for seq_dir in sorted(data_dir.glob("[0-9]*")):
            if seq_dir.is_dir() and seq_dir.name.isdigit():
                sequence_dirs.append(seq_dir)
                print(f"✅ Found training sequence: {seq_dir.name}")
        print(f"📁 Found {len(sequence_dirs)} training sequences")
    else:  # val or test mode
        # Process specified validation/test sequences or auto-detect
        if args.val_sequences == "auto":
            # Auto-detect all sequences in the directory
            sequence_dirs = []
            for seq_dir in sorted(data_dir.glob("[0-9]*")):
                if seq_dir.is_dir() and seq_dir.name.isdigit():
                    sequence_dirs.append(seq_dir)
                    print(f"✅ Found {args.mode} sequence: {seq_dir.name}")
            print(f"📁 Found {len(sequence_dirs)} {args.mode} sequences (auto-detected)")
        else:
            # Process only specified validation/test sequences
            val_seq_ids = [x.strip() for x in args.val_sequences.split(',')]
            sequence_dirs = []
            for seq_id in val_seq_ids:
                # Try both 2-digit and 3-digit formats
                seq_dir_2digit = data_dir / seq_id
                seq_dir_3digit = data_dir / f"{int(seq_id):03d}" if seq_id.isdigit() else None
                
                if seq_dir_2digit.exists():
                    sequence_dirs.append(seq_dir_2digit)
                    print(f"✅ Found {args.mode} sequence: {seq_id}")
                elif seq_dir_3digit and seq_dir_3digit.exists():
                    sequence_dirs.append(seq_dir_3digit)
                    print(f"✅ Found {args.mode} sequence: {seq_dir_3digit.name}")
                else:
                    print(f"⚠️ {args.mode.capitalize()} sequence {seq_id} not found")
            print(f"📁 Found {len(sequence_dirs)} {args.mode} sequences: {[d.name for d in sequence_dirs]}")
    
    if not sequence_dirs:
        print(f"❌ No sequences found")
        return
    
    # Limit number of sequences if specified
    if args.max_sequences is not None and len(sequence_dirs) > args.max_sequences:
        sequence_dirs = sequence_dirs[:args.max_sequences]
        print(f"🔄 Limited to first {args.max_sequences} sequences")
    
    # Process sequences
    sample_idx = 0
    
    for seq_dir in tqdm(sequence_dirs, desc="Processing sequences"):
        try:
            result = process_aria_sequence_simple(seq_dir, visual_encoder, device)
            if result is None:
                continue
                
            latent_vectors, ground_truths = result
            
            # Save samples
            for lat_vec, gt in zip(latent_vectors, ground_truths):
                np.save(save_dir / f"{sample_idx}.npy", lat_vec)
                np.save(save_dir / f"{sample_idx}_gt.npy", gt)
                # Create dummy rotation and weight files for compatibility
                np.save(save_dir / f"{sample_idx}_rot.npy", np.eye(3))
                np.save(save_dir / f"{sample_idx}_w.npy", np.ones(1))
                sample_idx += 1
                
        except Exception as e:
            print(f"❌ Error processing {seq_dir.name}: {e}")
            continue
    
    print(f"\n🎉 Simple latent caching complete!")
    print(f"📊 Created {sample_idx} {args.mode} samples in {save_dir}")
    
    if args.mode == "train":
        print(f"\n📝 Next steps:")
        print(f"1. Create validation cache:")
        print(f"   python data/latent_caching_aria.py \\")
        print(f"     --mode val --save_dir aria_latent_data/val \\")
        print(f"     --val_sequences '100,101,102,103,104' --device {args.device}")
        print(f"   Or to limit sequences:")
        print(f"   python data/latent_caching_aria.py \\")
        print(f"     --mode val --save_dir aria_latent_data/val \\")
        print(f"     --max_sequences 5 --device {args.device}")
        print(f"2. Run training: python src/train.py data=aria_latent model=aria_vio")
    else:
        print(f"\n📝 Ready for training!")
        print(f"Run: python src/train.py data=aria_latent model=aria_vio")

if __name__ == "__main__":
    main()