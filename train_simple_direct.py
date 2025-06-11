#!/usr/bin/env python3
"""
Simple direct training script for VIFT on full AriaEveryday dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime
import json

class AriaLatentDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir) / split
        self.samples = []
        
        # Find all samples
        for gt_file in sorted(self.data_dir.glob("*_gt.npy")):
            idx = gt_file.stem.split('_')[0]
            visual_file = self.data_dir / f"{idx}_visual.npy"
            imu_file = self.data_dir / f"{idx}_imu.npy"
            
            if visual_file.exists() and imu_file.exists():
                self.samples.append({
                    'visual': visual_file,
                    'imu': imu_file,
                    'gt': gt_file
                })
        
        print(f"Found {len(self.samples)} samples in {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        visual = torch.from_numpy(np.load(sample['visual'])).float()
        imu = torch.from_numpy(np.load(sample['imu'])).float()
        gt = torch.from_numpy(np.load(sample['gt'])).float()
        
        return visual, imu, gt

class SimpleVIFT(nn.Module):
    def __init__(self, visual_dim=512, imu_dim=256, hidden_dim=512, output_dim=7):
        super().__init__()
        
        # Visual encoder
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # IMU encoder
        self.imu_encoder = nn.Sequential(
            nn.Linear(imu_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Fusion and prediction
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, visual, imu):
        # visual: [B, T, 256], imu: [B, T, 256]
        B, T, _ = visual.shape
        
        # Process each timestep
        outputs = []
        for t in range(T):
            v_feat = self.visual_encoder(visual[:, t])
            i_feat = self.imu_encoder(imu[:, t])
            
            # Fuse features
            fused = torch.cat([v_feat, i_feat], dim=-1)
            pred = self.fusion(fused)
            
            outputs.append(pred)
        
        return torch.stack(outputs, dim=1)  # [B, T, 7]

def compute_loss(pred, gt):
    """Compute loss with separate translation and rotation components"""
    # Split translation and rotation
    pred_trans = pred[..., :3]  # [B, T, 3]
    pred_rot = pred[..., 3:]    # [B, T, 4]
    
    gt_trans = gt[..., :3]
    gt_rot = gt[..., 3:]
    
    # Normalize quaternions
    pred_rot = torch.nn.functional.normalize(pred_rot, p=2, dim=-1)
    gt_rot = torch.nn.functional.normalize(gt_rot, p=2, dim=-1)
    
    # Translation loss (MSE in centimeters)
    trans_loss = torch.mean((pred_trans - gt_trans) ** 2)
    
    # Rotation loss (1 - |<q1, q2>|)
    rot_loss = 1.0 - torch.abs(torch.sum(pred_rot * gt_rot, dim=-1)).mean()
    
    # Combined loss with weighting
    total_loss = trans_loss + 100.0 * rot_loss
    
    return total_loss, trans_loss, rot_loss

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_trans_loss = 0
    total_rot_loss = 0
    
    for visual, imu, gt in tqdm(dataloader, desc="Training"):
        visual = visual.to(device)
        imu = imu.to(device)
        gt = gt.to(device)
        
        # Forward pass
        pred = model(visual, imu)
        
        # Compute loss
        loss, trans_loss, rot_loss = compute_loss(pred, gt)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        total_trans_loss += trans_loss.item()
        total_rot_loss += rot_loss.item()
    
    n_batches = len(dataloader)
    return total_loss / n_batches, total_trans_loss / n_batches, total_rot_loss / n_batches

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_trans_loss = 0
    total_rot_loss = 0
    
    with torch.no_grad():
        for visual, imu, gt in tqdm(dataloader, desc="Evaluating"):
            visual = visual.to(device)
            imu = imu.to(device)
            gt = gt.to(device)
            
            # Forward pass
            pred = model(visual, imu)
            
            # Compute loss
            loss, trans_loss, rot_loss = compute_loss(pred, gt)
            
            # Track losses
            total_loss += loss.item()
            total_trans_loss += trans_loss.item()
            total_rot_loss += rot_loss.item()
    
    n_batches = len(dataloader)
    return total_loss / n_batches, total_trans_loss / n_batches, total_rot_loss / n_batches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/mnt/ssd_ext/incSeg-data/aria_latent_data_pretrained')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', type=str, default='full_dataset_checkpoints')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create timestamp directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / timestamp
    run_dir.mkdir(exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create datasets
    train_dataset = AriaLatentDataset(args.data_dir, 'train')
    val_dataset = AriaLatentDataset(args.data_dir, 'val')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = SimpleVIFT().to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    train_history = []
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss, train_trans_loss, train_rot_loss = train_epoch(model, train_loader, optimizer, args.device)
        
        # Evaluate
        val_loss, val_trans_loss, val_rot_loss = evaluate(model, val_loader, args.device)
        
        # Step scheduler
        scheduler.step(val_loss)
        
        # Log results
        print(f"Train Loss: {train_loss:.4f} (Trans: {train_trans_loss:.4f}, Rot: {train_rot_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Trans: {val_trans_loss:.4f}, Rot: {val_rot_loss:.4f})")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save history
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_trans_loss': train_trans_loss,
            'train_rot_loss': train_rot_loss,
            'val_loss': val_loss,
            'val_trans_loss': val_trans_loss,
            'val_rot_loss': val_rot_loss,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }
            torch.save(checkpoint, run_dir / 'best_checkpoint.pth')
            print("  => Saved best checkpoint")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }
            torch.save(checkpoint, run_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Save training history
    with open(run_dir / 'training_history.json', 'w') as f:
        json.dump(train_history, f, indent=2)
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Results saved to: {run_dir}")

if __name__ == '__main__':
    main()