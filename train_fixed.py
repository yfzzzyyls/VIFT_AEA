#!/usr/bin/env python3
"""Fixed training script addressing model collapse and incorporating RPMG loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from rich.console import Console
import numpy as np
from pathlib import Path

console = Console()

class CollapsePrevention(nn.Module):
    """Loss component to prevent model collapse to constant outputs."""
    
    def __init__(self, min_std=1e-4, diversity_weight=0.1):
        super().__init__()
        self.min_std = min_std
        self.diversity_weight = diversity_weight
    
    def forward(self, predictions, visual_features=None):
        """
        Calculate diversity loss to prevent constant predictions.
        
        Args:
            predictions: [B, D] predicted poses
            visual_features: Optional [B, F] visual features for correlation
        
        Returns:
            diversity_loss: scalar loss encouraging prediction diversity
        """
        batch_size = predictions.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Calculate standard deviation across batch
        pred_std = torch.std(predictions, dim=0)
        
        # Penalize low standard deviation (constant outputs)
        std_penalty = torch.sum(torch.relu(self.min_std - pred_std))
        
        # Encourage correlation with visual features if provided
        correlation_loss = 0.0
        if visual_features is not None:
            # Simple correlation: predictions should vary when features vary
            feat_std = torch.std(visual_features.view(batch_size, -1), dim=0).mean()
            pred_feat_corr = torch.abs(torch.corrcoef(
                torch.cat([predictions.mean(dim=1, keepdim=True), 
                          visual_features.view(batch_size, -1).mean(dim=1, keepdim=True)], dim=1).T
            )[0, 1])
            correlation_loss = 1.0 - pred_feat_corr
        
        return self.diversity_weight * (std_penalty + correlation_loss)


class MonitoredVIOModule(L.LightningModule):
    """Enhanced VIO module with collapse monitoring and RPMG loss option."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        use_rpmg: bool = True,
        stride: int = 20,
        monitor_collapse: bool = True,
        diversity_weight: float = 0.1,
        angle_weight: float = 100.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.stride = stride
        self.monitor_collapse = monitor_collapse
        self.collapse_prevention = CollapsePrevention(diversity_weight=diversity_weight)
        
        # Setup loss function
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent))
        
        if use_rpmg:
            console.print("[green]Using RPMG loss for rotation[/green]")
            from src.metrics.weighted_loss import RPMGPoseLoss
            self.loss_fn = RPMGPoseLoss(angle_weight=angle_weight)
        else:
            console.print("[yellow]Using standard ARVRLossWrapper[/yellow]")
            from src.metrics.arvr_loss_wrapper import ARVRLossWrapper
            self.loss_fn = ARVRLossWrapper(
                rotation_weight=angle_weight,
                translation_weight=1.0,
                use_log_scale=True
            )
        
        # Track statistics
        self.pred_stats = {'trans_std': [], 'rot_std': []}
        
    def forward(self, visual_features, imu_features):
        return self.model(visual_features, imu_features)
    
    def check_collapse(self, predictions, phase='train'):
        """Monitor for model collapse to constant outputs."""
        if not self.monitor_collapse:
            return False
            
        # Split predictions into translation and rotation
        trans_pred = predictions[:, :3]
        rot_pred = predictions[:, 3:]
        
        # Calculate standard deviations
        trans_std = torch.std(trans_pred, dim=0).mean().item()
        rot_std = torch.std(rot_pred, dim=0).mean().item()
        
        # Store for tracking
        self.pred_stats['trans_std'].append(trans_std)
        self.pred_stats['rot_std'].append(rot_std)
        
        # Check for collapse
        collapsed = trans_std < 1e-5 or rot_std < 1e-5
        
        if collapsed or self.global_step % 100 == 0:
            self.log(f'{phase}/trans_std', trans_std, prog_bar=True)
            self.log(f'{phase}/rot_std', rot_std, prog_bar=True)
            
            if collapsed:
                console.print(f"[red]WARNING: Model collapse detected at step {self.global_step}![/red]")
                console.print(f"  Translation std: {trans_std:.6f}")
                console.print(f"  Rotation std: {rot_std:.6f}")
                console.print(f"  Sample predictions: {predictions[:3].detach().cpu().numpy()}")
        
        return collapsed
    
    def training_step(self, batch, batch_idx):
        # Extract data
        visual_features = batch['visual_features']
        imu_features = batch['imu_features']
        target_poses = batch['relative_poses']
        
        # Forward pass
        predictions = self(visual_features, imu_features)
        
        # Check for collapse
        self.check_collapse(predictions, 'train')
        
        # Calculate main loss
        if hasattr(self.loss_fn, '__call__'):
            pose_loss = self.loss_fn(predictions, target_poses)
        else:
            # Handle different loss signatures
            trans_loss = F.smooth_l1_loss(predictions[:, :3], target_poses[:, :3])
            rot_loss = F.smooth_l1_loss(predictions[:, 3:], target_poses[:, 3:])
            pose_loss = trans_loss + self.hparams.angle_weight * rot_loss
        
        # Add diversity loss to prevent collapse
        diversity_loss = self.collapse_prevention(predictions, visual_features)
        
        # Total loss
        total_loss = pose_loss + diversity_loss
        
        # Logging
        self.log('train/pose_loss', pose_loss, prog_bar=True)
        self.log('train/diversity_loss', diversity_loss)
        self.log('train/total_loss', total_loss)
        
        # Log gradient norms to monitor training health
        if self.global_step % 100 == 0:
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
            total_norm = total_norm ** 0.5
            self.log('train/grad_norm', total_norm)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Similar to training but without diversity loss
        visual_features = batch['visual_features']
        imu_features = batch['imu_features']
        target_poses = batch['relative_poses']
        
        predictions = self(visual_features, imu_features)
        
        # Check for collapse
        self.check_collapse(predictions, 'val')
        
        # Calculate loss
        if hasattr(self.loss_fn, '__call__'):
            pose_loss = self.loss_fn(predictions, target_poses)
        else:
            trans_loss = F.smooth_l1_loss(predictions[:, :3], target_poses[:, :3])
            rot_loss = F.smooth_l1_loss(predictions[:, 3:], target_poses[:, 3:])
            pose_loss = trans_loss + self.hparams.angle_weight * rot_loss
        
        self.log('val/loss', pose_loss, prog_bar=True)
        
        # Calculate unaligned trajectory error for better monitoring
        # This helps detect direction issues early
        pred_trans = predictions[:, :3].detach().cpu().numpy()
        gt_trans = target_poses[:, :3].detach().cpu().numpy()
        
        # Simple direction check
        if len(pred_trans) > 1:
            pred_direction = pred_trans[-1] - pred_trans[0]
            gt_direction = gt_trans[-1] - gt_trans[0]
            
            # Normalize
            pred_norm = pred_direction / (np.linalg.norm(pred_direction) + 1e-8)
            gt_norm = gt_direction / (np.linalg.norm(gt_direction) + 1e-8)
            
            direction_similarity = np.dot(pred_norm, gt_norm)
            self.log('val/direction_similarity', direction_similarity)
        
        return pose_loss
    
    def configure_optimizers(self):
        # Optimizer with weight decay to prevent overfitting
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.optimizer_cfg.lr,
            weight_decay=1e-4
        )
        
        # Scheduler with warmup
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.optimizer_cfg.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos'
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    """Main training function with fixes applied."""
    
    console.print("[bold cyan]Starting Fixed VIO Training[/bold cyan]")
    console.print(f"  Use RPMG Loss: {cfg.get('use_rpmg', True)}")
    console.print(f"  Stride: {cfg.get('stride', 20)}")
    console.print(f"  Monitor Collapse: {cfg.get('monitor_collapse', True)}")
    
    # Set training parameters
    cfg.data.stride = cfg.get('stride', 20)  # Use larger stride
    cfg.data.batch_size = 32  # Reasonable batch size
    
    # Initialize data module with fixed parameters
    from src.data.aria_datamodule import AriaDataModule
    datamodule = AriaDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        stride=cfg.data.stride,
        pose_scale=100.0,  # Scale to cm for numerical stability
        window_size=cfg.data.get('window_size', 11),
    )
    
    # Initialize model
    from src.models.multihead_vio import MultiHeadVIO
    model = MultiHeadVIO(
        visual_input_dim=cfg.model.visual_dim,
        imu_input_dim=cfg.model.imu_dim,
        pose_output_dim=cfg.model.output_dim,
        **cfg.model
    )
    
    # Create Lightning module with monitoring
    lit_model = MonitoredVIOModule(
        model=model,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.get('scheduler', {}),
        use_rpmg=cfg.get('use_rpmg', True),
        stride=cfg.data.stride,
        monitor_collapse=cfg.get('monitor_collapse', True),
        diversity_weight=cfg.get('diversity_weight', 0.1),
        angle_weight=cfg.get('angle_weight', 100.0),
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            filename='vio-fixed-{epoch:02d}-{val_loss:.4f}'
        ),
        EarlyStopping(
            monitor='val/loss',
            patience=20,
            mode='min'
        ),
        RichProgressBar()
    ]
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.get('precision', 32),
        callbacks=callbacks,
        gradient_clip_val=1.0,  # Prevent exploding gradients
        log_every_n_steps=10,
        val_check_interval=0.25,  # Validate 4x per epoch
        accumulate_grad_batches=cfg.trainer.get('accumulate_grad_batches', 1),
    )
    
    # Train model
    trainer.fit(lit_model, datamodule=datamodule)
    
    console.print("[bold green]Training completed![/bold green]")
    
    # Final collapse check
    if lit_model.monitor_collapse:
        trans_stds = lit_model.pred_stats['trans_std']
        rot_stds = lit_model.pred_stats['rot_std']
        
        console.print("\n[bold]Final Statistics:[/bold]")
        console.print(f"  Average translation std: {np.mean(trans_stds):.6f}")
        console.print(f"  Average rotation std: {np.mean(rot_stds):.6f}")
        console.print(f"  Min translation std: {np.min(trans_stds):.6f}")
        console.print(f"  Min rotation std: {np.min(rot_stds):.6f}")


if __name__ == "__main__":
    main()