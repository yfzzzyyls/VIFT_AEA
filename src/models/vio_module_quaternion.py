import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
import hydra
from typing import Dict
import numpy as np
from scipy.spatial.transform import Rotation

class VIOLitModuleQuaternion(LightningModule):
    """VIO Lightning Module modified for quaternion output"""
    
    def __init__(
            self,
            net,
            optimizer,
            scheduler,
            criterion,
            compile,
            tester,
            metrics_calculator,
        ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['net'])
        self.net = net
        self.criterion = criterion
        self.tester = tester
        self.metrics_calculator = metrics_calculator
        
        # Initialize metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        
        # Track whether we need to convert Euler to quaternion
        self.gt_is_euler = True  # Set based on your data format

    def forward(self, x, target):
        return self.net(x, target)
    
    def convert_euler_to_quaternion_batch(self, euler_poses):
        """
        Convert batch of Euler poses to quaternions
        Input: [B, seq_len, 6] or [B*seq_len, 6]
        Output: Same shape but with 7 dims
        """
        original_shape = euler_poses.shape
        if len(original_shape) == 3:
            B, seq_len, _ = original_shape
            euler_poses = euler_poses.reshape(-1, 6)
        else:
            B_seq = original_shape[0]
            
        # Prepare output
        quat_poses = torch.zeros(euler_poses.shape[0], 7, device=euler_poses.device)
        
        # Copy translation
        quat_poses[:, :3] = euler_poses[:, :3]
        
        # Convert rotations on CPU for scipy
        euler_cpu = euler_poses[:, 3:].cpu().numpy()
        
        # Batch convert using scipy
        for i in range(euler_cpu.shape[0]):
            r = Rotation.from_euler('xyz', euler_cpu[i])
            quat = r.as_quat()  # [x, y, z, w]
            quat_poses[i, 3:] = torch.from_numpy(quat).to(euler_poses.device)
        
        # Reshape back if needed
        if len(original_shape) == 3:
            quat_poses = quat_poses.reshape(B, seq_len, 7)
            
        return quat_poses

    def training_step(self, batch, batch_idx):
        x, target = batch
        
        # Forward pass - model outputs quaternions
        out = self.forward(x, target)
        
        # Convert target from Euler to quaternion if needed
        if self.gt_is_euler and target.shape[-1] == 6:
            target_quat = self.convert_euler_to_quaternion_batch(target)
        else:
            target_quat = target
        
        # Compute loss using quaternion-aware loss function
        if hasattr(self.criterion, 'forward_quaternion'):
            loss = self.criterion.forward_quaternion(out, target_quat)
        else:
            # Fallback: split and compute separate losses
            trans_loss = torch.nn.functional.mse_loss(out[..., :3], target_quat[..., :3])
            
            # Quaternion loss using dot product
            pred_quat = out[..., 3:]
            target_q = target_quat[..., 3:]
            
            # Normalize quaternions
            pred_quat = pred_quat / (torch.norm(pred_quat, dim=-1, keepdim=True) + 1e-8)
            target_q = target_q / (torch.norm(target_q, dim=-1, keepdim=True) + 1e-8)
            
            # Compute dot product (closer to 1 means more similar)
            dot = torch.sum(pred_quat * target_q, dim=-1)
            # Use 1 - |dot| as loss (0 when identical, 1 when perpendicular)
            quat_loss = torch.mean(1.0 - torch.abs(dot))
            
            # Combine losses
            loss = trans_loss + quat_loss

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        
        # Forward pass
        out = self.forward(x, target)
        
        # Convert target if needed
        if self.gt_is_euler and target.shape[-1] == 6:
            target_quat = self.convert_euler_to_quaternion_batch(target)
        else:
            target_quat = target
        
        # Compute loss
        if hasattr(self.criterion, 'forward_quaternion'):
            loss = self.criterion.forward_quaternion(out, target_quat)
        else:
            # Same as training
            trans_loss = torch.nn.functional.mse_loss(out[..., :3], target_quat[..., :3])
            
            pred_quat = out[..., 3:]
            target_q = target_quat[..., 3:]
            
            pred_quat = pred_quat / (torch.norm(pred_quat, dim=-1, keepdim=True) + 1e-8)
            target_q = target_q / (torch.norm(target_q, dim=-1, keepdim=True) + 1e-8)
            
            dot = torch.sum(pred_quat * target_q, dim=-1)
            quat_loss = torch.mean(1.0 - torch.abs(dot))
            
            loss = trans_loss + quat_loss

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # This method is not used for our custom testing
        pass

    def on_test_epoch_end(self):
        # Only run custom testing if tester is available
        if self.tester is not None:
            results = self.tester.test(self.net)
            
            if self.metrics_calculator is not None:
                metrics = self.metrics_calculator.calculate_metrics(results)
                for name, value in metrics.items():
                    self.log(f"test/{name}", value)
            
            save_dir = self.trainer.logger.log_dir
            self.tester.save_results(results, save_dir)

    def setup(self, stage):
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}