import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
from typing import Any, Dict, Optional

class AriaVIOModule(LightningModule):
    """Aria-compatible VIO module that wraps the original VIFT architecture"""
    
    def __init__(
        self,
        net,
        optimizer,
        scheduler,
        criterion,
        compile: bool = False,
        tester = None,
        metrics_calculator = None,
    ):
        super().__init__()
        
        self.save_hyperparameters(logger=False, ignore=['net', 'criterion'])
        self.net = net
        self.criterion = criterion
        self.tester = tester
        self.metrics_calculator = metrics_calculator
        
        # Initialize metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
    
    def forward(self, x, target=None):
        """Forward pass - handle VIFT input format"""
        # VIFT expects (batch, target) format for training
        # x is the visual_inertial_features [batch, seq_len, 768]
        
        # Create VIFT-compatible batch format: (visual_inertial_features, None, None)
        batch = (x, None, None)
        
        # Forward through the pose transformer
        return self.net(batch, target)
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # Handle VIFT-compatible format: (visual_inertial_features, poses)
        visual_inertial_features, target = batch
        
        # Forward pass through the original VIFT transformer
        preds = self.forward(visual_inertial_features, target)
        loss = self.criterion(preds, target)
        
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log learning rate
        if self.optimizers():
            current_lr = self.optimizers().param_groups[0]['lr']
            self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        visual_inertial_features, target = batch
        
        preds = self.forward(visual_inertial_features, target)
        loss = self.criterion(preds, target)
        
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        visual_inertial_features, target = batch
        
        preds = self.forward(visual_inertial_features)
        loss = self.criterion(preds, target)
        
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log metrics if available
        if self.metrics_calculator is not None:
            metrics = self.metrics_calculator(preds, target)
            for metric_name, metric_value in metrics.items():
                self.log(f"test/{metric_name}", metric_value, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer = self.hparams.optimizer(params=self.parameters())
        
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
    
    def setup(self, stage: str):
        """Lightning hook called at the beginning of fit/validate/test/predict"""
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)