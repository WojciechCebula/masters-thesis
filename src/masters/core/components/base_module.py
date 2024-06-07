from typing import Any, Dict

import torch.nn as nn
import torch.optim as optim

from lightning import LightningModule
from torchmetrics import MetricCollection


class BaseLightningModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_function: nn.Module,
        optimizer: optim.Optimizer,
        train_metrics: MetricCollection,
        test_metrics: MetricCollection,
        scheduler: optim.lr_scheduler = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_metrics = train_metrics.clone(prefix='train/')
        self.val_metrics = test_metrics.clone(prefix='val/')
        self.test_metrics = test_metrics.clone(prefix='test/')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.optimizer(params=self.model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss',
                    'interval': 'epoch',
                    'frequency': 1,
                },
            }
        return {'optimizer': optimizer}
