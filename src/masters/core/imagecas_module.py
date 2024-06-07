from typing import Dict

import torch

from masters.core.components.base_module import BaseLightningModule


class ImageCASLightningModule(BaseLightningModule):
    def sum_and_log_losses(self, loss: torch.Tensor | Dict[str, torch.Tensor], stage: str):
        if isinstance(loss, torch.Tensor):
            return loss

        log_dict = {f'{stage}/{name}': value for name, value in loss.items()}
        self.log_dict(log_dict, on_step=False, on_epoch=True)

        return sum(loss.values())

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self.forward(x)
        y_hat = torch.relu(y_hat)

        loss = self.loss_function(y_hat, y)
        loss = self.sum_and_log_losses(loss, stage='train')
        self.log('train/loss', loss, on_step=False, on_epoch=True)

        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self.forward(x)
        y_hat = torch.relu(y_hat)

        loss = self.loss_function(y_hat, y)
        loss = self.sum_and_log_losses(loss, stage='val')
        self.log('val/loss', loss, on_step=False, on_epoch=True)

        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self.forward(x)
        y_hat = torch.relu(y_hat)

        loss = self.loss_function(y_hat, y)
        loss = self.sum_and_log_losses(loss, stage='test')
        self.log('test/loss', loss, on_step=False, on_epoch=True)

        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx):
        x = batch['image']
        y_hat = self.forward(x)
        return {'prediction': torch.relu(y_hat)}

    # def log_images(self, y_hat, y, x, amount: int = 10):
    #     y_hat, y, x = torch.sigmoid(y_hat).squeeze(), y.float().squeeze(), x.squeeze()
    #     grid = make_grid([y_hat[:amount], y[:amount], x[:amount]])
    #     self.loggers[1].experiment.add_image('images', grid, 0)
