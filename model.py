import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl


class Model(pl.LightningModule):

    def __init__(self):
        super().__init__()

    # Model computation and steps
    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self):
        pass

    def test_step(self):
        pass

    # Network configuration
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=self.step_size,
                                                 gamma=self.lr_decay)
        return [optimizer], [lr_scheduler]
