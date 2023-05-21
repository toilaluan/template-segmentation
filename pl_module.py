from typing import Any
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import JaccardIndex
from losses import DiceLoss


class SegmentationModule(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.iou_metric = JaccardIndex("binary")
        self.dice_loss = DiceLoss()

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            out = out.logits
        return out

    def compute_loss(self, out, y):
        return self.dice_loss(out, y) * self.args.loss_ratio + (
            1 - self.args.loss_ratio
        ) * F.binary_cross_entropy_with_logits(out, y)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.compute_loss(out, y)
        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.compute_loss(out, y)
        self.iou_metric.update(out, y)
        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("val_iou", self.iou_metric.compute(), prog_bar=True)
        self.iou_metric.reset()
