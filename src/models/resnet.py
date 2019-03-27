#!/usr/bin/env python
from dataclasses import dataclass
from torch import nn
from src.models.resnet_base import resnet50


@dataclass
class CELoss:
    ce_loss: float = 0.0
    num_obs: int = 0

    def update(self, loss: 'CELoss'):
        ce_loss = loss.ce_loss
        n = self.num_obs
        self.num_obs += 1
        self.ce_loss = self.ce_loss*(n/(n+1)) + ce_loss/(n+1)

    def pbardict(self):
        return {
            'ce_loss': format(self.ce_loss, '.4f')
        }


class ResnetM(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.num_classes = hparams.num_classes
        self.base = resnet50(num_classes=self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.loss_obj = CELoss

    def forward(self, batch):
        input_mat = batch['sgram'].unsqueeze(1)
        return self.base(input_mat)

    def loss(self, model_outs, batch):
        target = batch['cid']
        loss = self.criterion(model_outs, target)
        metric = CELoss(loss, 1)
        return metric, loss
