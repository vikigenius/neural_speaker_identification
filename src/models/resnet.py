#!/usr/bin/env python
from torch import nn
from src.models.resnet_base import resnet50
from src.models.ops import CELoss


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
        if self.num_classes == 2:
            target = batch['gid']
        else:
            target = batch['cid']
        loss = self.criterion(model_outs, target)
        metric = CELoss(loss, 1)
        return metric, loss
