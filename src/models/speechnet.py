#!/usr/bin/env python
from torch import nn
from src.models.fcnet import FCNet
from src.models.sincnet import SincNet
from src.models.ops import CELoss


class SpeechNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.sincnet_params = hparams['cnn']
        self.fc_params = hparams['dnn']
        self.class_params = hparams['class']
        self.cost = nn.NLLLoss()
        self.loss_obj = CELoss
        self._init_layers()
        self.num_classes = hparams.num_classes

    def _init_layers(self):
        self.sincnet = SincNet(self.sincnet_params)

        self.fc_params['input_dim'] = self.sincnet.out_dim
        self.fc = FCNet(self.fc_params)

        self.class_params['input_dim'] = self.fc.layer_dims[-1]
        self.classifier = FCNet(self.class_params)

    def forward(self, batch):
        raw = batch['raw']
        x = self.sincnet(raw)
        x = self.fc(x)
        x = self.classifier(x)
        return x

    def loss(self, out, batch):
        tgt = batch['cid']
        loss = self.cost(out, tgt)
        metric = CELoss(loss, 1)
        return metric, loss
