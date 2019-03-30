#!/usr/bin/env python
import torch
from torch import nn


class SincNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()

    def _init_layers(self):

        for layer in range(self.num_conv_layers):
