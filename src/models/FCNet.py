#!/usr/bin/env python
import torch
import numpy as np
from torch import nn
from src.models.ops import Normalize


class FCNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.input_dim = hparams['input_dim']
        self.layer_dims = hparams['layer_dims']

        self.input_normalization = hparams['input_normalization']

        self.drop_probs = hparams['drop_probs']
        self.act_funs = hparams['act_funs']
        self.num_layers = len(self.layer_dims)

        self.normalization = hparams['normalization']

        self.norm = nn.ModuleList([])
        self.wx = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

    def _init_layers(self):
        if self.input_normalization:
            self.norm0 = Normalize(self.input_dim)

        current_input = self.input_dim

        for i in range(self.num_layers):
            # Dropout
            self.drop.append(nn.Dropout(p=self.drop_probs[i]))

            # Activation
            self.act.append(nn.Dropout(p=self.drop_probs[i]))

            add_bias = True

            # Layer Norm Initialization
            if self.normalization:
                self.norm.append(Normalize(current_input))
                add_bias = False

            # Linear Operation
            self.wx.append(nn.Linear(
                current_input, self.layer_dims[i], bias=add_bias))

            # weight initialization
            start = -np.sqrt(0.01/(current_input + self.layer_dims[i]))
            end = np.sqrt(0.01/(current_input + self.layer_dims[i]))
            self.wx[i].weight = torch.nn.Parameter(torch.Tensor(
                self.layer_dims[i], current_input).uniform_(start, end))
            self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.fc_lay[i]))

            current_input = self.layer_dims[i]

    def forward(self, x):
        if self.input_normalization:
            x = self.norm0(self.input_normalization, x)

        for i in range(self.num_layers):
            x = self.wx[i](x)

            if self.normalization:
                x = self.norm(self.normalization, x)

            x = self.act[i](x)

            x = self.drop[i](x)

        return x
