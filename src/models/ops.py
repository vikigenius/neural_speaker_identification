#!/usr/bin/env python
from torch import nn
from dataclasses import dataclass


class Normalize(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.ModuleDict({
            'layer': nn.LayerNorm(num_features),
            'batch': nn.BatchNorm1d(num_features, momentum=0.05)
        })

    def forward(self, features, norm_type):
        return self.norm[norm_type](features)


activations = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'linear': lambda x: x
}


def act_fun(act: str):
    if act in activations:
        return activations[act]
    else:
        raise f'Unsupported Activation {act}, options are {activations.keys()}'


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
