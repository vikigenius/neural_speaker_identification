#!/usr/bin/env python
from torch import nn
import torch
import torch.nn.functional as F
from src.models.resnet_base import resnet50
from src.models.ops import CELoss, Identity
from src.utils.math_utils import nextpow2
from src.utils import torch_utils


class SpecNet(nn.Module):
    def __init__(self, num_classes, sf, win_size, hop_len,
                 window=torch.hamming_window):
        super().__init__()
        self.num_classes = num_classes
        self.base = resnet50(pretrained=True)
        self.base.fc = Identity()
        self.base.avgpool = Identity()
        self.fc = nn.Linear(9, 1)
        self.classifier = nn.Linear(2048, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.loss_obj = CELoss
        self.sf = sf
        self.win_length = round(1e-3*win_size*self.sf)
        self.hop_length = round(1e-3*hop_len*self.sf)
        self.n_fft = 2**nextpow2(self.win_length)
        self.hop_len = hop_len

        self.window = window(self.win_length, device=torch_utils.device)

    def spectrogram(self, signal: torch.Tensor):
        window = self.window
        spec = torch.stft(signal, self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length, window=window)
        mag_spec = spec.pow(2).sum(-1)  # Mag Spectrogram
        if mag_spec.size(1) != 257:  # Debug
            raise RuntimeError(
                f'Expected SPEC size 257, got {mag_spec.size(2)}')
        spec_mean = mag_spec.mean(2, keepdim=True)
        spec_std = mag_spec.std(2, keepdim=True)
        mag_spec -= spec_mean
        mag_spec /= spec_std
        return mag_spec.to(torch.float)

    def forward(self, batch):
        signal = batch['raw']
        spec = self.spectrogram(signal).unsqueeze(1)
        spec = spec.repeat(1, 3, 1, 1)  # Convert to RGB
        resout = self.base(spec)
        x = resout.permute(0, 1, 3, 2)
        x = self.fc(x)
        x = x.permute(0, 1, 3, 2)
        width = x.size(3)
        x = F.avg_pool2d(x, (1, width))
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def loss(self, model_outs, batch):
        if self.num_classes == 2:
            target = batch['gid']
        else:
            target = batch['cid']
        loss = self.criterion(model_outs, target)
        metric = CELoss(loss, 1)
        return metric, loss
