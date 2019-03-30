#!/usr/bin/env python
import torch
import torch.nn.functional as F
from torch import nn
from src.models.ops import act_fun, Normalize
from src.models.sinc_conv_fast import SincConv_fast


class SincNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.num_filters = hparams['num_filters']
        self.filter_lens = hparams['filter_lens']
        self.max_pool_lens = hparams['max_pool_lens']
        self.drop_probs = hparams['drop_probs']
        self.act_funs = hparams['act_funs']
        self.num_layers = len(self.num_filters)

        self.input_dim = hparams['input_dim']
        self.input_normalization = hparams['input_normalization']

        self.conv = nn.ModuleList([])

        self.normalization = hparams['normalization']
        self.norm = nn.ModuleList([])

        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

    def _init_layers(self):
        if self.input_normalization:
            self.norm0 = Normalize(self.input_dim)

        current_input = self.input_dim

        self.conv.append(SincConv_fast(
            self.cnn_N_filt[0], self.cnn_len_filt[0], self.fs))

        for i in range(self.num_layers):
            N_filt = int(self.num_filters[i])
            filt_len = int(self.filter_lens[i])
            max_pool_len = self.max_pool_lens[i]

            # Dropout
            self.drop.append(nn.Dropout(p=self.drop_probs[i]))

            # activation
            self.act.append(act_fun[self.act_funs[i]])

            # layer norm initialization
            self.norm.append(nn.LayerNorm(
                [N_filt, int((current_input-filt_len+1)/max_pool_len)]))

            if i != 0:
                self.conv.append(nn.Conv1d(
                    self.cnn_N_filt[i-1], N_filt, filt_len))

            current_input = int((current_input-filt_len+1)/max_pool_len)

        self.out_dim = current_input*N_filt

    def forward(self, x):
        batch = x.shape[0]
        seq_len = x.shape[1]

        if self.input_normalization:
            x = self.norm0(self.input_normalization, x)

        x = x.view(batch, 1, seq_len)

        for i in range(self.num_layers):
            x = torch.abs(self.conv[i](x))
            x = F.max_pool1d(x, self.max_pool_lens[i])

            if self.normalization:
                x = self.norm(self.normalization, x)

            x = self.act[i](x)

            x = self.drop[i](x)

        x = x.view(batch, -1)
        return x
