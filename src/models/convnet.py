#!/usr/bin/env python
from torch import nn
from src.utils.torch_utils import act_fun


class ConvNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.num_filters = hparams['num_filters']
        self.filter_lens = hparams['filter_lens']
        self.max_Pool_lens = hparams['cnn']['max_pool_lens']
        self.drop_probs = hparams['drop_probs']
        self.act_funs = hparams['act_funs']
        self.num_layers = len(self.num_filters)

        self.input_dim = hparams['input_dim']
        self.conv = nn.ModuleList([])
        self.lnorm = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

    def _init_layers(self):
        current_input = self.input_dim

        self.conv.append(SincConv_fast(
            self.cnn_N_filt[0], self.cnn_len_filt[0], self.fs))

        for i in range(self.num_layers):
            N_filt = int(self.num_filters[i])
            filt_len = int(self.filter_lens[i])
            max_pool_len = self.max_Pool_lens[i]

            # Dropout
            self.drop.append(nn.Dropout(p=self.drop_probs[i]))

            # activation
            self.act.append(act_fun[self.act_funs[i]])

            # layer norm initialization
            self.lnorm.append(nn.LayerNorm(
                [N_filt, int((current_input-filt_len+1)/max_pool_len)]))

            self.bn.append(nn.BatchNorm1d(
                N_filt, int((current_input-filt_len+1)/max_pool_len),
                momentum=0.05))

            if i != 0:
                self.conv.append(nn.Conv1d(
                    self.cnn_N_filt[i-1], N_filt, filt_len))

            current_input = int((current_input-filt_len+1)/max_pool_len)

        self.out_dim = current_input*N_filt
