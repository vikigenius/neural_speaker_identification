#!/usr/bin/env python
import torch
from torch.autograd import Variable


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
