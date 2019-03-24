#!/usr/bin/env python
from math import ceil, log2


def nextpow2(x):
    return ceil(log2(abs(x)))
