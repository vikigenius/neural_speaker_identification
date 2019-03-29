#!/usr/bin/env python
import librosa
import numpy as np


class ProcessedRaw(object):
    def __init__(self, cwlen, sf, daf=0.2):
        self.sf = sf
        self.cwlen = cwlen
        self.daf = daf

    def load(self, path: str):
        signal = librosa.load(path, sr=self.sf)

        # Normalize
        signal /= np.abs(signal)

        # Get Random Chunk
        wlen = self.cwlen*self.sf
        slen = signal.shape[0]
        offs = np.random.randint(slen - wlen)

        raw_chunk = signal[offs:]

        af = np.random.uniform(1.0 - self.daf, 1.0 + self.daf)

        return raw_chunk*af
