#!/usr/bin/env python
import librosa
import numpy as np


class ProcessedRaw(object):
    def __init__(self, sf, cwlen, cwshift=None, max_chunks=None):
        self.sf = sf
        self.cwlen = cwlen
        self.cwshift = cwshift
        self.max_chunks = max_chunks

    def _get_chunks(self, signal):
        wlen = self.cwlen*self.sf
        slen = signal.shape[0]
        wshift = self.cwshift*self.sf

        # split signals into chunks
        beg_samp = 0
        end_samp = self.wlen

        sig_arr = np.zeros((self.max_chunks, wlen))
        count_fr = 0

        while end_samp < slen and count_fr < self.max_chunks:
            sig_arr[count_fr, :] = signal[beg_samp:end_samp]
            beg_samp = beg_samp+wshift
            end_samp = beg_samp+wlen
            count_fr += 1

        return sig_arr[:count_fr, :]

    def _get_sample(self, signal):
        # Get Random Chunk
        wlen = self.cwlen*self.sf
        slen = signal.shape[0]
        offs = np.random.randint(slen - wlen)
        raw_sample = signal[offs:]
        return raw_sample

    def load(self, path: str):
        signal = librosa.load(path, sr=self.sf)

        # Normalize
        signal /= np.abs(signal)

        if self.cwshift:
            return self._get_chunks(signal)
        else:
            return self._get_sample(signal)
