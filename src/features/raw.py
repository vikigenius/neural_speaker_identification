#!/usr/bin/env python
import librosa
import scipy
import numpy as np


class ProcessedRaw(object):
    def __init__(self, sf, preprocess=True):
        self.sf = sf
        self.do_process = preprocess
        if self.sf == 16000:
            self.dc_alpha = 0.99
        elif self.sf == 8000:
            self.dc_alpha = 0.999
        else:
            raise ValueError('Only 16 and 8Khz supported')
        self.pe_alpha = 0.97

    def _preprocess(self, signal):
        # Remove DC component and add a small dither
        signal = scipy.signal.lfilter([1, -1], [1, -self.dc_alpha], signal)
        dither = np.random.random_sample(
            signal.shape) + np.random.random_sample(
            signal.shape) - 1
        spow = np.std(signal)
        signal = signal + 1e-6*spow*dither

        signal = scipy.signal.lfilter([1 - self.pe_alpha], 1, signal)
        return signal

    def _get_chunks(self, signal, cwlen, cwshift, max_chunks):
        wlen = int(self.cwlen*self.sf)
        slen = signal.shape[0]
        wshift = int(self.cwshift*self.sf)

        # split signals into chunks
        beg_samp = 0
        end_samp = wlen

        sig_arr = np.zeros((self.max_chunks, wlen))
        count_fr = 0

        while end_samp < slen and count_fr < max_chunks:
            sig_arr[count_fr, :] = signal[beg_samp:end_samp]
            beg_samp = beg_samp+wshift
            end_samp = beg_samp+wlen
            count_fr += 1

        return sig_arr[:count_fr, :]

    def _get_sample(self, signal, cwlen):
        # Get Random Chunk
        wlen = int(cwlen*self.sf)
        slen = signal.shape[0]
        offs = np.random.randint(slen - wlen)
        raw_sample = signal[offs:offs+wlen]
        return raw_sample

    def load(self, path: str):
        signal, _ = librosa.load(path, sr=self.sf)
        return signal

    def load_sample(self, path: str, cwlen, normalize=False):
        signal = self.load(path)
        signal = self._get_sample(signal, cwlen)
        # Normalize
        if normalize:
            signal /= np.abs(np.max(signal))
        if self.do_process:
            signal = self._preprocess(signal)
        return signal

    def load_chunks(self, path: str, cwlen, cwshift,
                    max_chunks, normalize=False):
        signal = self.load(path)
        return self._get_chunks(signal, cwlen, max_chunks)
