#!/usr/bin/env python
import librosa
import numpy as np


class SpeechFeatures(object):
    def __init__(self, cwlen, cwshift, sf):
        self.sf = sf

    def _preprocess(self):
        pass

    def _load(self, afile):
        signal, _ = librosa.load(afile, sr=self.sf)
        return signal

    def load_raw(self, preprocess=True):
        pass

    def load_sgram(self, preprocess=False):
        pass
