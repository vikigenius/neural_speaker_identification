#!/usr/bin/env python
import logging
import numpy as np
import librosa
import scipy
from random import uniform
from src.utils.math_utils import nextpow2


logger = logging.getLogger(__name__)


class Spectrum(object):
    def __init__(self, hparams):
        self.sample_freq = hparams.sample_freq
        self.duration = hparams.duration
        self.preprocess = hparams.preprocess
        self.Tw = hparams.window_size
        self.Ts = hparams.window_shift
        self.win_type = hparams.window_type

        if self.sample_freq == 16000:
            self.dc_alpha = 0.99
        elif self.sample_freq == 8000:
            self.dc_alpha = 0.999
        else:
            raise ValueError('Only 16 and 8Khz supported')

        self.pe_alpha = 0.97

    def _resample_audio(self, afile: str) -> np.ndarray:
        """
        Takes in a string path afile and returns a numpy nd array
        representing a 16-bit mono channel with sampling rate = 16000
        after truncating the audio according to self.duration

        Args:
            afile: path of audio file
        Returns:
            numpy.ndarray
        """
        # Load the file
        duration = librosa.get_duration(filename=afile)
        if duration <= 3.1:
            logger.warn(f'Duration < 3.1 for {afile}')
        roffset = uniform(0.0, duration - 3.1)
        data, _ = librosa.load(afile, sr=self.sample_freq, mono=True,
                               offset=roffset, duration=self.duration)
        return data

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

    def generate(self, afile: str):
        """
        Takes in a string path afile and returns a numpy nd array
        representing the magnitude spectrum of the signal

        Args:
            afile: path of audio file
        Returns:
            numpy.ndarray
        """
        resampled = self._resample_audio(afile)
        preprocessed = resampled
        if self.preprocess:
            preprocessed = self._preprocess(resampled)

        # sfft

        sf = self.sample_freq
        Tw = self.Tw  # Window size
        Ts = self.Ts

        Nw = round(1e-3*Tw*sf)
        Ns = round(1e-3*Ts*sf)
        n_fft = 2**nextpow2(Nw)
        spec1 = librosa.core.stft(preprocessed, n_fft=n_fft, hop_length=Ns,
                                  win_length=Nw, window=self.win_type)
        return np.abs(spec1)
