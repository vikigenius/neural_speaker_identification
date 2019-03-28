#!/usr/bin/env python
import logging
import numpy as np
import librosa
import scipy
from random import randint
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

    def _sample(self, signal, seqlen: int):
        """
        Helper function to sample a contiguos subsequence of
        length seqlen from signal
        Args:
            signal: numpy.ndarray, the signal
            seqlen: int, the sequence length
        Returns:
            numpy.ndarray, the sampled signal
        """
        nframes = len(signal)
        roffset = randint(0, nframes - seqlen)
        sampled = signal[roffset:roffset+seqlen]
        return sampled

    def _get_resampled_chunks(self, afile: str):
        """
        Takes in a string path afile and returns chunks of audio each
        representing a 16-bit mono channel with sampling rate = 16000

        Args:
            afile: path of audio file
        Returns:
            List[np.ndarray]
        """
        # Load the file
        signal, _ = librosa.load(afile, sr=self.sample_freq, mono=True)
        nframes = len(signal)
        duration = nframes/self.sample_freq
        if duration <= self.duration:
            logger.warn(f'Duration less than specified for {afile}')
        chunks = []
        if duration > 2*self.duration:
            # Can sample 2 chunks
            mid = int(nframes/2)
            chunks.append(signal[:mid])
            chunks.append(signal[mid:])
        else:
            chunks.append(signal)
        num_samples = int(self.duration*self.sample_freq)
        chunks = [self._sample(chunk, num_samples) for chunk in chunks]
        return chunks

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
        resampled_chunks = self._get_resampled_chunks(afile)
        if self.preprocess:
            processed = [self._preprocess(chunk) for chunk in resampled_chunks]
        else:
            processed = resampled_chunks

        # stft

        sf = self.sample_freq
        Tw = self.Tw  # Window size
        Ts = self.Ts

        Nw = round(1e-3*Tw*sf)
        Ns = round(1e-3*Ts*sf)
        n_fft = 2**nextpow2(Nw)

        spectrograms = [librosa.core.stft(
            chunk, n_fft=n_fft,
            hop_length=Ns, win_length=Nw,
            window=self.win_type) for chunk in processed]
        mag_specs = [np.abs(chunk) for chunk in spectrograms]
        return mag_specs
