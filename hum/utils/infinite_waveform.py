"""
Simple infinite waveform
"""
import random
import numpy as np


class InfiniteWaveform(object):
    """
    A tiny little class emulating an infinite waveform with a given spectrum.
    Note that if choosing to add noise to the waveform, the generator will be considerably slower
    """

    def __init__(self, gen_spectrum, noise_amp=None):
        """
        noise_amp is the amplitude of the random noise added to the waveform,
        proportionally to the range of amplitude in the signal
        """

        self.reconstituted_wf = np.fft.irfft(gen_spectrum)
        self.noise_amp = noise_amp
        if noise_amp:
            max_signal = np.max(self.reconstituted_wf)
            min_signal = np.min(self.reconstituted_wf)
            max_amp = max_signal - min_signal
            self.noise_amp = max_amp * noise_amp
        else:
            self.noise_amp = 0
        self.win_size = len(self.reconstituted_wf)

    def query(self, bt, tt):
        """
        Returns a generator of the infinite waveform from bt to tt
        """
        if not self.noise_amp:
            for i in range(bt, tt):
                yield self.reconstituted_wf[i % self.win_size]
        else:
            # a random seed is chosen for each i, this allows for consistency between queries
            # the drawback is that it slows down the iterator quite a bit: 30 times slower for 1 millions samples
            for i in range(bt, tt):
                random.seed(a=i)
                noise = random.random() * self.noise_amp
                yield self.reconstituted_wf[i % self.win_size] + noise

    def __getitem__(self, idx):
        random.seed(a=idx)
        return (
            self.reconstituted_wf[idx % self.win_size]
            + random.random() * self.noise_amp
        )
