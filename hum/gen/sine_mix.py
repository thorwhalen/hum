import matplotlib.pylab as plt
from numpy import sin, arange, pi, ones, linspace, ndarray
from functools import partial

DFLT_N_SAMPLES = 21 * 2048
DFLT_SR = 44100


def mk_sine_wf(freq=5, n_samples=DFLT_N_SAMPLES, sr=DFLT_SR, phase=0, gain=1):
    """Make a sine waveform

    :param freq: Frequency (in Hz)
    :param n_samples: The number of samples of waveform you want
    :param sr: Sample rate
    :param phase: Phase (in radians)
    :param gain: (A number to multiply the base sine wave by)
    :return: Waveform. A numpy array of samples of the specified sine wave
    """
    return gain * sin(phase + arange(n_samples) * 2 * pi * freq / sr)


def freq_based_stationary_wf(freqs=(200, 400, 600, 800), weights=None,
                             n_samples: int = DFLT_N_SAMPLES, sr: int = DFLT_SR) -> ndarray:
    """
    Makes a stationary waveform by mixing a number of freqs together, possibly with different weights.

    :param freqs: List(-like) of frequencies (in Hz)
    :param weights: The weights these frequencies should have (all weights will be normalized
    :param n_samples: The number of samples of waveform you want
    :param sr: Sample rate
    :return: Waveform. A numpy array of samples of the specified sine wave
    """
    if weights is None:
        weights = ones(len(freqs))
    assert len(freqs) == len(weights)
    _mk_sine_wf = partial(mk_sine_wf, n_samples=n_samples, sr=sr)
    wf = sum(_mk_sine_wf(freq) * weights[i] for i, freq in enumerate(freqs))
    return wf / sum(weights)
