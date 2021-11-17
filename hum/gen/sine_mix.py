"""
Utils to mix sine waves
"""
from numpy import sin, arange, pi, ones, ndarray, random, all
from hum.gen.util import DFLT_N_SAMPLES, DFLT_SR


def mk_sine_wf(freq=5, n_samples=DFLT_N_SAMPLES, sr=DFLT_SR, phase=0, gain=1):
    """Make a sine waveform

    :param freq: Frequency (in Hz)
    :param n_samples: The number of samples of waveform you want
    :param sr: Sample rate
    :param phase: Phase (in radians)
    :param gain: (A number to multiply the base sine wave by)
    :return: Waveform. A numpy array of samples of the specified sine wave

    >>> n_samples = random.randint(2,5)
    >>> wf = mk_sine_wf(n_samples=n_samples)
    >>> assert len(wf) == n_samples
    >>> wf = mk_sine_wf(n_samples=3)
    >>> assert all(wf == [0.0, 0.000712379226274755, 0.0014247580910282892])
    """
    return gain * sin(phase + arange(n_samples) * 2 * pi * freq / sr)


def freq_based_stationary_wf(
    freqs=(200, 400, 600, 800),
    weights=None,
    n_samples: int = DFLT_N_SAMPLES,
    sr: int = DFLT_SR,
) -> ndarray:
    """
    Makes a stationary waveform by mixing a number of freqs together,
    possibly with different weights.

    :param freqs: List(-like) of frequencies (in Hz)
    :param weights: The weights these frequencies should have (all weights will be normalized
    :param n_samples: The number of samples of waveform you want
    :param sr: Sample rate
    :return: Waveform. A numpy array of samples of the specified sine wave

    >>> n_samples = random.randint(2,5)
    >>> wf = freq_based_stationary_wf(n_samples=n_samples)
    >>> assert len(wf) == n_samples
    >>> wf = freq_based_stationary_wf(n_samples = 3, weights = [1,2,3,4])
    >>> assert all(wf == [0.0, 0.08534908048813569, 0.16988139234280178])
    """
    if weights is None:
        weights = ones(len(freqs))
    assert len(freqs) == len(weights)
    _mk_sine_wf = partial(mk_sine_wf, n_samples=n_samples, sr=sr)
    wf = sum(_mk_sine_wf(freq) * weights[i] for i, freq in enumerate(freqs))
    return wf / sum(weights)


#################################################################################
# Soft marking everything below for deprecation
import random
from typing import Callable, Union, Tuple, Optional, Iterable
from functools import partial
from dataclasses import dataclass
from numbers import Number
from itertools import islice


@dataclass
class MinMaxRand:
    """Like a partial, but meant for bounded rand functions.
    Could have just done:

    ```
    r = partial(random.uniform, min, max)
    r()
    ```

    But we need to be able to have access to r.min and r.max
    """

    min: Number = 0
    max: Number = 1
    rand_func: Callable[[Number, Number], Number] = random.uniform

    def __call__(self):
        return self.rand_func(self.min, self.max)


@dataclass
class MinMaxRandDict:
    """
    >>> from hum.gen.sine_mix import MinMaxRandDict, MinMaxRand
    >>>
    >>> rand_gen = MinMaxRandDict((
    ...     ('rpm', MinMaxRand(100, 1000)),
    ...     ('temperature', MinMaxRand(10, 25)),
    ...     ('pressure', MinMaxRand(50, 500)),
    ... ))
    >>> t = list(rand_gen.read(4))
    >>> assert len(t) == 4
    >>> assert list(t[0]) == ['rpm', 'temperature', 'pressure']
    >>> t  # doctest: +SKIP
    [{'rpm': 577.8533852127333,
      'temperature': 18.306495707512724,
      'pressure': 357.72481026748113},
     {'rpm': 457.0605450014562,
      'temperature': 23.94969543436332,
      'pressure': 483.1119334545017},
     {'rpm': 489.8445225473431,
      'temperature': 17.176275589569695,
      'pressure': 415.8337855923849},
     {'rpm': 120.84000962564362,
      'temperature': 21.153044572476936,
      'pressure': 449.5124385844328}]
    """

    iid_seed_gen: Union[dict, Tuple[Tuple[str, MinMaxRand]]] = (
        ('rpm', MinMaxRand(100, 1000)),
        ('temperature', MinMaxRand(10, 25)),
        ('pressure', MinMaxRand(50, 500)),
    )

    def __post_init__(self):
        self.iid_seed_gen = dict(self.iid_seed_gen)

    def __call__(self, n=1):
        if n == 1:
            return {k: v() for k, v in self.iid_seed_gen.items()}
        else:
            return self.read(n)

    def __iter__(self):
        while True:
            yield self()

    def read(self, n=1):
        return islice(self, n)


def dflt_wf_params_to_wf(
    weights, freqs=None, n_samples: int = DFLT_N_SAMPLES, sr: int = DFLT_SR,
):
    if freqs is None:
        freqs = tuple(range(200, (len(weights) + 1) * 200, 200))
    return freq_based_stationary_wf(freqs, weights, n_samples, sr)


def dflt_sprout_to_wf_params(annot):
    # TODO: Need to include the possibility of normalization
    params = tuple(annot.values())
    return params


def asis(x):
    return x


# def numerical_annotations_and_waveform_chunks(

# TODO: Replicate in slink
@dataclass
class NumAnnotsAndWaveformChunks:
    seeds: Union[dict, Tuple[Tuple[str, MinMaxRand]]] = (
        ('rpm', MinMaxRand(100, 1000)),
        ('temperature', MinMaxRand(10, 25)),
        ('pressure', MinMaxRand(50, 500)),
    )
    seed_to_sprout_gen: Callable = MinMaxRandDict
    sprout_to_annot: Callable = asis
    sprout_to_wf_params: Callable = dflt_sprout_to_wf_params
    wf_params_to_wf: Callable = dflt_wf_params_to_wf

    def __post_init__(self):
        self.sprout_gen = self.seed_to_sprout_gen(self.seeds)

    def __call__(self, *args, **kwargs):
        sprout = self.sprout_gen()
        annot = self.sprout_to_annot(sprout)
        params = self.sprout_to_wf_params(annot)
        wf = self.wf_params_to_wf(params)
        return annot, wf

    def __iter__(self):
        while True:
            yield self()

    def read(self, n=1):
        return islice(self, n)


##########
