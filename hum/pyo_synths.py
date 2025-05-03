"""A collection of pyo synths"""

from pyo import LFO, Noise, Biquadx, Adsr, Sine
from hum.pyo_util import Synth


@Synth
def sine(freq=440, volume=0.3):
    return Sine(freq=freq, mul=volume)


@Synth
def pulse(freq=440, duty=0.5, volume=0.3):
    return LFO(freq=freq, sharp=duty, type=2, mul=volume)


@Synth(settings="type")
def noise(cutoff=1000, q=1, volume=0.2, type=0):
    noise = Noise(mul=volume)
    return Biquadx(noise, freq=cutoff, q=q, type=type, stages=2)


@Synth(dials="freq")
def simple_waveforms(freq=440, attack=0.01, waveform="sine"):
    env = Adsr(
        attack=attack, decay=0.1, sustain=0.8, release=0.1, dur=0, mul=0.5
    ).play()
    wave = {
        "sine": Sine,
        "triangle": lambda freq, mul: LFO(freq=freq, type=3, mul=mul),
        "square": lambda freq, mul: LFO(freq=freq, type=1, mul=mul),
    }.get(waveform, Sine)
    return wave(freq=freq, mul=env)
