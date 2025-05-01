#!/usr/bin/python

"""
Template for a RadioPyo song (version 1.0).

A RadioPyo song is a musical python script using the python-pyo
module to create the audio processing chain. You can connect to
the radio here : http://radiopyo.acaia.ca/

There is only a few rules:
    1 - It must be a one-page script.
    2 - No soundfile, only synthesis.
    3 - The script must be finite in time, with fade-in and fade-out
        to avoid clicks between pieces. Use the DURATION variable.

belangeo - 2014

"""
from pyo import *
import sys


TITLE = "Degradation"  # The title of the music
ARTIST = "belangeo"  # Your artist name
DURATION = 120  # The duration of the music in seconds


def dict_diff(a, b):
    """
    Return the elements of a that are not keys of b.

    >>> dict_diff({'a': 1, 'b': 2}, {'b': 3})
    {'a': 1}
    """
    return {k: a[k] for k in a if k not in b}


def processing_chain(params=None):
    pre_locals = locals()
    params = params or {}

    metro_rate = params.get("metro_rate", 0.1)
    degrade_bits = params.get("degrade_bits", 32)
    filt_freq = params.get("filter_freq", 5000)
    verb_fb = params.get("reverb_feedback", 0.75)

    fade = Fader(fadein=0.001, fadeout=10, dur=DURATION).play()

    env = CosTable([(0, 0), (40, 1), (500, 0.2), (8191, 0)])
    env2 = CosTable([(0, 0), (20, 1), (500, 1), (2000, 0.3), (8191, 0)])

    met = Metro(0.1, 8).play()

    car = TrigChoice(
        met.mix(),
        [50] * 12 + [75, 99] * 2 + [151],
        init=50,
        mul=RandInt(3, 0.3125, 2, 4),
    )
    ind = SampHold(Clip(Sine(0.02, 0, 3, 2), min=0, max=4), met, 1)
    amp = TrigEnv(met, env, 0.25, mul=[1, 0.5, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5])
    fm = FM(car, [0.25, 0.33, 0.5, 0.75], ind, amp)
    srscl = SampHold(Sine(0.05, 0, 0.06, 0.1), met.mix(), 1)
    deg = Degrade(fm, 32, srscl)
    filt = Biquad(deg.mix(2), freq=5000)

    low = Biquad(filt, freq=200, mul=0.3)
    b1 = Biquad(filt, freq=200, q=5, type=2)
    b2 = Biquad(filt, freq=500, q=5, type=2)
    b3 = Biquad(filt, freq=1000, q=5, type=2)
    b4 = Biquad(filt, freq=1700, q=8, type=2)
    b5 = Biquad(filt, freq=2500, q=8, type=2)

    delb1 = SDelay(b1, delay=0.8, maxdelay=1)
    delb3 = SDelay(b3, delay=1.6, maxdelay=2, mul=0.5)
    delb4 = SDelay(b4, delay=3.2, maxdelay=4, mul=0.75)
    delb5 = SDelay(b5, delay=6.4, maxdelay=7, mul=0.75)

    total = low + delb1 + b2 + delb3 + delb4 + delb5

    rev = WGVerb(total, feedback=0.75, bal=0.15, mul=fade * 1.3).out()

    rumvol = TrigEnv(met[0] + met[4], env2, 0.39, mul=0.4)
    rumble = Rossler(pitch=[0.09, 0.09003], chaos=0.25, mul=fade * rumvol).out()
    bass = Sine(freq=[40, 40], mul=fade * rumvol * 0.25).out()

    return dict_diff(locals(), pre_locals)


def compute_chain(chain_func, *, dur=DURATION, filename=None, fileformat=7):
    s = Server(duplex=0, audio="offline").boot()
    s.recordOptions(dur=dur, filename=filename, fileformat=fileformat)

    locals().update(chain_func())

    s.start()


if __name__ == "__main__":
    compute_chain(
        processing_chain,
        dur=DURATION,
        filename=sys.argv[1],
    )
