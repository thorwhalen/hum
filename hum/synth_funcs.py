"""Synth funcs"""

from pyo import *
from hum.pyo_util import add_synth_defaults, add_default_dials, add_default_settings


def sine(freq=440, volume=0.3):
    return Sine(freq=freq, mul=volume)


def pulse(freq=440, duty=0.5, volume=0.3):
    return LFO(freq=freq, sharp=duty, type=2, mul=volume)


@add_synth_defaults(settings="type")
def noise(cutoff=1000, q=1, volume=0.2, type=0):
    noise = Noise(mul=volume)
    return Biquadx(noise, freq=cutoff, q=q, type=type, stages=2)


@add_synth_defaults(dials="freq")
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


@add_default_dials(["freq", "cutoff", "reso", "decay", "mult"])
@add_default_settings(["shape"])
def ankg_fm303(freq=110, cutoff=2000, reso=5, decay=0.5, mult=0.1, shape=1):
    decay_val = decay.value if isinstance(decay, SigTo) else decay
    mult_val = mult.value if isinstance(mult, SigTo) else mult
    freq_val = freq.value if isinstance(freq, SigTo) else freq

    env = Adsr(
        attack=0.05, decay=decay_val, sustain=0.1, release=0.05, dur=0.2 + decay_val
    )
    env.play()  # This line is crucial

    env2 = env * mult_val
    env2.play()
    wave1 = Phasor(freq=freq_val, phase=0, mul=1)
    wave2 = Phasor(freq=-freq_val, phase=0.5, mul=1)

    if shape == 0:
        wave = Phasor(freq=freq_val, phase=0, mul=env2)
    else:
        wave = ((wave1 + wave2) - 1) * env2

    filt = Biquadx([wave, wave], freq=cutoff, q=reso, type=0, stages=2)
    return filt


@add_synth_defaults(dials="IpulseHz")
def intro_crunch_pulsar(IpulseHz=110):
    wavPas = DataTable(size=256)  # default dummy waveform
    envPulse = LinTable([(0, 1), (8191, 1)])
    pas = Pulsar(
        table=wavPas,
        env=envPulse,
        freq=IpulseHz,
        frac=[IpulseHz * 0.5, IpulseHz * 0.48],
        interp=4,
        mul=1.25,
    )
    rezons = Waveguide(pas, freq=[32.7032, 65.4064, 98.1, 81.758], dur=30, mul=0.1)
    rezfilt = Tone(rezons, freq=1500)
    rev = WGVerb(pas, feedback=0.99, cutoff=10000, bal=0.25, mul=0.75)
    return rezfilt + rev


@add_synth_defaults(dials=["base_pitch", "spread"])
def intro_buzz_pulsar(base_pitch=60, spread=0.007):
    wav = HarmTable([0.1, 0, 0.2, 0, 0.1, 0, 0, 0, 0.04, 0, 0, 0, 0.02])
    env = HannTable()
    fade = Fader(fadein=4, fadeout=60, mul=1)
    pitches = [midiToHz(base_pitch + i * spread) for i in range(6)]
    buzz = Pulsar(table=wav, env=env, freq=pitches, frac=0.7, mul=fade * 0.05)
    mix = Mix(buzz, voices=2)
    rev = Freeverb(mix, size=1.0, damp=0.25, bal=0.75)
    return rev


def intro_high_sines(base_freq=4000, mod_freq=0.4, mod_mul=0.005):
    wav = HarmTable([0.1, 0, 0.2, 0, 0.1, 0, 0, 0, 0.04, 0, 0, 0, 0.02])
    fade = Fader(fadein=3, fadeout=20, mul=mod_mul)
    fade.play()
    mod = Osc(table=wav, freq=mod_freq)
    car = Sine(freq=[base_freq, base_freq + 40, base_freq - 10], mul=mod)
    pan = SPan(car, pan=[0, 0.5, 1], mul=fade)
    return pan
