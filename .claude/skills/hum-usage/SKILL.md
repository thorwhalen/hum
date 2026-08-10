---
name: hum-usage
description: Use when you need to synthesize or manipulate audio in Python with `hum` — build a synthesizer from a function, change parameters ("knobs") in real time, record those changes as timestamped events, replay them, or render them to audio (WAV) bytes. Triggers on "make a synth", "play a tone/sine", "real-time audio parameters", "record/replay synth events", "render events to audio/WAV", "transpose/compose an event sequence". Requires the audio engine: `pip install hum[audio]` (pyo needs portaudio/portmidi).
---

# Using `hum`

`hum` wraps the `pyo` audio engine in a `Synth` that records every parameter
change as a timestamped event you can replay or render.

**Install:** `pip install hum[audio]` (the `audio` extra pulls `pyo`, which needs
the system libraries portaudio and portmidi). Plotting/chunking utilities work
without it, but anything involving `Synth` needs the audio engine.

## Build and play a synth

A synth is any function returning a pyo object; `Synth` wraps it:

```python
from pyo import Sine
from hum.pyo_util import Synth

s = Synth(lambda freq=220: Sine(freq=freq))
with s:  # context manager starts/stops the audio server
    s(freq=330)  # change a knob in real time  (or  s['freq'] = 330)
```

## Dials vs settings

Decorate to declare which params are smoothly-interpolated **dials** vs
graph-rebuilding **settings**:

```python
@Synth(dials="freq", settings="waveform")
def my_synth(freq=440, waveform="sine"): ...
```

## Record, replay, render

```python
events = s.get_recording()  # [(t, {param: value}), ...]
from hum.pyo_util import round_event_times

events = list(round_event_times(events, 0.1))  # tidy timestamps

s.replay_events(events)  # play them back
wav_bytes = s.render_events(events)  # render to WAV bytes (no real-time)
```

Event sequences are plain data — transpose, reverse, concatenate, or time-scale
them with ordinary Python before replaying/rendering. `ReplayEvents` (in
`hum.pyo_util`) iterates events with correct timing (`time_scale`, `emit_none`,
`ensure_sorted`).

## No audio engine?

`import hum` and `hum.plot_wf` / `hum.disp_wf` / `hum.simple_chunker` work
without `pyo`. Accessing `hum.Synth` without the `audio` extra raises an
ImportError telling you to install it.
