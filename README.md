# hum

A python synthesizer for creating and manipulating audio signals.

To install:	```pip install hum```

Note: Some functionalities depend on [pyo](https://belangeo.github.io/pyo/download.html), 
which itself requires some tools  (namely, [portaudio](https://www.portaudio.com/) 
and [Portmidi](https://github.com/PortMidi/portmidi)) that _may_ have to be 
installed manually.

## Overview

The `Synth` class is a powerful wrapper around Pyo's audio engine, providing:

1. **Real-time parameter control** - Change audio parameters (knobs) during playback
2. **Event recording** - Automatically record all parameter changes with timestamps
3. **Event playback** - Replay recorded events exactly as they happened
4. **Event rendering** - Convert recorded events into audio files
5. **Context manager** - Clean resource management with `with` statements

## Getting Started with Synth

```python
from hum.pyo_util import Synth
```

## Simple Synth Test

Let's start with a simple sine wave synthesizer that demonstrates the basic functionality:

```python
import time
from pyo import Sine
import recode 
from hum.pyo_util import Synth, DFLT_PYO_SR, round_event_times
from hum.extra_util import estimate_frequencies

base_freq = 220

# Make a synth that plays a simple sine wave
def simple_sine(freq=base_freq):
    return Sine(freq=freq)

s = Synth(simple_sine)

# Define a sequence of frequencies to play
freq_sequence = [base_freq] + [base_freq * 3 / 2, base_freq * 2]

# Play the frequencies in sequence
with s:
    time.sleep(1)  # let base_freq play for a second
    s(freq=freq_sequence[1])  # Change to 330 Hz
    time.sleep(1)  # play that for a second
    s['freq'] = freq_sequence[2]  # Change to 440 Hz (alternative syntax)
    time.sleep(1)  # play that for a second
    # The context manager exits here, which stops the synth
```

After playing the synth, you can retrieve the recorded events:

```python
# Get the recorded events
events = s.get_recording()

# Round timestamps for better readability
events = list(round_event_times(events, round_to=0.1))

# These are the events we expect to see:
expected_events = [
    (
        0.0,
        {
            # First event contains the synth's initial parameters
            'freq': {'value': 220, 'time': 0.025, 'mul': 1, 'add': 0},
        },
    ),
    (1.0, {'freq': 330.0}),  # Second event at 1.0 seconds
    (2.0, {'freq': 440}),    # Third event at 2.0 seconds
    (3.0, {}),               # Final event marks the end of recording
]

# Verify that we got what we expected
assert events == expected_events
```

You can render these events to audio:

```python
# Render events to WAV format bytes
wav_bytes = s.render_events(events)

# Decode and verify the WAV bytes
wf, sr = recode.decode_wav_bytes(wav_bytes)
assert sr == DFLT_PYO_SR
total_duration = len(wf) / sr
assert abs(total_duration - 3.0) < 0.1
```

This simple example demonstrates the core functionality of the `Synth` class:
- Creating a synthesizer with a function that returns a Pyo object
- Playing the synth with real-time parameter changes
- Recording parameter changes as events
- Rendering events to audio

## Interactive Usage in a REPL

For more interactive work, you can use the `Synth` class in a REPL environment. This allows you to experiment with different parameters in real-time.

### Parameter Types: Dials vs Settings

The `Synth` class distinguishes between two types of parameters:

1. **Dials** - Real-time controllable parameters that use `pyo.SigTo` for smooth transitions
2. **Settings** - Parameters that require rebuilding the synthesis graph when changed

By default, all parameters are treated as dials. You can specify which parameters should be dials or settings using the decorator syntax:

```python
from pyo import LFO, Adsr, Sine
from hum.pyo_util import Synth
from time import sleep

# Note that here we explicitly tell Synth what arguments are "dials" and what are "settings"
@Synth(dials='freq', settings='waveform attack')
def simple_waveform_synth(freq=440, attack=0.01, waveform='sine'):
    env = Adsr(attack=attack, decay=0.1, sustain=0.8, release=0.1, dur=0, mul=0.5).play()
    wave = {
        'sine': Sine,
        'triangle': lambda freq, mul: LFO(freq=freq, type=3, mul=mul),
        'square': lambda freq, mul: LFO(freq=freq, type=1, mul=mul),
    }.get(waveform, Sine)
    return wave(freq=freq, mul=env)
```

In this example:
- `freq` is a dial - It changes smoothly in real-time
- `waveform` and `attack` are settings - Changing them rebuilds the synth

### Interactive Control

For interactive control in a REPL, you need to explicitly start and stop the synth:

```python
# Start the synth (begins making sound)
simple_waveform_synth.start()

# Change the frequency
simple_waveform_synth(freq=440 * 3 / 2)  # Change to 660 Hz
# The sound continues to play, but now at a different frequency

# Change multiple parameters at once
simple_waveform_synth(freq=440, waveform='triangle')
# Now playing a triangle wave at 440 Hz

# Change the waveform and attack time
simple_waveform_synth(waveform='square', attack=0.5)
# Note: You won't hear the change in attack time until the next note is played!

# Stop the synth when done
simple_waveform_synth.stop()
```

Note that when you restart a synth, it retains its last state:

```python
# Start the synth again - it will use the last parameter values
simple_waveform_synth.start()  # Still has square waveform and attack=0.5

# Change back to sine wave
simple_waveform_synth(waveform='sine')  # Note the attack is still 0.5

# Stop the synth when done
simple_waveform_synth.stop()
```

> **Warning**: Always remember to stop your synths when done to avoid resource issues. Using the context manager approach is recommended for automatic cleanup.

## Precomputed Knob Changes

Instead of interactive control, you can also precompute a sequence of parameter changes:

```python
from pyo import LFO, Adsr, Sine
from hum.pyo_util import Synth
from time import sleep

@Synth(dials='freq', settings='waveform attack')
def simple_waveform_synth(freq=440, attack=0.01, waveform='sine'):
    env = Adsr(attack=attack, decay=0.1, sustain=0.8, release=0.1, dur=0, mul=0.5).play()
    wave = {
        'sine': Sine,
        'triangle': lambda freq, mul: LFO(freq=freq, type=3, mul=mul),
        'square': lambda freq, mul: LFO(freq=freq, type=1, mul=mul),
    }.get(waveform, Sine)
    return wave(freq=freq, mul=env)

with simple_waveform_synth as s:
    sleep(1)  # Play default settings for a second
    s(freq=440 * 3 / 2)  # Change to 660 Hz
    sleep(1) 
    s(freq=440, waveform='triangle')  # Change to triangle wave at 440 Hz
    sleep(0.5)  # Shorter wait this time
    s(waveform='square', attack=0.5)  # Change to square wave with longer attack
    sleep(2)  # Wait a bit longer
    s(waveform='sine')  # Change back to sine wave (attack still 0.5)
    sleep(1)  # Wait for the final change to be heard
```

## Recording and Analyzing Events

The `Synth` class automatically records all parameter changes with timestamps:

```python
# Get the recorded events after playing
events = simple_waveform_synth.get_recording()
```

This gives you a list of timestamped parameter changes:

```python
[
  (0,
   {'freq': {'value': 440, 'time': 0.025, 'mul': 1, 'add': 0},
    'attack': 0.01,
    'waveform': 'sine'}),
  (1.0052499771118164, {'freq': 660.0}),
  (2.007974863052368, {'freq': 440, 'waveform': 'triangle'}),
  (2.0084967613220215, {'waveform': 'triangle'}),
  (2.515920877456665, {'waveform': 'square', 'attack': 0.5}),
  (2.518988609313965, {'waveform': 'square', 'attack': 0.5}),
  (4.524500846862793, {'waveform': 'sine'}),
  (4.525563716888428, {'waveform': 'sine'}),
  (5.528840780258179, {})
]
```

You can round the timestamps for better readability:

```python
from hum.pyo_util import round_event_times

events = list(round_event_times(simple_waveform_synth.get_recording(), 0.1))
```

This gives you a cleaner view:

```python
[
  (0.0,
   {'freq': {'value': 440, 'time': 0.025, 'mul': 1, 'add': 0},
    'attack': 0.01,
    'waveform': 'sine'}),
  (1.0, {'freq': 660.0}),
  (2.0, {'freq': 440, 'waveform': 'triangle'}),
  (2.0, {'waveform': 'triangle'}),
  (2.5, {'waveform': 'square', 'attack': 0.5}),
  (2.5, {'waveform': 'square', 'attack': 0.5}),
  (4.5, {'waveform': 'sine'}),
  (4.5, {'waveform': 'sine'}),
  (5.5, {})
]
```

## Composing with Event Sequences

You can create, edit, or manipulate event sequences directly:

```python
# Manually define an event sequence
events = [
    (
        0.0,
        {
            'freq': {'value': 440, 'time': 0.025, 'mul': 1, 'add': 0},
            'attack': 0.01,
            'waveform': 'sine',
        },
    ),
    (1.0, {'freq': 660.0}),
    (2.0, {'freq': 440, 'waveform': 'triangle'}),
    (2.0, {'waveform': 'triangle'}),
    (2.5, {'waveform': 'square', 'attack': 0.5}),
    (2.5, {'waveform': 'square', 'attack': 0.5}),
    (4.5, {'waveform': 'sine'}),
    (4.5, {'waveform': 'sine'}),
    (5.5, {}), 
]
```

### Replaying Events

You can replay any event sequence through a compatible synth:

```python
@Synth(dials='freq', settings='waveform attack')
def simple_waveform_synth(freq=440, attack=0.01, waveform='sine'):
    env = Adsr(attack=attack, decay=0.1, sustain=0.8, release=0.1, dur=0, mul=0.5).play()
    wave = {
        'sine': Sine,
        'triangle': lambda freq, mul: LFO(freq=freq, type=3, mul=mul),
        'square': lambda freq, mul: LFO(freq=freq, type=1, mul=mul),
    }.get(waveform, Sine)
    return wave(freq=freq, mul=env)

# Replay the event sequence
simple_waveform_synth.replay_events(events)
```

### Rendering Events to Audio

You can render events to audio without real-time playback:

```python
import recode 

# Render events to WAV format bytes
wav_bytes = simple_waveform_synth.render_events(events)

# Decode the WAV bytes for analysis or saving
wf, sr = recode.decode_wav_bytes(wav_bytes)

# Visualize the waveform
from hum import disp_wf 
disp_wf(wf, sr)
```

## Advanced Features

### ReplayEvents Class

The `ReplayEvents` class is a utility for replaying timestamped events with proper timing:

```python
from hum.pyo_util import ReplayEvents

# Get your events, either from a recording or created manually
events = [
    (0.0, {'freq': {'value': 440, 'time': 0.025, 'mul': 1, 'add': 0}}),
    (1.0, {'freq': 330.0}),
    (2.0, {'freq': 440}),
    (3.0, {})
]

# Create a replay generator and iterate through it
for knob_update in ReplayEvents(events):
    print(f"Update: {knob_update}")
    # Do something with each update
```

The `ReplayEvents` class supports options like:
- `emit_none`: If True, yields None to simulate time passing
- `time_scale`: Speed up or slow down playback (e.g., 2.0 for twice as fast)
- `ensure_sorted`: Sort events by timestamp before playback

### Event Manipulation and Composition

You can manipulate event sequences programmatically:

```python
# Create a modified version by changing timestamps
faster_events = [(t/2, params) for t, params in events]

# Create a reversed sequence
reversed_events = [(events[-1][0] - t, params) for t, params in events]

# Combine sequences by appending
melody = events + [(t + events[-1][0], params) for t, params in events]

# Transpose a sequence by modifying frequency values
def transpose(events, semitones):
    factor = 2 ** (semitones/12)
    result = []
    for t, params in events:
        new_params = params.copy()
        if 'freq' in new_params:
            if isinstance(new_params['freq'], dict):
                new_params['freq'] = {
                    k: v * factor if k == 'value' else v
                    for k, v in new_params['freq'].items()
                }
            else:
                new_params['freq'] = new_params['freq'] * factor
        result.append((t, new_params))
    return result

# Transpose up a perfect fifth (7 semitones)
fifth_up = transpose(events, 7)
```

### Complex Synthesis Graphs

You can create more sophisticated synthesis graphs by combining Pyo objects:

```python
from pyo import Sine, Delay, Chorus, Harmonizer, MoogLP

@Synth(dials='freq cutoff', settings='delay_time num_voices')
def complex_synth(freq=440, cutoff=2000, delay_time=0.25, num_voices=3):
    # Create a sine oscillator
    osc = Sine(freq=freq, mul=0.3)
    
    # Add harmonizer for multiple voices
    harm = Harmonizer(osc, transpo=[0, 7, 12][:num_voices], feedback=0.1)
    
    # Add a filter
    filt = MoogLP(harm, freq=cutoff, res=0.3)

    # etc...
    
```

## Hooking a Synth to Event Streams

You can connect your synths to external event sources like keyboards, sensors, or algorithmic generators. This allows you to create interactive instruments or audio installations.

### Keyboard Control Integration

The `keyboard_control.py` module provides a way to connect keyboard events to your synths, creating an interactive musical instrument controlled by your computer keyboard.

The module works by:

1. Detecting keyboard events using the `pynput` library
2. Mapping keys to synth parameters using a configurable dictionary
3. Calling your synth function with the mapped parameters when keys are pressed
4. Supporting multiple callback patterns and configuration options

### Command Line Usage

The simplest way to try keyboard control is through the command line:

```bash
python -m hum.examples.keyboard_control \
  --callback "hum.pyo_synths:sine" \
  --arg-mapping '{"q":130.81,"w":146.83,"e":164.81,"r":174.61,"t":196.00,"y":220.00,"u":246.94,"i":261.63,"o":293.66,"p":329.63,"a":261.63,"s":293.66,"d":329.63,"f":349.23,"g":392.00,"h":440.00,"j":493.88,"k":523.25,"l":587.33,"z":523.25,"x":587.33,"c":659.25,"v":698.46,"b":783.99,"n":880.00,"m":987.77}' \
  --debug
```

This creates a two-octave piano keyboard layout on your QWERTY keyboard, where:
- The top two rows (Q-P and A-L) form the white keys
- The bottom row (Z-M) forms the black keys
- Each key is mapped to a specific frequency in Hz

### Programmatic Usage

You can also use the keyboard control module in your own code:

```python
from hum.examples.keyboard_control import keyboard_reader
from hum.pyo_synths import sine  # A pre-defined sine wave synth

# Define a mapping from keys to frequencies
key_mapping = {
    "a": 261.63,  # C4
    "s": 293.66,  # D4
    "d": 329.63,  # E4
    "f": 349.23,  # F4
    "g": 392.00,  # G4
    "h": 440.00,  # A4
    "j": 493.88,  # B4
    "k": 523.25   # C5
}

# Create a keyboard reader with the mapping
reader = keyboard_reader(
    callback=sine,
    arg_mapping=key_mapping,
    exit_key="escape",
    read_rate=0.1,
    debug=True
)

# Process keyboard events
for event in reader:
    if event:
        print(f"Key pressed: {event['key']}")
    if event and event['key'] == 'escape':
        break
```

### Advanced Mapping

You can map keys to complex parameter dictionaries for more control:

```python
# Map keys to multiple parameters
advanced_mapping = {
    "a": {"freq": 261.63, "waveform": "sine"},
    "s": {"freq": 293.66, "waveform": "triangle"},
    "d": {"freq": 329.63, "waveform": "square", "attack": 0.2}
}

# Use with a more complex synth
reader = keyboard_reader(
    callback=simple_waveform_synth,
    arg_mapping=advanced_mapping
)
```

### How It Works Under the Hood

The `keyboard_control.py` module:

1. Registers a callback function with the `pynput` library to capture key presses
2. Processes the raw key data into a standardized format
3. Looks up the key in the mapping dictionary to find associated parameters
4. Calls the synth callback in the appropriate way based on its type:
   - If it's a `Synth` instance, uses the `update` method
   - If it's a regular function, calls it with the mapped parameters
5. Yields events in a generator pattern for processing in your code

This flexible design allows you to use keyboard control with any type of synth function or parameter mapping, making it easy to create interactive audio experiences.

## Use Cases

The `Synth` and keyboard control tools can be used for:

1. **Interactive Music Creation** - Build synthesizers and play them with your keyboard
2. **Sound Design** - Experiment with different parameters in real-time
3. **Education** - Demonstrate audio concepts with interactive examples
4. **Algorithmic Composition** - Create and manipulate event sequences programmatically
5. **Audio Prototyping** - Quickly test audio ideas before implementing them in other systems
6. **Game Audio** - Create dynamic sound effects that respond to game events
7. **Installation Art** - Build interactive audio installations
8. **Live Performance** - Control audio in real-time for live shows

## Tips and Best Practices

1. **Use Context Managers** - The `with` statement ensures proper cleanup
2. **Round Timestamps** - Use `round_event_times` for readable event sequences
3. **Resource Management** - Always stop synths when done to free audio resources
4. **Separate Concerns** - Use dials for real-time control and settings for structural changes
5. **Start Simple** - Build up complexity gradually as you get comfortable with the system
6. **Use Generators** - The functional approach to audio processing is powerful and flexible
7. **Test with Rendering** - Use `render_events` to test your sequences without real-time playback
8. **Documentation** - Document your synth functions with docstrings and examples

## Under the Hood

The `Synth` class works by:
1. Converting parameters to Pyo's `SigTo` objects for smooth transitions
2. Managing a Pyo server for audio processing
3. Recording all parameter changes with timestamps
4. Managing the underlying synthesis graph rebuilding when necessary
5. Handling event serialization and deserialization

The `keyboard_control` module:
1. Uses non-blocking event detection to capture keystrokes
2. Processes key events into a consistent format
3. Maps keys to synth parameters using configurable dictionaries
4. Handles different types of callback functions adaptively