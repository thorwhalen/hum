import time
import pytest
from pyo import Sine
import recode
import os

from hum.pyo_util import Synth, DFLT_PYO_SR, round_event_times
from hum.extra_util import estimate_frequencies

running_in_ci = os.environ.get("CI") in ("true", "1")

if running_in_ci:
    synth_special_kwargs = dict(audio="dummy")
else:
    synth_special_kwargs = {}


@pytest.mark.skipif(running_in_ci, reason="Skipped on CI: Need speakers!")
def test_synth_frequency_sequence():
    """
    Test that the Synth correctly plays different frequencies in sequence and
    that these frequencies can be recorded, rendered, and verified.
    """

    base_freq = 220

    # Make a synth that plays a simple sine waves
    def simple_sine(freq=base_freq):
        return Sine(freq=freq)

    s = Synth(simple_sine, **synth_special_kwargs)

    # Define a sequence of frequencies to play
    freq_sequence = [base_freq] + [base_freq * 3 / 2, base_freq * 2]

    # Play the frequencies in sequence
    with s:
        time.sleep(1)  # let base_freq play for a second
        s(freq=freq_sequence[1])  # Change to 330 Hz
        time.sleep(1)  # ... play that for a second
        s['freq'] = freq_sequence[
            2
        ]  # ...then change to 440 Hz (doing it differently)
        time.sleep(1)  # ... play that for a second
        # ... tnad htne exit the context manager, which stops the synth

    # These actions on the synth should be recorded, with the time of the
    # change recorded as well.
    # You can get the recorded events like so:
    events = s.get_recording()
    # Often, you might want to round timing for consistency and readability:
    events = list(round_event_times(events, round_to=0.1))
    # These are the events we expect to see:
    expected_events = [
        (
            0.0,
            {
                # Note that the first event is the event create by the defaults of the synth function (and other internals)
                'freq': {'value': 220, 'time': 0.025, 'mul': 1, 'add': 0},
            },
        ),
        (1.0, {'freq': 330.0}),
        (2.0, {'freq': 440}),
        # Note that the last event's "knobs" dict is empty, as we are just recording when the synth is stopped
        (3.0, {}),
    ]
    assert events == expected_events, f"Expected {expected_events}, got {events}"

    # You can "render" these. By default, the synth will render to a WAV format bytes.
    # This means that you can then save these, play these, or do whatever you want with them.
    wav_bytes = s.render_events(events)

    # Verify basic properties of the rendered audio
    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 0, "No bytes returned"

    # Decode the WAV bytes
    wf, sr = recode.decode_wav_bytes(wav_bytes)

    # Verify sample rate and duration
    assert sr == DFLT_PYO_SR, "Sample rate mismatch"
    total_duration = len(wf) / sr
    assert (
        abs(total_duration - 3.0) < 0.1
    ), f"Expected ~3 seconds of audio, got {total_duration}"

    # Analyze the frequencies in the audio
    estimated_frequencies = list(map(int, estimate_frequencies(wf, sr, chunker=sr)))
    expected_frequencies = list(map(int, freq_sequence))

    # Verify that the frequencies match what we expected
    assert (
        estimated_frequencies == expected_frequencies
    ), f"Frequency mismatch! Expected {expected_frequencies}, got {estimated_frequencies}"
