import time
import pytest
from pyo import Sine
import recode
import os

from hum.pyo_util import Synth, DFLT_PYO_SR, round_event_times
from hum.extra_util import estimate_frequencies

running_in_ci = os.environ.get("CI") in ("true", "1")

if running_in_ci:
    synth_special_kwargs = dict(audio='dummy')
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
    def simple_sine(freq=base_freq, volume=0.1):
        return Sine(freq=freq, mul=volume)

    synth = Synth(simple_sine, **synth_special_kwargs)

    # Define a sequence of frequencies to play
    freq_sequence = [base_freq] + [base_freq * 3 / 2, base_freq * 2]

    # Play the frequencies in sequence
    with synth:
        time.sleep(1)  # let base_freq play for a second
        synth['freq'] = freq_sequence[1]  # Change to 330 Hz
        time.sleep(1)
        synth['freq'] = freq_sequence[2]  # Change to 440 Hz
        time.sleep(1)

    # Get the recorded events and round timing for consistency
    events = synth.get_recording()
    events = list(round_event_times(events, round_to=0.1))
    expected_events = [
        (
            0.0,
            {
                'freq': {'value': 220, 'time': 0.025, 'mul': 1, 'add': 0},
                'volume': {'value': 0.1, 'time': 0.025, 'mul': 1, 'add': 0},
            },
        ),
        (1.0, {'freq': 330.0}),
        (2.0, {'freq': 440}),
        (3.0, {}),
    ]
    assert events == expected_events, f"Expected {expected_events}, got {events}"

    # Render the recording to audio
    wav_bytes = synth.render_recording(events, suffix_buffer_seconds=0)

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
