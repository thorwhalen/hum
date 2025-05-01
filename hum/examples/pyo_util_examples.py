"""
Example functions for testing the Synth class
"""


def example_01_basic_dual_osc():
    from hum.pyo_util import Synth, DFLT_PYO_SR
    import time
    from pyo import Sine, Mix, ButLP, SigTo

    # Dual sine oscillator with LFO-controlled lowpass filter
    def dual_osc_graph(freq1=220, freq2=330, amp=0.3, lfo_freq=0.5):
        osc1 = Sine(freq=freq1)
        osc2 = Sine(freq=freq2)
        blend = Mix([osc1, osc2], voices=2) * amp
        lfo = Sine(freq=lfo_freq).range(400, 2000)
        return ButLP(blend, freq=lfo)

    synth = Synth(dual_osc_graph, nchnls=2)
    # 👂 Frequency shifts with LFO sweeping filter
    with synth:
        time.sleep(0.5)
        synth["freq1"] = 440
        time.sleep(1)
        synth["freq2"] = SigTo(550, time=0.5)  # Smooth shift
        time.sleep(0.5)
        synth.knobs.update(dict(freq1={"value": 880, "time": 0.1}, freq2=1100))
        time.sleep(1)

    # ------------------------- replay the events -------------------------
    time.sleep(0.5)

    events = synth.get_recording()
    synth.replay_events(events)

    # ------------------------- get the wav_bytes of this recording --------------------
    wav_bytes = synth.render_events()
    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 0, "No bytes returned"
    # verify that the bytes are a valid WAV file
    import recode

    wf, sr = recode.decode_wav_bytes(wav_bytes)
    assert sr == DFLT_PYO_SR, "Sample rate mismatch"
    len(wf) > sr * 3, "Not enough samples in the WAV file"

    return wf, sr


def example_02_distortion_and_reverb():
    from hum.pyo_util import Synth
    import time
    from pyo import Sine, Atan2, Biquadx, WGVerb, SigTo

    # Distorted sine tone through a filter and reverb
    def distorted_reverb_graph(freq=440, amp=0.25, cutoff=1000):
        base = Sine(freq=freq) * amp
        distorted = Atan2(base * 4)
        filtered = Biquadx(distorted, freq=cutoff, q=3, type=0)
        return WGVerb(filtered, feedback=0.8, cutoff=5000, bal=0.3)

    # 👂 Brightness control of distorted tone
    with Synth(distorted_reverb_graph) as synth:
        time.sleep(1)
        synth["cutoff"] = 500
        time.sleep(1)
        synth["cutoff"] = SigTo(3000, time=2.0)
        time.sleep(2)


def example_03_detuned_polyphonic_feel():
    from hum.pyo_util import Synth
    import time
    from pyo import Sine, Mix, MoogLP, SigTo

    # Two slightly detuned sine waves with resonant Moog filter
    def poly_synth_graph(freq=330, amp=0.2, cutoff=800, res=0.5):
        osc1 = Sine(freq=freq * 0.995)
        osc2 = Sine(freq=freq * 1.005)
        summed = Mix([osc1, osc2], voices=2) * amp
        return MoogLP(summed, freq=cutoff, res=res)

    # 👂 Sweeping resonance and cutoff
    with Synth(poly_synth_graph) as synth:
        time.sleep(1)
        synth["cutoff"] = 200
        time.sleep(1)
        synth["cutoff"] = SigTo(4000, time=1.5)
        synth["res"] = 0.9
        time.sleep(2)


def example_04_dynamic_tremolo_and_filter_sweep():
    from hum.pyo_util import Synth
    import time
    from pyo import Sine, ButLP

    # Tremolo via amplitude modulation with filter
    def tremolo_filter_graph(freq=440, trem_rate=3.0, amp=0.5, cutoff=1200):
        tone = Sine(freq=freq)
        trem = Sine(freq=trem_rate).range(0.2, 1.0)
        modulated = tone * trem * amp
        return ButLP(modulated, freq=cutoff)

    # 👂 Tremolo rate and filter brightness changes
    with Synth(tremolo_filter_graph) as synth:
        time.sleep(1)
        synth["trem_rate"] = 7.0
        time.sleep(1)
        synth["cutoff"] = 400
        time.sleep(1)
        synth["cutoff"] = 2000
        time.sleep(2)


def example_05_offline_rendering():
    import statistics
    from hum.pyo_util import OfflineSynthRenderer
    from pyo import Sine, ButLP

    import recode

    def my_graph(frequency, tempo, amplitude, cutoff):
        osc = Sine(freq=frequency)
        lfo = Sine(freq=tempo / 60.0).range(0.2, 1.0)
        modulated = osc * amplitude * lfo
        return ButLP(modulated, freq=cutoff)

    frames = [
        {"tempo": 120, "frequency": 440, "amplitude": 0.5, "cutoff": 1000},
        {"tempo": 135, "frequency": 330, "amplitude": 0.7, "cutoff": 500},
        {"tempo": 100, "frequency": 550, "amplitude": 0.3, "cutoff": 2000},
        {"tempo": 150, "frequency": 220, "amplitude": 0.9, "cutoff": 750},
        {"tempo": 110, "frequency": 660, "amplitude": 0.6, "cutoff": 1500},
    ]

    renderer = OfflineSynthRenderer(
        synth_func=my_graph,
        parameter_frames=frames,
        frame_durations=2.0,
        sr=44100,
        egress=recode.decode_wav_bytes,
    )

    wf, sr = renderer.render()
    assert sr == 44100
    assert isinstance(wf, list)
    assert len(wf) == int(len(frames) * 2.0 * 44100)
    assert isinstance(wf[0], int)
    assert statistics.stdev(wf) > 0, "all samples are the same, something is wrong"

    return wf, sr


def example_06_knob_recording_playback():
    from hum.pyo_util import Synth
    import time
    import recode
    from pyo import Sine, ButLP, SigTo

    # Dual sine oscillator with LFO-controlled lowpass filter
    def dual_osc_graph(freq1=220, freq2=330, amp=0.3, lfo_freq=0.5):
        osc1 = Sine(freq=freq1)
        osc2 = Sine(freq=freq2)
        blend = Mix([osc1, osc2], voices=2) * amp
        lfo = Sine(freq=lfo_freq).range(400, 2000)
        return ButLP(blend, freq=lfo)

    synth = Synth(dual_osc_graph)

    with synth:
        synth["freq1"] = 440
        time.sleep(1)
        synth["freq2"] = SigTo(550, time=0.5)  # Smooth shift
        time.sleep(1)
        synth.knobs.update(dict(freq1={"value": 880, "time": 0.1}, freq2=1100))
        time.sleep(1)

    control_events = synth.get_recording()
    assert len(control_events) > 0, "No frames recorded"

    def dual_osc_graph_offline(freq1=220, freq2=330, amp=0.3, lfo_freq=0.5):
        osc1 = Sine(freq=freq1)
        osc2 = Sine(freq=freq2)
        blend = Mix([osc1, osc2], voices=2) * amp
        lfo_freq_sig = (
            lfo_freq if isinstance(lfo_freq, (PyoObject, SigTo)) else Sig(lfo_freq)
        )
        lfo = Sine(freq=lfo_freq_sig).range(400, 2000)
        return ButLP(blend, freq=lfo)

    synth2 = Synth(dual_osc_graph_offline)
    wf, sr = synth2.render_events(control_events, egress=recode.decode_wav_bytes)
    return wf, sr
