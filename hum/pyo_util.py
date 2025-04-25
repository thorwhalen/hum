"""pyo utils"""

import os
import tempfile
import inspect
from inspect import Signature, Parameter
from typing import Callable, Optional, Union, Dict, Protocol, TypeVar, runtime_checkable
from collections.abc import MutableMapping
import time

from pyo import *

DFLT_PYO_SR = 44100
DFLT_PYO_NCHNLS = 1
DFLT_PYO_AUDIO = "portaudio"
DFLT_PYO_VERBOSITY = 1
DFLT_TIME_TIME = 0.025  # default time of pyo.SigTo, but should we default to 0 instead?


T = TypeVar('T')

KnobsDict = Dict[str, Union[float, SigTo, Dict[str, float]]]
Recorder = Callable[[float, KnobsDict], None]


@runtime_checkable
class Appendable(Protocol[T]):
    """
    A protocol for objects that can be appended to.
    """

    def append(self, item: T) -> None: ...


class PyoServer(Server):
    """
    A subclass of the pyo Server class that makes it a context manager.

    See pyo issue #300: https://github.com/belangeo/pyo/issues/300

    """

    _boot_with_new_buffer = True

    def __enter__(self):
        self.boot(newBuffer=self._boot_with_new_buffer)
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.shutdown()


class Knobs(MutableMapping):
    """
    A class for managing signal parameters (as SigTo) with optional recording.

    Behaves like a MutableMapping (dict-like) and supports automatic recording
    of parameter updates with timestamps.
    """

    def __init__(self, param_dict: KnobsDict, *, record: Recorder = None):
        """
        Parameters
        ----------
        param_dict : dict
            A dictionary of parameters to be used in the synthesizer.
            The keys are the names of the parameters, and the values are the initial values.
        record : Optional[callable]
            A function that takes a timestamp and a dictionary of parameters as arguments.
            This function will be called whenever the parameters are updated.
            Default is None, which means no recording will be done.
        """
        self._params = {
            k: v if isinstance(v, SigTo) else dict_to_sigto(v)
            for k, v in param_dict.items()
        }
        self._record = record

    def __call__(self, **updates_kwargs):
        self.update(updates_kwargs)

    def update(self, updates: KnobsDict = (), /, **kwargs):
        """
        Update the parameters of the synthesizer.

        Parameters
        ----------

        updates : dict
            A dictionary of parameters to update. The keys are the names of the parameters,
            and the values are the new values for those parameters.

        Mostly, the knob updates values are numbers, but you can also express how a
        particular knob should transition to that number ("value") from where it is now,
        by specifying a "time" parameter. The time parameter is the number of seconds
        it takes to transition to the new value.
        You can also specify a "mul" and "add" parameter, which are multiplied and
        added to the value respectively.
        Essentially, you can specify a `pyo.SigTo` object as the value of a knob.

        """
        if isinstance(updates, dict):
            combined = updates.copy()
        elif hasattr(updates, "__iter__"):
            combined = dict(updates)
        else:
            raise TypeError("Knobs.update() requires a mapping or iterable of pairs")

        combined.update(kwargs)
        self._fast_update(combined)

    def _fast_update(self, updates: KnobsDict):
        """
        The fast update method is used for real-time updates, where speed is critical.
        """
        for k, v in updates.items():
            sig = self._params[k]
            if isinstance(v, dict):
                # Note: No validation of the v keys. If you pass keys that are not
                # valid SigTo parameters, (i.e. other than value, time, mul or add)
                # it will simply add them to the SigTo object as attributes, but
                # that won't have any (audible) effect.
                # The reason we're not validating is that this update is in a real-time
                # context, and we don't want to slow it down with validation.
                for attr_name, attr_value in v.items():
                    setattr(sig, attr_name, attr_value)
            else:
                sig.value = v

        if self._record:
            self._record(time.time(), updates)

    @property
    def __signature__(self):
        return Signature(
            parameters=[
                Parameter(name, kind=Parameter.KEYWORD_ONLY) for name in self._params
            ]
        )

    def __setitem__(self, key, value):
        self.update({key: value})

    def __getitem__(self, key):
        return self._params[key]

    def __delitem__(self, key):
        del self._params[key]

    def __iter__(self):
        return iter(self._params)

    def __len__(self):
        return len(self._params)

    def __repr__(self):
        return f"<Knobs {list(self._params.keys())}>"


# TODO: Add validation of values
def get_pyoobj_params(pyoobj):
    """
    Get the parameters of a PyoObject subclass.
    """
    # get the signature of the function
    signa = inspect.signature(pyoobj)
    # get the dict of parameters, using names as keys and .default as values if provided, and None if not
    specs = {
        k: (v.default if v.default is not inspect.Parameter.empty else None)
        for k, v in signa.parameters.items()
    }
    return specs


def dict_to_sigto(d: Union[float, dict]) -> SigTo:
    """Create a SigTo object from a parameter spec dictionary or a number."""
    if isinstance(d, dict):
        # TODO: Simplify?
        sigto_kwargs = {
            k: v
            for k, v in d.items()
            if k in ['value', 'time', 'init', 'mul', 'add'] and v is not None
        }
    else:
        sigto_kwargs = {'value': d}
    return SigTo(**sigto_kwargs)


RecordFactory = Callable[[], Appendable]


def sigto_to_dict(sigto: SigTo) -> Dict[str, float]:
    """
    Convert a SigTo object to a dictionary.
    """
    return {
        'value': sigto.value,
        'time': sigto.time,
        'mul': sigto.mul,
        'add': sigto.add,
    }


def serialize_knobs(knobs):
    """
    Convert the knobs to a dictionary.
    """
    return {
        k: sigto_to_dict(v) if isinstance(v, SigTo) else v for k, v in knobs.items()
    }


def _json_friendly_records(records):
    """
    Convert the recorded frames to a JSON-friendly format.
    """

    def _processed_record(record):
        timestamp, knobs = record
        return timestamp, serialize_knobs(knobs)

    return list(map(_processed_record, records))


class Synth:
    """
    A class for creating a real-time synthesizer using pyo.

    The synth_func is a function whose arguments determine the parameters of the synthesizer.
    We call a dictionary of such parameters "knobs":
    These are dicts that specify the value for each knob (i.e. sound parameter) that
    has changed (all other sound parameters remain as they were).

    Mostly, the knobs values are numbers, but you can also express how a particular knob
    should transition to that number ("value") from where it is now, by specifying
    a "time" parameter. The time parameter is the number of seconds it takes to
    transition to the new value.
    You can also specify a "mul" and "add" parameter, which are multiplied and added to the value
    respectively.
    Essentially, you can specify a `pyo.SigTo` object as the value of a knob.
    """

    default_time_time = DFLT_TIME_TIME

    def __init__(
        self,
        synth_func,
        *,
        sr=DFLT_PYO_SR,
        nchnls=DFLT_PYO_NCHNLS,
        record_on_start: bool = True,
        event_log_factory: RecordFactory = list,  # No argument factory that makes an Appendable
        audio='portaudio',
        verbosity=DFLT_PYO_VERBOSITY,
        **server_kwargs,
    ):
        """

        Parameters
        ----------
        synth_func : callable
            A function that returns a pyo object. The function should accept keyword arguments
            that are the parameters of the synthesizer.
        sr : int
            The sample rate of the server. Default is 44100.
        nchnls : int
            The number of channels of the server. Default is 1.
        record_on_start : bool
            Whether to start recording when the server starts. Default is True.
        event_log_factory : callable
            A function that returns an empty list or other Appendable object to store the
            recorded events. Default is list.
        audio : str
            The audio driver to use. Default is 'portaudio'.
        verbosity : int
            The verbosity level of the server. Default is 1.
        server_kwargs : dict
            Additional keyword arguments to pass to the pyo Server constructor.
        """
        self._server_kwargs = dict(
            server_kwargs, sr=sr, nchnls=nchnls, verbosity=verbosity, audio=audio
        )
        self._synth_func = synth_func
        self._server = None
        self.output = None
        self.knobs = None
        self._synth_func_params = get_pyoobj_params(self._synth_func)

        # Recording
        self._record_on_start = record_on_start
        self._event_log_factory = event_log_factory
        self._recording = False
        self._recording_start_time = None
        self._recorded_events = None

    def _init_recorded_events(self):
        self._recorded_events = self._event_log_factory()
        # The first even shold be the initial state of the synth_func
        self._recorded_events.append(self._synth_func_params)

    def _record_callback(self, t, updates):
        rel_time = t - self._recording_start_time
        self._recorded_events.append((rel_time, updates))

    def start_recording(self):
        self._recording = True
        self._recorded_events = self._event_log_factory()
        self._recording_start_time = time.time()
        if self.knobs is None:
            raise RuntimeError(
                "Cannot start recording without initializing knobs. "
                "Did you forget to call start() or do a with synth... block?"
            )

        self.knobs._record = self._record_callback
        self._record_callback(self._recording_start_time, serialize_knobs(self.knobs))

    def stop_recording(self):
        self._recording = False
        if self.knobs:
            self.knobs._record = False

        # Inject a final dummy event at current relative time
        if self._recorded_events and self._recording_start_time is not None:
            now = time.time()
            rel_now = now - self._recording_start_time
            self._recorded_events.append((rel_now, {}))

    def get_recording(self, process_recording: Callable = _json_friendly_records):
        return process_recording(self._recorded_events)

    def start(self):
        if self._server is None:
            self._server = Server(**self._server_kwargs).boot()

        self._initial_knob_params = {
            name: dict_to_sigto(spec) for name, spec in self._synth_func_params.items()
        }

        self.knobs = Knobs(self._initial_knob_params)
        self.output = self._synth_func(**self._initial_knob_params)
        self.output.out()
        self._server.start()

        if self._record_on_start:
            self.start_recording()

    def stop(self):
        if self.output is not None:
            self.output.stop()
        if self._server is not None:
            self._server.stop()
        if self._record_on_start:
            self.stop_recording()

    def render_recording(
        self,
        control_events=None,
        *,
        output_filepath=None,
        egress=lambda x: x,
        file_format='wav',
    ):

        if not control_events:
            control_events = self.get_recording(
                process_recording=_json_friendly_records
            )
            if not control_events:
                raise ValueError("Nothing to render. No control events recorded.")

        base_time = control_events[0][0]
        normalized = [(t - base_time, knobs) for t, knobs in control_events]
        total_duration = normalized[-1][0] + 0.1  # add small buffer

        offline_server_kwargs = dict(self._server_kwargs, audio="offline")
        server = Server(**offline_server_kwargs).boot()
        table = NewTable(length=total_duration)

        all_keys = set(k for _, knobs in control_events for k in knobs)
        raw_params = {k: SigTo(value=0, time=0.01) for k in all_keys}
        synth_output = self._synth_func(**raw_params)
        table_recorder = TableRec(synth_output, table=table).play()

        try:
            for i, (event_time, updates) in enumerate(normalized):
                next_time = (
                    normalized[i + 1][0] if i + 1 < len(normalized) else total_duration
                )
                dur = next_time - event_time

                for key, val in updates.items():
                    if isinstance(val, dict):
                        for attr in ['value', 'time', 'mul', 'add']:
                            if attr in val:
                                setattr(raw_params[key], attr, val[attr])
                    else:
                        raw_params[key].value = val

                server.recordOptions(dur=dur)
                server.start()
        finally:
            table_recorder.stop()

        if output_filepath is None:
            tmpfile = tempfile.NamedTemporaryFile(
                suffix=f".{file_format}", delete=False
            )
            output_filepath = tmpfile.name
            tmpfile.close()

        table.save(output_filepath)

        if egress is None:
            return output_filepath
        elif callable(egress):
            with open(output_filepath, "rb") as f:
                audio = f.read()
            return egress(audio)
        else:
            raise ValueError("Invalid egress")

    def __del__(self):
        if self._server is not None:
            self._server.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


# --------------------------------------------------------------------------------------
# Example functions for testing the Synth class
# --------------------------------------------------------------------------------------


def example_01_basic_dual_osc():
    from hum.pyo_util import Synth
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
        synth.knobs['freq1'] = 440
        time.sleep(1)
        synth.knobs['freq2'] = SigTo(550, time=0.5)  # Smooth shift
        time.sleep(0.5)
        synth.knobs.update(dict(freq1={'value': 880, 'time': 0.1}, freq2=1100))
        time.sleep(1)

    wav_bytes = synth.render_recording()
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
        synth.knobs['cutoff'] = 500
        time.sleep(1)
        synth.knobs['cutoff'] = SigTo(3000, time=2.0)
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
        synth.knobs['cutoff'] = 200
        time.sleep(1)
        synth.knobs['cutoff'] = SigTo(4000, time=1.5)
        synth.knobs['res'] = 0.9
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
        synth.knobs['trem_rate'] = 7.0
        time.sleep(1)
        synth.knobs['cutoff'] = 400
        time.sleep(1)
        synth.knobs['cutoff'] = 2000
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
        {'tempo': 120, 'frequency': 440, 'amplitude': 0.5, 'cutoff': 1000},
        {'tempo': 135, 'frequency': 330, 'amplitude': 0.7, 'cutoff': 500},
        {'tempo': 100, 'frequency': 550, 'amplitude': 0.3, 'cutoff': 2000},
        {'tempo': 150, 'frequency': 220, 'amplitude': 0.9, 'cutoff': 750},
        {'tempo': 110, 'frequency': 660, 'amplitude': 0.6, 'cutoff': 1500},
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
        synth.knobs['freq1'] = 440
        time.sleep(1)
        synth.knobs['freq2'] = SigTo(550, time=0.5)  # Smooth shift
        time.sleep(1)
        synth.knobs.update(dict(freq1={'value': 880, 'time': 0.1}, freq2=1100))
        time.sleep(1)

    recorded_frames = synth.get_recording()
    assert len(recorded_frames) > 0, "No frames recorded"
    return recorded_frames
