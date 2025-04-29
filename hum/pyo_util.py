"""pyo utils"""

import os
import tempfile
import inspect
from inspect import Signature, Parameter
from typing import (
    Iterator,
    Callable,
    Optional,
    Union,
    Dict,
    Protocol,
    Set,
    TypeVar,
    runtime_checkable,
)
from collections.abc import MutableMapping
import time

from pyo import PyoObject
from pyo import *  # TODO: change to be explicit object imports

DFLT_PYO_SR = 44100
DFLT_PYO_NCHNLS = 1
DFLT_PYO_AUDIO = "portaudio"
DFLT_PYO_VERBOSITY = 1
DFLT_TIME_TIME = 0.025  # default time of pyo.SigTo, but should we default to 0 instead?


T = TypeVar('T')

KnobsValue = Union[float, SigTo, Dict[str, float]]
KnobsDict = Dict[str, KnobsValue]
Recorder = Callable[[float, KnobsDict], None]
SigToValueType = Union[float, int, PyoObject]
_sigto_value_types = set(SigToValueType.__args__)


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


def knob_params(*include: str):
    """
    Decorator to specify which parameters should be treated as live knobs (SigTo).
    """

    def decorator(func):
        func._knob_params = set(include)
        return func

    return decorator


def knob_exclude(*exclude: str):
    """
    Decorator to specify which parameters should NOT be treated as live knobs.
    """

    def decorator(func):
        func._knob_exclude = set(exclude)
        return func

    return decorator


class Knobs(MutableMapping):
    """
    A class for managing signal parameters (as SigTo) with optional recording.

    Behaves like a MutableMapping (dict-like) and supports automatic recording
    of parameter updates with timestamps.
    """

    def __init__(self, param_dict: KnobsDict, *, record: Recorder = None, live=False):
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
        self.is_live = live
        self._record = record
        self._params = {}
        self._sig_to_params = {
            k: v for k, v in param_dict.items() if isinstance(v, SigTo)
        }

        if live:
            self._params = param_dict
            # self._params = {
            #     k: v if isinstance(v, SigTo) else dict_to_sigto(v)
            #     for k, v in param_dict.items()
            # }
            self._fast_update = self._live_update
        else:
            self._params = param_dict  # .copy()
            self._update_log = []
            self._fast_update = self._offline_update

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

    def _live_update(self, updates: KnobsDict):
        for k, v in updates.items():
            sig = self._params[k]
            if isinstance(sig, SigTo):
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
            else:
                # Not a SigTo: Replace value directly
                self._params[k] = v

        if self._record:
            self._record(time.time(), updates)

    # TODO: I don't like the fact that there's a _update_log PLUS a self._record
    #   Se if we can merge them -- and add control over time stamps
    def _offline_update(self, updates: KnobsDict):
        """
        The offline update method is used for non-real-time updates, where we can
        afford to be a bit slower.
        """
        self._update_log.append(updates)

        if self._record:
            self._record(time.time(), updates)

    def get_update_log(self):
        """
        Get the update log.
        """
        return self._update_log

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


# TODO: Deprecate if params_dict deprecates
def is_valid_sigto_value(value):
    """
    Check if the value is a valid SigTo value.
    """

    return isinstance(value, SigToValueType.__args__)


# TODO: Deprecate? Was to replace get_pyoobj_params, but now we use synth_func_defaults
def params_dict(pyoobj, is_valid_value=is_valid_sigto_value):
    """
    Get a dict of {key: value} pairs from a callable object (e.g. a PyoObject).
    Only (k,v) items whose v satisfies is_valid_value(v) are included.
    """
    # get the signature of the function
    signa = inspect.signature(pyoobj)

    def sigto_items():
        for k, v in signa.parameters.items():
            value = v.default
            if is_valid_sigto_value(value):
                yield k, value

    return dict(sigto_items())


def synth_func_defaults(func: Callable) -> dict:
    """
    Get the {name: default_value} pairs from a callable object (e.g. a PyoObject).
    Also asserts that all arguments of function have a default value.
    """
    # get the signature of the function
    func_signature = inspect.signature(func)

    def name_and_default_pairs():
        for k, v in func_signature.parameters.items():
            if (
                v.kind == inspect.Parameter.VAR_KEYWORD
                or v.kind == inspect.Parameter.VAR_POSITIONAL
            ):
                # Skip *args and **kwargs
                continue
            elif v.default is inspect.Parameter.empty:
                raise ValueError(
                    f"Synth function {func.__name__} has no default value for {k}"
                )
            yield k, v.default

    return dict(name_and_default_pairs())


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


# TODO: Not using this. Thought I needed it. Delete if not needed.
def auto_play(obj):
    """
    Recursively call .play() on all PyoObjects inside an object or list.
    """
    if isinstance(obj, PyoObject):
        obj.play()
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for item in obj:
            auto_play(item)
    # Optional: if your graph sometimes returns dicts of PyoObjects
    elif isinstance(obj, dict):
        for item in obj.values():
            auto_play(item)
    # Otherwise: ignore non-Pyo things (ints, floats, etc.)


import time
from typing import Iterator, Union, List, Tuple, Dict

KnobsDict = Dict[str, Union[float, dict]]  # reusing your existing type


class ReplayEvents:
    """
    An iterator that replays a sequence of timestamped knob updates in real-time.

    This class takes a list of (timestamp, KnobsDict) events and yields
    each update at the correct real-time intervals, automatically sleeping
    between updates to recreate the original timing.

    Usage:
    ------
    Simply wrap your recorded control events and feed them into play_events:

        synth.play_events(ReplayEvents(control_events))

    Why use ReplayEvents?
    ----------------------
    - Centralizes real-time replay logic.
    - Makes it easy to later add speed controls, jitter, filtering, etc.
    - Keeps `play_events` clean and flexible.
    - Supports a future where playback streams can be manipulated easily.

    Parameters
    ----------
    control_events : List[Tuple[float, KnobsDict]]
        A list of (timestamp, knob updates) pairs.
    speed : float
        A multiplier for playback speed. 1.0 = normal speed,
        2.0 = twice as fast, 0.5 = half speed, etc. Default is 1.0.
    """

    def __init__(
        self, control_events: List[Tuple[float, KnobsDict]], speed: float = 1.0
    ):
        self.control_events = control_events
        self.speed = speed

    def __iter__(self) -> Iterator[Union[KnobsDict, None]]:
        if not self.control_events:
            return
        base_time, first_update = self.control_events[0]
        yield first_update
        for (curr_time, updates), (next_time, _) in zip(
            self.control_events, self.control_events[1:]
        ):
            sleep_duration = (next_time - curr_time) / self.speed
            time.sleep(max(sleep_duration, 0))
            yield updates


def list_if_string(x):
    """
    Convert a string to a list of strings.
    """
    if isinstance(x, str):
        return [x]
    return x


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
        knob_params: Optional[Set[str]] = None,
        knob_exclude: Optional[Set[str]] = None,
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
        self._synth_func_params = synth_func_defaults(synth_func)

        _knob_params = knob_params or getattr(
            synth_func, '_knob_params', set(self._synth_func_params)
        )
        _knob_params = list_if_string(_knob_params)
        _knob_exclude = knob_exclude or getattr(synth_func, '_knob_exclude', set())
        knob_exclude = list_if_string(knob_exclude)
        self._knob_params = set(_knob_params) - set(_knob_exclude)
        self._knob_defaults = {k: self._synth_func_params[k] for k in _knob_params}

        # Recording
        self._record_on_start = record_on_start
        self._event_log_factory = event_log_factory
        self._recording = False
        self._recording_start_time = None
        self._recorded_events = None

        # In Synth.__init__:
        self.knobs = Knobs(self._synth_func_params, live=False)

    @property
    def _initial_knob_params(self):
        _initial_knob_params = {}
        for name, spec in self._synth_func_params.items():
            if name in self._knob_params:
                _initial_knob_params[name] = dict_to_sigto(spec)
            else:
                _initial_knob_params[name] = spec

        return _initial_knob_params

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

        _initial_knob_params = self._initial_knob_params.copy()

        self.knobs = Knobs(
            _initial_knob_params, record=self._record_callback, live=True
        )

        try:
            self.output = self._synth_func(**_initial_knob_params)
        except TypeError as e:
            raise TypeError(
                f"Failed to initialize synth function '{self._synth_func.__name__}'.\n"
                f"Perhaps some arguments were wrongly wrapped into SigTo.\n"
                f"Consider :\n"
                f"  * using the sigto_include argument to control which parameters are live.\n"
                f"  * using the sigto_exclude argument to control which parameters are not live.\n"
                f"Alternatively, at function definition time, you can control this by:\n"
                f"  * using the @knob_params decorator to control which parameters are live.\n"
                f"  * using the @knob_exclude decorator to control which parameters are not live.\n"
                f"Original error: {e}"
            ) from e

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

    def play_events(
        self,
        events: Iterator[Union[KnobsDict, None]],
        *,
        idle_sleep_time=0.01,
        inter_event_delay=0,
        tail_time=1.0,
    ):
        """
        Play an event stream, applying updates from an iterator of KnobsDicts or None.

        Parameters
        ----------
        events : Iterator[Union[KnobsDict, None]]
            An iterator yielding KnobsDict updates (or None for idle).
        tail_time : float
            Time to wait after finishing event playback to allow sound to finish.
        idle_sleep_time : float
            Time to sleep if the iterator yields None (idle).
        inter_event_delay : float
            Time to sleep systematically after each non-None event.
        """
        with self:
            for event in events:
                if event is None:
                    time.sleep(idle_sleep_time)
                    continue

                self.knobs.update(event)

                if inter_event_delay > 0:
                    time.sleep(inter_event_delay)

            time.sleep(tail_time)

    def replay_events(self, control_events, *, tail_time=1.0):
        self.play_events(ReplayEvents(control_events), tail_time=tail_time)

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

        TIME_IDX = 0
        base_time = control_events[0][TIME_IDX]
        normalized = [(t - base_time, knobs) for t, knobs in control_events]
        total_duration = normalized[-1][TIME_IDX] + 0.1  # add small buffer

        offline_server_kwargs = dict(self._server_kwargs, audio="offline")
        server = Server(**offline_server_kwargs).boot()
        table = NewTable(length=total_duration)

        all_keys = set(k for _, knobs in control_events for k in knobs)
        raw_params = {k: SigTo(value=0, time=0.01) for k in all_keys}
        synth_output = self._synth_func(**raw_params).out()
        table_recorder = TableRec(synth_output, table=table).play()

        try:
            for i, (event_time, updates) in enumerate(normalized):
                next_time = (
                    normalized[i + 1][TIME_IDX]
                    if i + 1 < len(normalized)
                    else total_duration
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

    # ------------------------- replay the events -------------------------
    time.sleep(0.5)

    events = synth.get_recording()
    synth.replay_events(events)

    # ------------------------- get the wav_bytes of this recording --------------------
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
        synth.knobs['freq1'] = 440
        time.sleep(1)
        synth.knobs['freq2'] = SigTo(550, time=0.5)  # Smooth shift
        time.sleep(1)
        synth.knobs.update(dict(freq1={'value': 880, 'time': 0.1}, freq2=1100))
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
    wf, sr = synth2.render_recording(control_events, egress=recode.decode_wav_bytes)
    return wf, sr
