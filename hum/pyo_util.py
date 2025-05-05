"""pyo utils"""

import os
from functools import partial
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
    List,
    Set,
    Tuple,
    TypeVar,
    Mapping,
    runtime_checkable,
)
from collections.abc import MutableMapping
import time

from hum.util import round_numbers

from pyo import PyoObject
from pyo import *  # TODO: change to be explicit object imports


# -------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------

DFLT_PYO_SR = 44100
DFLT_PYO_NCHNLS = 1
DFLT_PYO_AUDIO = "portaudio"
DFLT_PYO_VERBOSITY = 1
DFLT_TIME_TIME = 0.025  # default time of pyo.SigTo, but should we default to 0 instead?


T = TypeVar("T")

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


# -------------------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------------------


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


_PLAY_REQUIRED = set()


def _build_play_required():
    import pyo

    global _PLAY_REQUIRED

    for name, cls in inspect.getmembers(pyo, inspect.isclass):
        if issubclass(cls, PyoObject) and cls.__module__.startswith("pyo"):
            if "play" in cls.__dict__:
                _PLAY_REQUIRED.add(cls.__name__)


# You can call this manually if dynamic loading is desired
def refresh_play_required_list():
    _build_play_required()


# Fallback manual list
_MANUAL_FALLBACK = {
    "Sine",
    "Osc",
    "Phasor",
    "LFO",
    "Adsr",
    "Fader",
    "Metro",
    "Pattern",
    "TrigEnv",
    "Randi",
    "Noise",
    "Choice",
    "Seq",
    "TrigFunc",
    "Counter",
}


def must_be_played(obj_or_cls) -> bool:
    """
    Returns True if the given pyo object or class needs `.play()` to produce sound.

    >>> from pyo import Sine, Freeverb, Adsr
    >>> must_be_played(Sine)
    True
    >>> must_be_played(Freeverb)
    False
    >>> must_be_played(Adsr)
    True
    """
    cls = obj_or_cls if isinstance(obj_or_cls, type) else obj_or_cls.__class__
    name = cls.__name__
    if not _PLAY_REQUIRED:
        _build_play_required()
    return name in _PLAY_REQUIRED or name in _MANUAL_FALLBACK


import ast
import inspect


class PyoPlayAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.objs_created = {}
        self.play_calls = set()

    def visit_Assign(self, node):
        # Detect PyoObject instance creation: foo = Sine(...) or foo = Adsr(...)
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            class_name = node.value.func.id
            if class_name in _MANUAL_FALLBACK:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.objs_created[target.id] = class_name
        self.generic_visit(node)

    def visit_Expr(self, node):
        # Detect .play() calls like: foo.play()
        if isinstance(node.value, ast.Call) and isinstance(
            node.value.func, ast.Attribute
        ):
            if node.value.func.attr == "play":
                if isinstance(node.value.func.value, ast.Name):
                    self.play_calls.add(node.value.func.value.id)
        self.generic_visit(node)


def analyze_synth_func_play(func):
    src = inspect.getsource(func)
    tree = ast.parse(src)
    analyzer = PyoPlayAnalyzer()
    analyzer.visit(tree)

    # Build the report
    report = {}
    for var, cls in analyzer.objs_created.items():
        report[var] = {
            "class": cls,
            "must_be_played": True,
            "play_called": var in analyzer.play_calls,
        }

    return report


def round_event_times(events, round_to=0.001):
    """
    Round the times (d[0]) of events ([d, ...]) to the nearest multiple of round_to.

    >>> events = [(0, {'freq': 220}), (1.0021910667419434, {'freq': 330.0})]
    >>> list(round_event_times(events, round_to=0.001))
    [(0.0, {'freq': 220}), (1.002, {'freq': 330.0})]
    >>> list(round_event_times(events, round_to=0.01))
    [(0.0, {'freq': 220}), (1.0, {'freq': 330.0})]
    """
    return round_numbers(
        events, round_to=round_to, index_of_item_number=0, egress=tuple
    )


from typing import Union, Iterable, List
import re


def ensure_identifier_list(x: Union[str, Iterable[str]]) -> List[str]:
    """
    Convert input to a list of valid Python identifiers.

    If input is a string, extract all word sequences.
    If input is an iterable, convert to a list.

    Args:
        x: String or iterable of strings

    Returns:
        List of valid Python identifiers

    Raises:
        ValueError: If any items in the list are not valid Python identifiers

    Examples:

        >>> ensure_identifier_list("foo bar baz")
        ['foo', 'bar', 'baz']
        >>> ensure_identifier_list(["foo", "bar"])
        ['foo', 'bar']
        >>> ensure_identifier_list("foo, bar.baz")
        ['foo', 'bar', 'baz']
        >>> ensure_identifier_list(None)
        []

    """
    # Handle string input by extracting word sequences
    if isinstance(x, str):
        # Extract word sequences (consecutive alphanumeric chars + underscore)
        identifiers = re.findall(r"\w+", x)
    # Handle iterable input by converting to list
    elif isinstance(x, Iterable):
        identifiers = list(x)
    elif x is None:
        return []
    else:
        raise TypeError(f"Expected string or iterable, got {type(x).__name__}")

    # Check if all items are valid Python identifiers
    invalid = [item for item in identifiers if not str(item).isidentifier()]
    if invalid:
        raise ValueError(
            f"The following items are not valid Python identifiers: {', '.join(repr(item) for item in invalid)}"
        )

    return identifiers


# -------------------------------------------------------------------------------------
# Knobs
# -------------------------------------------------------------------------------------


class Knobs(MutableMapping):
    """
    A simplified class for managing signal parameters (as SigTo).

    Behaves like a MutableMapping (dict-like) for parameter access.
    Recording functionality has been moved to the Synth class.
    """

    def __init__(self, param_dict: KnobsDict):
        """
        Parameters
        ----------
        param_dict : dict
            A dictionary of parameters to be used in the synthesizer.
            The keys are the names of the parameters, and the values are the initial values.
        """
        self._params = param_dict
        self._sig_to_params = {
            k: v for k, v in param_dict.items() if isinstance(v, SigTo)
        }

    def __call__(self, **updates_kwargs):
        """Update parameters using keyword arguments."""
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

        for k, v in combined.items():
            sig = self._params[k]
            if isinstance(sig, SigTo):
                if isinstance(v, dict):
                    # Note: No validation of the v keys. If you pass keys that are not
                    # valid SigTo parameters, (i.e. other than value, time, mul or add)
                    # it will simply add them to the SigTo object as attributes, but
                    # that won't have any (audible) effect.
                    for attr_name, attr_value in v.items():
                        setattr(sig, attr_name, attr_value)
                else:
                    sig.value = v
            else:
                # Not a SigTo: Replace value directly
                self._params[k] = v

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


# -------------------------------------------------------------------------------------
# Synth helpers
# -------------------------------------------------------------------------------------


def add_default_dials(dials: Union[str, List[str]]):
    def decorator(func):
        func._default_dials = set(ensure_identifier_list(dials))
        return func

    return decorator


def add_default_settings(settings: Union[str, List[str]]):
    def decorator(func):
        func._default_settings = set(ensure_identifier_list(settings))
        return func

    return decorator


def add_synth_defaults(dials=None, settings=None):
    """
    Decorator to add default dials and settings to a synth function.
    """

    def decorator(func):
        if dials is not None:
            func._default_dials = set(ensure_identifier_list(dials))
        if settings is not None:
            func._default_settings = set(ensure_identifier_list(settings))
        return func

    return decorator


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
            if k in ["value", "time", "init", "mul", "add"] and v is not None
        }
    else:
        sigto_kwargs = {"value": d}
    return SigTo(**sigto_kwargs)


RecordFactory = Callable[[], Appendable]


def sigto_to_dict(sigto: SigTo) -> Dict[str, float]:
    """
    Convert a SigTo object to a dictionary.
    """
    return {
        "value": sigto.value,
        "time": sigto.time,
        "mul": sigto.mul,
        "add": sigto.add,
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


class ReplayEvents:
    """
    Replays timestamped knob updates, optionally emitting None or sleeping to simulate
    timing.

    Parameters
    ----------
    events : List[Tuple[float, KnobsDict]]
        A list of (timestamp, knob updates) pairs.
    emit_none : bool
        If True, yields None to simulate time passage; if False, uses time.sleep.
    time_scale : float
        Time compression/stretch factor: 2.0 = 2x faster, 0.5 = half speed.
    ensure_sorted : bool
        Whether to sort the events by timestamp.

    Example
    -------

    Consider these events:

    >>> events = [
    ...     (0.0, {'freq': 220}),
    ...     (0.05, {'freq': 330}),
    ...     (0.10, {'freq': 440}),
    ... ]

    >>> list(ReplayEvents(events, time_scale=50.0))
    [{'freq': 220}, {'freq': 330}, {'freq': 440}]

    That is, we should really get back the same events, without the timestamps.

    >>> list(ReplayEvents(events, time_scale=10.0))
    [{'freq': 220}, {'freq': 330}, {'freq': 440}]

    >>> list(ReplayEvents(events, time_scale=10.0)) == list(dict(events).values())
    True

    The timestamps are used to pace when new events are emitted.
    Note: Usually we'll use time_scale=1.0 (the default), but we use time_scale=50 here
    just to accelerate the tests.

    The default is to sleep until the next event should be emitted, but you can also
    emulate a real-time sensor reads by setting emit_none=True.
    In this case, the instance will emit `None` if there's no event to emit at that time.

    >>> list(ReplayEvents(events, emit_none=True, time_scale=2.0))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    [{'freq': 220}, None, ...]

    """

    def __init__(self, events, *, emit_none=True, time_scale=1.0, ensure_sorted=False):
        if ensure_sorted:
            if not events:
                raise ValueError("Events list must not be empty.")
            self.events = sorted(events, key=lambda x: x[0])
        else:
            self.events = events
        self.emit_none = emit_none
        self.time_scale = time_scale

    def __iter__(self):
        prev_time = self.events[0][0]
        for timestamp, knobs in self.events:
            delay = (timestamp - prev_time) / self.time_scale
            if self.emit_none:
                steps = int(delay / 0.01)
                for _ in range(steps):
                    yield None
            else:
                if delay > 0:
                    time.sleep(delay)
            yield knobs
            prev_time = timestamp


# -------------------------------------------------------------------------------------
# Synth
# -------------------------------------------------------------------------------------


def _resolve_dials_and_settings(dials, settings, synth_func_params):
    """
    Resolve and validate dials and settings parameters for a synth.

    Parameters
    ----------
    dials : Optional[Union[str, List[str], Set[str]]]
        Parameters to treat as live knobs (controllable in real-time).
        If None, defaults to all parameters not in settings.
    settings : Optional[Union[str, List[str], Set[str]]]
        Parameters to exclude from live knobs.
        If None, defaults to all parameters not in dials.
    synth_func_params : Dict[str, Any]
        Dictionary of all available parameters from the synth function.

    Returns
    -------
    Tuple[Set[str], Set[str]]
        Resolved dials and settings as sets.

    Examples
    --------
    >>> assert (
    ...     _resolve_dials_and_settings(None, None, {'freq': 440, 'amp': 0.5})
    ...     == ({'freq', 'amp'}, set())
    ... )

    >>> _resolve_dials_and_settings('freq', None, {'freq': 440, 'amp': 0.5})
    ({'freq'}, {'amp'})

    >>> _resolve_dials_and_settings(None, ['amp'], {'freq': 440, 'amp': 0.5})
    ({'freq'}, {'amp'})

    >>> _resolve_dials_and_settings(['freq', 'amp'], ['amp'], {'freq': 440, 'amp': 0.5})
    ({'freq'}, {'amp'})
    """
    param_names = set(synth_func_params)

    dials_set = set(ensure_identifier_list(dials))
    settings_set = set(ensure_identifier_list(settings))

    unknown = (dials_set | settings_set) - param_names
    if unknown:
        raise ValueError(f"Unknown parameters specified: {', '.join(unknown)}")

    if dials is None and settings is None:
        return param_names, set()
    if dials is None:
        return param_names - settings_set, settings_set
    if settings is None:
        return dials_set, param_names - dials_set

    return dials_set - settings_set, settings_set


class Synth(MutableMapping):
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
        dials: Optional[Set[str]] = None,
        settings: Optional[Set[str]] = None,
        sr=DFLT_PYO_SR,
        nchnls=DFLT_PYO_NCHNLS,
        value_trans: Optional[Dict[str, Callable]] = None,
        record_on_start: bool = True,
        event_log_factory: RecordFactory = list,  # No argument factory that makes an Appendable
        audio="portaudio",
        verbosity=DFLT_PYO_VERBOSITY,
        synth_func_kwargs=None,
        **server_kwargs,
    ):
        """
        Parameters
        ----------
        synth_func : callable
            A function that returns a pyo object. The function should accept keyword arguments
            that are the parameters of the synthesizer.
        dials : Optional[Set[str]]
            Parameters to treat as live knobs (controllable in real-time).
            If None, defaults to all parameters.
        settings : Optional[Set[str]]
            Parameters to exclude from live knobs.
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
        if synth_func_kwargs:
            synth_func_attributes = synth_func.__dict__.copy()
            synth_func = partial(synth_func, **synth_func_kwargs)
            synth_func.__dict__.update(synth_func_attributes)
        self._synth_func_kwargs = synth_func_kwargs
        self._synth_func = synth_func
        self._server = None
        self.output = None
        self._synth_func_params = synth_func_defaults(synth_func)

        assert value_trans is None or (
            isinstance(value_trans, Mapping)
            and all(map(callable, value_trans.values()))
        ), f"value_trans must be a dict of callables. Was: {value_trans}"
        self._has_value_trans = value_trans is not None
        self._value_trans = value_trans

        # if function has some default dials or settings, use them if not specified in Synth arguments
        dials = dials or getattr(synth_func, "_default_dials", None)
        settings = settings or getattr(synth_func, "_default_settings", None)

        self._dials, self._settings = _resolve_dials_and_settings(
            dials, settings, self._synth_func_params
        )

        # Recording
        self._record_on_start = record_on_start
        self._event_log_factory = event_log_factory
        self._recording = False
        self._recording_start_time = None
        self._recorded_events = None

        self.knobs = Knobs(self._synth_func_params)
        # self._knob_defaults = {k: self._synth_func_params[k] for k in dials}

    def __call__(self, **updates):
        """
        Call the synthesizer with the given updates.
        """
        self.update(updates)

    def update(self, updates: KnobsDict):
        """
        Update the synthesizer parameters and record changes if recording is active.

        - Live parameters (SigTo) are updated smoothly.
        - Rebuild parameters (non-SigTo, like n_voices) trigger a full synth rebuild.

        This method handles parameter updates and recording in one place, unlike the
        previous design where recording was handled in the Knobs class.
        """
        live_updates = {}
        rebuild_updates = {}

        for k, v in updates.items():
            if self._has_value_trans and k in self._value_trans:
                before_v = v
                v = self._value_trans[k](v)
                print(f"{before_v} -> {v}")
            if k in self._dials:
                live_updates[k] = v
            elif k in self._settings:
                rebuild_updates[k] = v
            else:
                raise ValueError(f"Unknown parameter: {k}")

        # Record the update if recording is active
        if self._recording and (live_updates or rebuild_updates):
            self._record_update(updates)

        if live_updates:
            self.knobs.update(live_updates)

        if rebuild_updates:
            self._rebuild_graph(rebuild_updates)

    def _record_update(self, updates: KnobsDict):
        """
        Record parameter updates with timestamps.

        Parameters
        ----------
        updates : dict
            Dictionary of parameter updates to record
        """
        if not self._recording_start_time:
            raise RuntimeError(
                "Recording start time not set. Did you call start_recording()?"
            )

        rel_time = time.time() - self._recording_start_time
        self._recorded_events.append((rel_time, updates))

    def _rebuild_graph(self, rebuild_updates: KnobsDict):
        """
        Rebuild the synthesizer graph when structural parameters change.

        This happens when non-SigTo parameters are updated, requiring a complete
        reconstruction of the synthesis graph.
        """
        # Stop current graph
        if self.output is not None:
            self.output.stop()

        # Merge current parameter values
        merged_params = {}

        # Start from live current knob values
        for k, v in self.knobs.items():
            if isinstance(v, SigTo):
                merged_params[k] = v.value  # unwrap SigTo current value
            else:
                merged_params[k] = v

        # Also update with explicit new rebuild updates
        merged_params.update(rebuild_updates)

        # Store updated synth_func_params
        self._synth_func_params.update(merged_params)

        # Rebuild knobs
        new_initial_knob_params = {}
        for name, spec in merged_params.items():
            if name in self._dials:
                new_initial_knob_params[name] = dict_to_sigto(spec)
            else:
                new_initial_knob_params[name] = spec

        self.knobs = Knobs(new_initial_knob_params)

        # Rebuild output
        try:
            self.output = self._synth_func(**new_initial_knob_params)
        except TypeError as e:
            raise TypeError(
                f"Failed to rebuild synth function '{self._synth_func.__name__}'.\n"
                f"Original error: {e}"
            ) from e

        self.output.out()

        # Record rebuild if recording is active
        if self._recording:
            self._record_update(rebuild_updates)

    @property
    def _initial_knob_params(self):
        _initial_knob_params = {}
        for name, spec in self._synth_func_params.items():
            if name in self._dials:
                try:
                    _initial_knob_params[name] = dict_to_sigto(spec)
                except Exception as e:
                    raise type(e)(
                        f"Invalid initial value for knob '{name}': {spec}. "
                        "Expected a number or a dictionary with 'value', 'time', 'mul', and/or 'add' keys. "
                        f"Original error: {e}"
                    )
            else:
                _initial_knob_params[name] = spec

        return _initial_knob_params

    def _init_recorded_events(self):
        """Initialize the recording structure."""
        self._recorded_events = self._event_log_factory()
        # The first event should be the initial state of the synth_func
        self._recorded_events.append((0, serialize_knobs(self.knobs)))

    def start_recording(self):
        """
        Start recording parameter updates.

        Records the initial state of all knobs and prepares to record future updates.
        """
        if self.knobs is None:
            raise RuntimeError(
                "Cannot start recording without initializing knobs. "
                "Did you forget to call start() or do a with synth... block?"
            )

        self._recording = True
        self._recorded_events = self._event_log_factory()
        self._recording_start_time = time.time()

        # Record initial state of all knobs
        initial_state = serialize_knobs(self.knobs)
        self._recorded_events.append((0, initial_state))

    def stop_recording(self):
        """
        Stop recording parameter updates.

        Adds a final empty event with the current timestamp to mark the end of recording.
        """
        if not self._recording:
            return

        self._recording = False

        # Inject a final dummy event at current relative time
        if self._recorded_events and self._recording_start_time is not None:
            now = time.time()
            rel_now = now - self._recording_start_time
            self._recorded_events.append((rel_now, {}))

    def get_recording(self, process_recording: Callable = _json_friendly_records):
        """Get the recorded parameter updates."""
        return process_recording(self._recorded_events)

    def get_update_log(self):
        """
        Get the recorded parameter updates.

        Returns a list of updates without timestamps (unlike get_recording).
        If nothing has been recorded, returns an empty list.
        """
        if not self._recorded_events:
            return []

        # Extract just the update dictionaries, ignoring timestamps
        return [updates for _, updates in self._recorded_events if updates]

    def start(self):
        """
        Start the synthesizer.

        This boots the server, initializes knobs, starts the audio output,
        and optionally begins recording.
        """
        if self._server is None:
            self._server = Server(**self._server_kwargs).boot()

        _initial_knob_params = self._initial_knob_params.copy()

        self.knobs = Knobs(_initial_knob_params)

        try:
            self.output = self._synth_func(**_initial_knob_params)
        except TypeError as e:
            raise TypeError(
                f"Failed to initialize synth function '{self._synth_func.__name__}'.\n"
                f"Perhaps some arguments were wrongly wrapped into SigTo.\n"
                f"Consider :\n"
                f"  * using the dials argument to control which parameters are live.\n"
                f"  * using the settings argument to control which parameters are not live.\n"
                f"Alternatively, at function definition time, you can control this by:\n"
                f"  * using the @add_default_dials decorator to control which parameters are live.\n"
                f"  * using the @add_default_settings decorator to control which parameters are not live.\n"
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

    # Rendering events (playing recordings etc.) --------------------------------------

    def render_events(
        self,
        control_events=None,
        *,
        output_filepath=None,
        egress=lambda x: x,
        file_format="wav",
        suffix_buffer_seconds=0.0,
    ):
        """Render the control events to an audio file or return it via an egress function."""
        if not control_events:
            control_events = self.get_recording()
            if not control_events:
                raise ValueError("Nothing to render. No control events recorded.")

        TIME_IDX = 0
        base_time = control_events[0][TIME_IDX]
        normalized = [(t - base_time, knobs) for t, knobs in control_events]
        total_duration = normalized[-1][TIME_IDX] + suffix_buffer_seconds

        offline_server_kwargs = dict(self._server_kwargs, audio="offline")
        server = Server(**offline_server_kwargs).boot()
        table = NewTable(length=total_duration)

        all_keys = set(k for _, knobs in control_events for k in knobs)
        raw_params = {k: SigTo(value=0, time=0.01) for k in all_keys}
        synth_output = self._synth_func(**raw_params).out()
        table_recorder = TableRec(synth_output, table=table).play()

        def delay_func(dur):
            server.recordOptions(dur=dur)
            server.start()

        try:
            self._apply_event_sequence(normalized, delay_func, raw_params=raw_params)
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

    def play_events(
        self,
        events: Union[Iterator[Union[KnobsDict, None]], List[Tuple[float, KnobsDict]]],
        *,
        idle_sleep_time=0.01,
        inter_event_delay=0,
        tail_time=1.0,
    ):
        """
        Play an event stream, applying updates from either:
        - an iterator yielding KnobsDicts or None (real-time stream), or
        - a list of (timestamp, KnobsDict) pairs (automatically wrapped).
        """
        # Auto-wrap if we detect a list of (timestamp, KnobsDict) format
        if isinstance(events, list) and events and isinstance(events[0], tuple):
            ts, ev = events[0]
            if isinstance(ts, (int, float)) and isinstance(ev, dict):
                events = ReplayEvents(events)

        filtered_events = []
        curr_time = 0.0
        for e in events:
            if e is not None:
                filtered_events.append((curr_time, e))
                curr_time += inter_event_delay
            else:
                time.sleep(idle_sleep_time)

        def delay_func(dur):
            time.sleep(dur)

        with self:
            self._apply_event_sequence(filtered_events, delay_func)
            time.sleep(tail_time)

    def replay_events(self, control_events, *, tail_time=1.0):
        self.play_events(ReplayEvents(control_events), tail_time=tail_time)

    def _apply_event_sequence(self, events, delay_func, raw_params=None):
        """
        Apply a sequence of events using a delay function.

        Parameters
        ----------
        events : Iterable[Tuple[float, KnobsDict]]
            Timestamped knob updates.
        delay_func : Callable[[float], None]
            Function to call to delay between events (e.g., time.sleep or offline record trigger).
        raw_params : Optional[Dict[str, SigTo]]
            If provided, updates will be applied to this dict instead of self.knobs.
        """
        if not events:
            return

        TIME_IDX = 0
        base_time = events[0][TIME_IDX]
        normalized = [(t - base_time, knobs) for t, knobs in events]

        for i, (event_time, updates) in enumerate(normalized):
            next_time = normalized[i + 1][TIME_IDX] if i + 1 < len(normalized) else None
            dur = next_time - event_time if next_time is not None else None

            for key, val in updates.items():
                target = raw_params[key] if raw_params else self.knobs[key]

                if self._has_value_trans:
                    if not isinstance(val, dict):
                        val = self._value_trans[key](val)
                    else:
                        if "value" in val:
                            val["value"] = self._value_trans[key](val["value"])

                if isinstance(target, SigTo):
                    if isinstance(val, dict):
                        for attr in ["value", "time", "mul", "add"]:
                            if attr in val:
                                setattr(target, attr, val[attr])
                    else:
                        target.value = val
                elif raw_params is None:
                    self.knobs[key] = val

            if dur is not None:
                delay_func(dur)

    # Context manager support ---------------------------------------------------------

    def __del__(self):
        if self._server is not None:
            self._server.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    # Handling the decorator mechanisms -----------------------------------------------

    def __new__(cls, synth_func=None, **kwargs):
        """
        Create a new Synth instance or return a partial function for configuration.

        Trace on the deliberation of __new__ vs function design:
        https://github.com/thorwhalen/hum/discussions/4#discussioncomment-13011135

        """
        # Case 1: Used directly as @Synth
        if synth_func is not None and callable(synth_func):
            instance = super().__new__(cls)
            instance.__init__(synth_func, **kwargs)
            return instance
        # Case 2: Used with config as @Synth(knob_params=...)
        else:
            return partial(cls, **kwargs)

    # MutableMapping interface --------------------------------------------------------

    @property
    def __signature__(self):
        return Signature(
            parameters=[
                Parameter(name, kind=Parameter.KEYWORD_ONLY) for name in self.knobs
            ]
        )

    def __setitem__(self, key, value):
        self.update({key: value})

    def __getitem__(self, key):
        return self.knobs[key]

    def __delitem__(self, key):
        del self.knobs[key]

    def __iter__(self):
        return iter(self.knobs)

    def __len__(self):
        return len(self.knobs)

    def __repr__(self):
        return f"<Synth {list(self.knobs.keys())}>"
