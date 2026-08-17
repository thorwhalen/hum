"""Partition a recorded control-event stream into dials and settings.

A ``Synth`` records *every* parameter it knows about: its first recorded event is
a full snapshot of the initial state, so a recording always mentions the
*settings* (structural, non-live parameters such as a waveform name) alongside
the *dials* (live, continuously-controllable parameters such as a frequency).

When such a recording is rendered offline, only the dials may be driven by a
control signal. Wrapping a setting in one hands the synth function a signal
object where it expected a plain value -- a synth that uses a setting as a dict
key then fails with ``TypeError: unhashable type``, and one that merely compares
it misbehaves silently.

:func:`plan_render_params` computes that split. It is deliberately free of any
``pyo`` import so the rule can be reasoned about -- and tested -- without the
optional audio engine installed.
"""

from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Sequence,
    Tuple,
)
from warnings import warn

__all__ = ["RenderParamPlan", "RenderParamWarning", "plan_render_params"]


ControlEvent = Tuple[float, Dict[str, Any]]


class RenderParamWarning(UserWarning):
    """Raised (as a warning) when a recorded event cannot be honored on render."""


class RenderParamPlan(NamedTuple):
    """How a recorded control-event stream maps onto a single render graph.

    Attributes
    ----------
    dial_keys : frozenset of str
        The dials actually mentioned by the recording. These -- and only these --
        get a live control signal.
    settings_values : dict
        Settings values baked into the graph at build time.
    dial_events : list of (float, dict)
        The event stream with every non-dial key removed. Timestamps and event
        count are preserved (an event may become empty), so render timing is
        unaffected by the filtering.
    ignored_settings_changes : list of (float, dict)
        Settings changes that appeared *after* the initial snapshot and were
        dropped. A single offline graph cannot express the rebuild they need.
    unknown_keys : frozenset of str
        Recorded keys that are neither a dial nor a setting; also dropped.
    """

    dial_keys: FrozenSet[str]
    settings_values: Dict[str, Any]
    dial_events: List[ControlEvent]
    ignored_settings_changes: List[ControlEvent]
    unknown_keys: FrozenSet[str]


def plan_render_params(
    control_events: Sequence[ControlEvent],
    *,
    dials: Iterable[str],
    settings: Iterable[str],
    settings_defaults: Mapping[str, Any] = None,
    warn_on_ignored: bool = True,
) -> RenderParamPlan:
    """Split a recorded control-event stream into dials to drive and settings to bake.

    Settings are taken from the recording's *initial snapshot* (its first event),
    falling back to ``settings_defaults`` for any setting the recording never
    mentions. Settings changes later in the stream are dropped with a warning:
    honoring one would require rebuilding the synthesis graph mid-render, which a
    single offline render cannot express. (This mirrors the live path, where such
    a change rebuilds the graph.)

    Parameters
    ----------
    control_events : sequence of (timestamp, updates)
        The recorded stream, oldest first.
    dials : iterable of str
        Names of the live parameters.
    settings : iterable of str
        Names of the structural parameters.
    settings_defaults : mapping, optional
        Fallback values for settings absent from the recording -- typically the
        synth function's own parameter defaults.
    warn_on_ignored : bool
        Whether to emit :class:`RenderParamWarning` for dropped keys.

    Returns
    -------
    RenderParamPlan

    Examples
    --------
    A recording of one dial (``freq``) and one setting (``waveform``). The
    setting is baked from the initial snapshot, never wrapped as a dial:

    >>> events = [(0.0, {'freq': 440, 'waveform': 'sine'}), (1.0, {'freq': 660})]
    >>> plan = plan_render_params(events, dials={'freq'}, settings={'waveform'})
    >>> sorted(plan.dial_keys)
    ['freq']
    >>> plan.settings_values
    {'waveform': 'sine'}
    >>> plan.dial_events
    [(0.0, {'freq': 440}), (1.0, {'freq': 660})]

    A setting the recording never mentions falls back to its default:

    >>> plan = plan_render_params(
    ...     [(0.0, {'freq': 440})],
    ...     dials={'freq'},
    ...     settings={'waveform'},
    ...     settings_defaults={'freq': 440, 'waveform': 'sine'},
    ... )
    >>> plan.settings_values
    {'waveform': 'sine'}

    A settings change *after* the initial snapshot is dropped (with a warning),
    and the event keeps its slot so the render timing does not shift:

    >>> import warnings
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter('always')
    ...     plan = plan_render_params(
    ...         [(0.0, {'freq': 440, 'waveform': 'sine'}), (1.0, {'waveform': 'square'})],
    ...         dials={'freq'},
    ...         settings={'waveform'},
    ...     )
    >>> plan.settings_values
    {'waveform': 'sine'}
    >>> plan.dial_events
    [(0.0, {'freq': 440}), (1.0, {})]
    >>> plan.ignored_settings_changes
    [(1.0, {'waveform': 'square'})]
    >>> len(caught), issubclass(caught[0].category, RenderParamWarning)
    (1, True)
    """
    dials, settings = frozenset(dials), frozenset(settings)
    settings_defaults = settings_defaults or {}

    recorded_keys = {k for _, updates in control_events for k in updates}
    dial_keys = recorded_keys & dials
    unknown_keys = recorded_keys - dials - settings

    initial_snapshot = control_events[0][1] if control_events else {}
    settings_values = {
        k: initial_snapshot[k] if k in initial_snapshot else settings_defaults[k]
        for k in sorted(settings)
        if k in initial_snapshot or k in settings_defaults
    }

    dial_events = [
        (t, {k: v for k, v in updates.items() if k in dial_keys})
        for t, updates in control_events
    ]
    ignored_settings_changes = [
        (t, ignored)
        for t, updates in control_events[1:]
        if (ignored := {k: v for k, v in updates.items() if k in settings})
    ]

    if warn_on_ignored:
        _warn_about_dropped_keys(ignored_settings_changes, unknown_keys)

    return RenderParamPlan(
        dial_keys=frozenset(dial_keys),
        settings_values=settings_values,
        dial_events=dial_events,
        ignored_settings_changes=ignored_settings_changes,
        unknown_keys=frozenset(unknown_keys),
    )


def _warn_about_dropped_keys(
    ignored_settings_changes: Sequence[ControlEvent],
    unknown_keys: Iterable[str],
) -> None:
    """Warn about recorded keys that the render cannot honor."""
    if ignored_settings_changes:
        changed = sorted(
            {k for _, updates in ignored_settings_changes for k in updates}
        )
        warn(
            f"Ignoring {len(ignored_settings_changes)} mid-stream change(s) to "
            f"settings {changed}: a single offline render builds one synthesis "
            f"graph, so only the settings of the initial snapshot are honored. "
            f"Split the recording and render each segment separately if you need "
            f"the change.",
            RenderParamWarning,
            stacklevel=3,
        )
    unknown_keys = sorted(unknown_keys)
    if unknown_keys:
        warn(
            f"Ignoring recorded key(s) {unknown_keys}: they are neither a dial "
            f"nor a setting of the synth function.",
            RenderParamWarning,
            stacklevel=3,
        )
