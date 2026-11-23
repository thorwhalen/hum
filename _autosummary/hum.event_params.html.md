# hum.event_params

Partition a recorded control-event stream into dials and settings.

A `Synth` records *every* parameter it knows about: its first recorded event is
a full snapshot of the initial state, so a recording always mentions the
*settings* (structural, non-live parameters such as a waveform name) alongside
the *dials* (live, continuously-controllable parameters such as a frequency).

When such a recording is rendered offline, only the dials may be driven by a
control signal. Wrapping a setting in one hands the synth function a signal
object where it expected a plain value – a synth that uses a setting as a dict
key then fails with `TypeError: unhashable type`, and one that merely compares
it misbehaves silently.

[`plan_render_params()`](#hum.event_params.plan_render_params) computes that split. It is deliberately free of any
`pyo` import so the rule can be reasoned about – and tested – without the
optional audio engine installed.

### Functions

| [`plan_render_params`](#hum.event_params.plan_render_params)(control_events, \*, dials, ...)   | Split a recorded control-event stream into dials to drive and settings to bake.   |
|-------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|

### Classes

| [`RenderParamPlan`](#hum.event_params.RenderParamPlan)(dial_keys, settings_values, ...)   | How a recorded control-event stream maps onto a single render graph.   |
|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|

### Exceptions

| [`RenderParamWarning`](#hum.event_params.RenderParamWarning)   | Raised (as a warning) when a recorded event cannot be honored on render.   |
|-----------------------------------------------------------------------|----------------------------------------------------------------------------|

### *class* hum.event_params.RenderParamPlan(dial_keys, settings_values, dial_events, ignored_settings_changes, unknown_keys)

Bases: [`NamedTuple`](https://docs.python.org/3/library/typing.html#typing.NamedTuple)

How a recorded control-event stream maps onto a single render graph.

#### dial_keys

The dials actually mentioned by the recording. These – and only these –
get a live control signal.

* **Type:**
  [*frozenset*](https://docs.python.org/3/builtins/stdtypes.html#frozenset) of [*str*](https://docs.python.org/3/builtins/stdtypes.html#str)

#### settings_values

Settings values baked into the graph at build time.

* **Type:**
  [*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### dial_events

The event stream with every non-dial key removed. Timestamps and event
count are preserved (an event may become empty), so render timing is
unaffected by the filtering.

* **Type:**
  [*list*](https://docs.python.org/3/builtins/stdtypes.html#list) of ([*float*](https://docs.python.org/3/builtins/functions.html#float), [*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict))

#### ignored_settings_changes

Settings changes that appeared *after* the initial snapshot and were
dropped. A single offline graph cannot express the rebuild they need.

* **Type:**
  [*list*](https://docs.python.org/3/builtins/stdtypes.html#list) of ([*float*](https://docs.python.org/3/builtins/functions.html#float), [*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict))

#### unknown_keys

Recorded keys that are neither a dial nor a setting; also dropped.

* **Type:**
  [*frozenset*](https://docs.python.org/3/builtins/stdtypes.html#frozenset) of [*str*](https://docs.python.org/3/builtins/stdtypes.html#str)

#### dial_events *: [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[float](https://docs.python.org/3/builtins/functions.html#float), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]]*

Alias for field number 2

#### dial_keys *: [FrozenSet](https://docs.python.org/3/library/typing.html#typing.FrozenSet)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)]*

Alias for field number 0

#### ignored_settings_changes *: [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[float](https://docs.python.org/3/builtins/functions.html#float), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]]*

Alias for field number 3

#### settings_values *: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]*

Alias for field number 1

#### unknown_keys *: [FrozenSet](https://docs.python.org/3/library/typing.html#typing.FrozenSet)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)]*

Alias for field number 4

### *exception* hum.event_params.RenderParamWarning

Bases: [`UserWarning`](https://docs.python.org/3/builtins/exceptions.html#UserWarning)

Raised (as a warning) when a recorded event cannot be honored on render.

### hum.event_params.plan_render_params(control_events, , dials, settings, settings_defaults=None, warn_on_ignored=True)

Split a recorded control-event stream into dials to drive and settings to bake.

Settings are taken from the recording’s *initial snapshot* (its first event),
falling back to `settings_defaults` for any setting the recording never
mentions. Settings changes later in the stream are dropped with a warning:
honoring one would require rebuilding the synthesis graph mid-render, which a
single offline render cannot express. (This mirrors the live path, where such
a change rebuilds the graph.)

* **Parameters:**
  * **control_events** ([`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`Tuple`](https://docs.python.org/3/library/typing.html#typing.Tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`Dict`](https://docs.python.org/3/library/typing.html#typing.Dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]]) – The recorded stream, oldest first.
  * **dials** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Names of the live parameters.
  * **settings** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Names of the structural parameters.
  * **settings_defaults** ([`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Fallback values for settings absent from the recording – typically the
    synth function’s own parameter defaults.
  * **warn_on_ignored** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to emit [`RenderParamWarning`](#hum.event_params.RenderParamWarning) for dropped keys.
* **Return type:**
  [`RenderParamPlan`](#hum.event_params.RenderParamPlan)

### Examples

A recording of one dial (`freq`) and one setting (`waveform`). The
setting is baked from the initial snapshot, never wrapped as a dial:

```pycon
>>> events = [(0.0, {'freq': 440, 'waveform': 'sine'}), (1.0, {'freq': 660})]
>>> plan = plan_render_params(events, dials={'freq'}, settings={'waveform'})
>>> sorted(plan.dial_keys)
['freq']
>>> plan.settings_values
{'waveform': 'sine'}
>>> plan.dial_events
[(0.0, {'freq': 440}), (1.0, {'freq': 660})]
```

A setting the recording never mentions falls back to its default:

```pycon
>>> plan = plan_render_params(
...     [(0.0, {'freq': 440})],
...     dials={'freq'},
...     settings={'waveform'},
...     settings_defaults={'freq': 440, 'waveform': 'sine'},
... )
>>> plan.settings_values
{'waveform': 'sine'}
```

A settings change *after* the initial snapshot is dropped (with a warning),
and the event keeps its slot so the render timing does not shift:

```pycon
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
```
