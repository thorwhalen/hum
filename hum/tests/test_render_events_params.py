"""Regression tests for the dial/settings split on render (issue #7).

``Synth.render_events`` used to build its live control signals from *every*
recorded key::

    all_keys = {k for _, knobs in control_events for k in knobs}
    raw_params = {k: SigTo(value=0, time=0.01) for k in all_keys}

A recording's first event is a full snapshot of the synth state, so it always
mentions the *settings* too. Every setting therefore reached the synth function
as a ``SigTo`` instead of its plain value -- crashing any synth that uses a
setting as a dict key with ``TypeError: unhashable type: 'SigTo'``, and silently
misbehaving in one that merely compares it.

The other half of the fix -- easy to miss -- is that the *event stream* handed to
``_apply_event_sequence`` must be filtered too: it looks every key up in
``raw_params``, which is now dial-only, so a settings key would ``KeyError``.

A third thing the split has to get right, found in review: the *fallback* for a
setting the recording never mentions must be the synth function's own defaults,
not ``Synth._synth_func_params`` -- which is live state that ``_rebuild_graph``
overwrites with the current values. Using it made an offline render depend on
what the session happened to do beforehand.

The first group of tests below exercises the split logic directly through the
pyo-free ``hum.event_params`` helper, so they run (and gate) in CI, where pyo is
not installed. The remaining groups drive ``render_events`` end to end against a
fake pyo, reproducing the reported crash and pinning the fallback source.
"""

import pytest

from hum.event_params import RenderParamWarning, plan_render_params


DIALS = {"freq", "amp"}
SETTINGS = {"waveform", "n_voices"}
DEFAULTS = {"freq": 440, "amp": 0.5, "waveform": "sine", "n_voices": 1}


def _plan(control_events, **kwargs):
    kwargs = dict(
        dict(dials=DIALS, settings=SETTINGS, settings_defaults=DEFAULTS), **kwargs
    )
    return plan_render_params(control_events, **kwargs)


# --- the split itself ----------------------------------------------------------------


def test_settings_never_become_dials():
    # The defect: `waveform` and `n_voices` were wrapped in a SigTo along with
    # the dials, because the initial snapshot mentions every parameter.
    plan = _plan([(0.0, dict(DEFAULTS)), (1.0, {"freq": 660})])
    assert plan.dial_keys == frozenset({"freq", "amp"})
    assert not (plan.dial_keys & SETTINGS)


def test_settings_are_baked_from_the_initial_snapshot():
    plan = _plan([(0.0, {"freq": 440, "waveform": "square", "n_voices": 3})])
    assert plan.settings_values == {"n_voices": 3, "waveform": "square"}


def test_settings_absent_from_the_recording_fall_back_to_defaults():
    # A recording that mentions no setting at all still needs the synth built
    # with its own defaults, not with SigTo-wrapped placeholders.
    plan = _plan([(0.0, {"freq": 440}), (1.0, {"freq": 660})])
    assert plan.settings_values == {"n_voices": 1, "waveform": "sine"}
    assert plan.dial_keys == frozenset({"freq"})


def test_a_dial_first_seen_mid_stream_is_still_driven():
    plan = _plan([(0.0, {"freq": 440}), (1.0, {"amp": 0.9})])
    assert plan.dial_keys == frozenset({"freq", "amp"})


# --- the half that is easy to miss: the event stream must be filtered too -------------


def test_events_handed_to_the_applier_carry_no_settings_key():
    # `_apply_event_sequence` does `raw_params[key]`; raw_params is dial-only,
    # so any surviving settings key would raise KeyError at render time.
    events = [
        (0.0, dict(DEFAULTS)),
        (1.0, {"freq": 660, "waveform": "square"}),
        (2.0, {"n_voices": 4}),
    ]
    with pytest.warns(RenderParamWarning):
        plan = _plan(events)
    surviving = {k for _, updates in plan.dial_events for k in updates}
    assert not (surviving & SETTINGS)
    assert surviving <= plan.dial_keys


def test_a_mid_stream_settings_change_warns_rather_than_raising():
    events = [(0.0, dict(DEFAULTS)), (1.0, {"waveform": "square"})]
    with pytest.warns(RenderParamWarning, match="waveform"):
        plan = _plan(events)
    assert plan.ignored_settings_changes == [(1.0, {"waveform": "square"})]
    # The initial snapshot's value is what gets baked, not the later one.
    assert plan.settings_values["waveform"] == "sine"


def test_settings_in_the_initial_snapshot_alone_do_not_warn(recwarn):
    _plan([(0.0, dict(DEFAULTS)), (1.0, {"freq": 660})])
    assert [w for w in recwarn.list if issubclass(w.category, RenderParamWarning)] == []


def test_filtering_preserves_event_count_and_timestamps():
    # Dropping events outright would shorten the render; each event keeps its
    # slot (possibly empty) so the inter-event delays are unchanged.
    events = [
        (10.0, dict(DEFAULTS)),
        (11.5, {"freq": 660}),
        (13.0, {"waveform": "square"}),
    ]
    with pytest.warns(RenderParamWarning):
        plan = _plan(events)
    assert [t for t, _ in plan.dial_events] == [10.0, 11.5, 13.0]
    assert plan.dial_events[-1] == (13.0, {})


def test_keys_that_are_neither_dial_nor_setting_are_dropped_with_a_warning():
    with pytest.warns(RenderParamWarning, match="mystery"):
        plan = _plan([(0.0, {"freq": 440, "mystery": 1})])
    assert plan.unknown_keys == frozenset({"mystery"})
    assert plan.dial_keys == frozenset({"freq"})


def test_empty_recording_is_handled():
    plan = _plan([])
    assert plan.dial_keys == frozenset()
    assert plan.dial_events == []
    assert plan.settings_values == dict(
        n_voices=DEFAULTS["n_voices"], waveform=DEFAULTS["waveform"]
    )


# --- end to end through render_events ------------------------------------------------

WAVEFORMS = {"sine": 0, "square": 1, "triangle": 2}


@pytest.fixture
def synth_factory(fake_pyo_util):
    """Build Synths whose synth function uses its setting as a dict key (issue #7).

    Uses ``fake_pyo_util``, not ``pyo_util``: these tests drive the render graph,
    and a real ``pyo.TableRec`` rejects the fake output object below. See
    ``conftest.py`` for the full reason.

    Returns ``(make_synth, calls)`` -- a factory rather than a single instance,
    because pinning the settings fallback needs a *fresh* Synth to compare a
    *used* one against.
    """
    calls = []

    class FakeOutput:
        def out(self):
            return self

        def stop(self):
            return self

    def synth_func(freq=440, waveform="sine"):
        # The reported crash: a SigTo-wrapped `waveform` is unhashable.
        WAVEFORMS[waveform]
        calls.append({"freq": freq, "waveform": waveform})
        return FakeOutput()

    def make_synth():
        return fake_pyo_util.Synth(synth_func, dials="freq", settings="waveform")

    return make_synth, calls


@pytest.fixture
def synth_and_calls(synth_factory):
    """A single fresh Synth from :func:`synth_factory`, plus its call log."""
    make_synth, calls = synth_factory
    return make_synth(), calls


def test_render_events_passes_settings_through_as_plain_values(
    synth_and_calls, tmp_path
):
    synth, calls = synth_and_calls
    events = [(0.0, {"freq": 440, "waveform": "square"}), (1.0, {"freq": 660})]

    audio = synth.render_events(events, output_filepath=str(tmp_path / "out.wav"))

    assert audio  # rendered without TypeError: unhashable type: 'SigTo'
    (call,) = calls
    assert call["waveform"] == "square"  # a plain string, not a SigTo
    assert isinstance(call["waveform"], str)
    assert type(call["freq"]).__name__ == "SigTo"  # the dial *is* driven


def test_render_events_survives_a_mid_stream_settings_change(synth_and_calls, tmp_path):
    # Before the fix's second half this raised KeyError('waveform') inside
    # _apply_event_sequence, once raw_params became dial-only.
    synth, calls = synth_and_calls
    events = [
        (0.0, {"freq": 440, "waveform": "sine"}),
        (1.0, {"freq": 660, "waveform": "triangle"}),
    ]

    with pytest.warns(RenderParamWarning, match="waveform"):
        audio = synth.render_events(events, output_filepath=str(tmp_path / "out.wav"))

    assert audio
    assert calls[0]["waveform"] == "sine"


# --- the settings fallback must be the function's defaults, not the live state --------


def test_the_synth_functions_defaults_survive_a_live_settings_change(synth_factory):
    # `_synth_func_params` is live state -- `_rebuild_graph` writes the current
    # values into it, and the initial `Knobs` shares the very dict object. The
    # defaults a render falls back to must be a separate, pristine copy.
    make_synth, _ = synth_factory
    synth = make_synth()
    defaults_before = dict(synth._synth_func_defaults)

    synth.update({"waveform": "square"})  # a settings change -> _rebuild_graph

    assert synth._synth_func_params["waveform"] == "square"  # live state moved
    assert synth._synth_func_defaults == defaults_before  # the defaults did not
    assert synth._synth_func_defaults["waveform"] == "sine"


def test_render_of_a_dial_only_stream_does_not_depend_on_session_history(
    synth_factory, tmp_path
):
    # Rendering must be a pure function of the event stream. Falling back to
    # `_synth_func_params` baked whatever the session last happened to be set
    # to, so a dial-only stream (exactly what a client that pre-filters its
    # events sends) rendered 'square' on a used Synth and 'sine' on a fresh one
    # -- silently, and with no way to reproduce it from the events alone.
    make_synth, calls = synth_factory
    dial_only_events = [(0.0, {"freq": 440}), (1.0, {"freq": 660})]

    used = make_synth()
    used.update({"waveform": "square"})  # a live settings change, then reuse
    calls.clear()
    used.render_events(dial_only_events, output_filepath=str(tmp_path / "used.wav"))
    waveform_from_used = calls[-1]["waveform"]

    calls.clear()
    fresh = make_synth()
    fresh.render_events(dial_only_events, output_filepath=str(tmp_path / "fresh.wav"))
    waveform_from_fresh = calls[-1]["waveform"]

    assert waveform_from_used == waveform_from_fresh
    assert waveform_from_used == "sine"  # the synth function's own default
