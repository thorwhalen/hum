"""Regression tests for Synth event recording (issue #5).

One logical parameter change — one ``Synth.update(...)`` / ``s(...)`` call —
must produce exactly one recorded event. Before the fix, any update that
touched a *settings* parameter was recorded twice: once by ``Synth.update()``
(the logging chokepoint, recording the full update) and once more by
``Synth._rebuild_graph()`` (re-recording the settings subset a few
milliseconds later), yielding near-duplicate events like::

    (2.0079, {'freq': 440, 'waveform': 'triangle'}),
    (2.0084, {'waveform': 'triangle'}),   # <-- duplicate

These tests run fully headless: no pyo server is booted and no real pyo
objects are created. When pyo is not installed at all (e.g. CI), a minimal
fake ``pyo`` module is injected just for this module, then cleaned up so the
import-safety tests still observe the real no-pyo environment.
"""

import importlib
import importlib.machinery
import importlib.util
import sys
import types

import pytest

HAS_PYO = importlib.util.find_spec("pyo") is not None


def _make_fake_pyo_module():
    """A minimal stand-in for pyo: just the names hum.pyo_util needs at import."""
    fake_pyo = types.ModuleType("pyo")
    # A real spec so importlib.util.find_spec("pyo") keeps working on this fake.
    fake_pyo.__spec__ = importlib.machinery.ModuleSpec("pyo", loader=None)

    class PyoObject:
        pass

    class SigTo(PyoObject):
        def __init__(self, value=0, time=0.025, init=None, mul=1, add=0):
            self.value, self.time, self.mul, self.add = value, time, mul, add

    class Server:
        def __init__(self, *args, **kwargs):
            pass

        def boot(self):
            return self

        def start(self):
            return self

        def stop(self):
            return self

        def shutdown(self):
            return self

    fake_pyo.PyoObject = PyoObject
    fake_pyo.SigTo = SigTo
    fake_pyo.Server = Server
    return fake_pyo


@pytest.fixture(scope="module")
def pyo_util():
    """Import hum.pyo_util — with real pyo when available, else a minimal fake."""
    if HAS_PYO:
        yield importlib.import_module("hum.pyo_util")
        return

    # No pyo installed (the CI case): inject a fake, import, then clean up so
    # other tests (notably test_import_safety) see the pristine environment.
    sys.modules["pyo"] = _make_fake_pyo_module()
    try:
        yield importlib.import_module("hum.pyo_util")
    finally:
        sys.modules.pop("pyo", None)
        sys.modules.pop("hum.pyo_util", None)
        import hum

        if hasattr(hum, "pyo_util"):
            delattr(hum, "pyo_util")


@pytest.fixture
def synth(pyo_util, monkeypatch):
    """A recording Synth with a dial ('freq') and settings ('waveform', 'attack').

    Headless: the synth function returns a fake output object (no audio graph),
    dial values are kept as plain numbers instead of pyo.SigTo signals (a fake
    dial — creating a SigTo requires a booted server), and the server is never
    started; recording is driven directly via start_recording().
    """

    class FakeOutput:
        def out(self):
            return self

        def stop(self):
            return self

    def synth_func(freq=440, attack=0.01, waveform="sine"):
        return FakeOutput()

    monkeypatch.setattr(pyo_util, "dict_to_sigto", lambda spec: spec)

    s = pyo_util.Synth(synth_func, dials="freq", settings="waveform attack")
    s.start_recording()
    return s


def _change_events(synth):
    """Recorded events minus the initial-state event; assert the wrapping."""
    events = synth.get_recording()
    # Schema: list of (relative_time, updates_dict), first event is the
    # initial state at time 0, times non-decreasing.
    assert all(
        isinstance(t, (int, float)) and isinstance(knobs, dict) for t, knobs in events
    )
    times = [t for t, _ in events]
    assert times == sorted(times)
    assert times[0] == 0
    return events[1:]


def test_dial_only_change_records_exactly_one_event(synth):
    synth.update({"freq": 660})
    assert [knobs for _, knobs in _change_events(synth)] == [{"freq": 660}]


def test_settings_only_change_records_exactly_one_event(synth):
    # Before the fix this recorded twice: once in update(), once in
    # _rebuild_graph() — both with the same content.
    synth.update({"waveform": "square", "attack": 0.5})
    assert [knobs for _, knobs in _change_events(synth)] == [
        {"waveform": "square", "attack": 0.5}
    ]


def test_dial_plus_setting_change_records_exactly_one_event(synth):
    # The issue's signature case: one call mixing a dial and a setting was
    # recorded as {'freq': 440, 'waveform': 'triangle'} AND, milliseconds
    # later, {'waveform': 'triangle'} again.
    synth.update({"freq": 440, "waveform": "triangle"})
    assert [knobs for _, knobs in _change_events(synth)] == [
        {"freq": 440, "waveform": "triangle"}
    ]


def test_sequence_of_changes_records_one_event_each(synth):
    # The full sequence from issue #5, headless: each logical change must
    # appear exactly once, in order, with its full update dict.
    synth.update({"freq": 660})
    synth.update({"freq": 440, "waveform": "triangle"})
    synth.update({"waveform": "square", "attack": 0.5})
    synth.update({"waveform": "sine"})
    assert [knobs for _, knobs in _change_events(synth)] == [
        {"freq": 660},
        {"freq": 440, "waveform": "triangle"},
        {"waveform": "square", "attack": 0.5},
        {"waveform": "sine"},
    ]
