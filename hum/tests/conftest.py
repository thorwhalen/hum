"""Shared test fixtures: import ``hum.pyo_util`` with or without a real ``pyo``.

The audio engine ``pyo`` is an optional extra (``pip install hum[audio]``) and is
absent from CI, yet the pure logic in ``hum.pyo_util`` (recording, event
partitioning, event replay) is worth gating there. So when pyo is missing we
inject a minimal fake just long enough to import the module, then clean it up so
``test_import_safety`` still observes a pristine, pyo-free environment.

The fake mirrors the two real behaviours these tests depend on:

- ``SigTo`` is **unhashable** (like ``pyo.SigTo``, which defines ``__eq__`` for
  signal arithmetic and so gets ``__hash__ = None``). That is what turns a
  SigTo-wrapped *setting* into ``TypeError: unhashable type: 'SigTo'`` inside a
  synth function that uses the setting as a dict key (issue #7).
- ``NewTable.save`` writes a real file, since ``render_events`` reads it back.
"""

import importlib
import importlib.machinery
import importlib.util
import sys
import types

import pytest

HAS_PYO = importlib.util.find_spec("pyo") is not None


def make_fake_pyo_module():
    """A minimal stand-in for pyo: the names ``hum.pyo_util`` needs."""
    fake_pyo = types.ModuleType("pyo")
    # A real spec so importlib.util.find_spec("pyo") keeps working on this fake.
    fake_pyo.__spec__ = importlib.machinery.ModuleSpec("pyo", loader=None)

    class PyoObject:
        pass

    class SigTo(PyoObject):
        def __init__(self, value=0, time=0.025, init=None, mul=1, add=0):
            self.value, self.time, self.mul, self.add = value, time, mul, add

        def __eq__(self, other):  # pragma: no cover - shape, not behaviour
            return NotImplemented

        # Defining __eq__ drops __hash__; real pyo signals are unhashable too.
        __hash__ = None

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

        def recordOptions(self, **kwargs):
            return self

    class NewTable:
        def __init__(self, length=0, **kwargs):
            self.length = length

        def save(self, path, *args, **kwargs):
            with open(path, "wb") as f:
                f.write(b"fake-audio")

    class TableRec:
        def __init__(self, obj, table=None, **kwargs):
            self.obj, self.table = obj, table

        def play(self):
            return self

        def stop(self):
            return self

    fake_pyo.PyoObject = PyoObject
    fake_pyo.SigTo = SigTo
    fake_pyo.Server = Server
    fake_pyo.NewTable = NewTable
    fake_pyo.TableRec = TableRec
    return fake_pyo


@pytest.fixture(scope="module")
def pyo_util():
    """Import hum.pyo_util -- with real pyo when available, else a minimal fake."""
    if HAS_PYO:
        yield importlib.import_module("hum.pyo_util")
        return

    # No pyo installed (the CI case): inject a fake, import, then clean up so
    # other tests (notably test_import_safety) see the pristine environment.
    sys.modules["pyo"] = make_fake_pyo_module()
    try:
        yield importlib.import_module("hum.pyo_util")
    finally:
        sys.modules.pop("pyo", None)
        sys.modules.pop("hum.pyo_util", None)
        import hum

        if hasattr(hum, "pyo_util"):
            delattr(hum, "pyo_util")
