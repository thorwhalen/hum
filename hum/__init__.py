"""
Tools to create and manipulate audio.

``import hum`` and the plotting/chunking utilities work with no audio engine
installed. The synthesis features (`Synth` and the ``hum.pyo_util`` /
``hum.pyo_synths`` / ``hum.synth_funcs`` modules) require the optional ``pyo``
audio engine — install it with ``pip install hum[audio]`` (pyo itself needs the
system libraries portaudio and portmidi). ``Synth`` is exposed lazily here, so
it is only imported (and only requires pyo) when you actually use it.
"""

from hum.util import simple_chunker
from hum.utils import plot_wf, disp_wf

# `Synth` lives in hum.pyo_util, which requires the optional `pyo` audio engine.
# Expose it lazily (PEP 562) so `import hum` does not require pyo.
_LAZY_FROM_PYO_UTIL = {"Synth", "ReplayEvents", "round_event_times"}


def __getattr__(name):
    if name in _LAZY_FROM_PYO_UTIL:
        import importlib

        return getattr(importlib.import_module("hum.pyo_util"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({"simple_chunker", "plot_wf", "disp_wf", *_LAZY_FROM_PYO_UTIL})
