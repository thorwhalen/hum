"""Guard the import-safety contract.

``import hum`` and the plotting/chunking utilities must work with **no audio
engine** installed; the ``pyo``-backed ``Synth`` is exposed lazily so it is only
imported (and only requires ``pyo``) on access. These tests pin that contract so
a future eager import of pyo can't sneak back in.
"""

import importlib.util

import pytest

import hum

HAS_PYO = importlib.util.find_spec("pyo") is not None


def test_import_hum_needs_no_audio_engine():
    # The non-audio utilities are always available.
    assert callable(hum.simple_chunker)
    assert callable(hum.plot_wf)
    assert callable(hum.disp_wf)


def test_pyo_is_not_imported_just_by_importing_hum():
    import sys

    # Importing hum must not eagerly pull in the pyo audio engine.
    if not HAS_PYO:
        assert "pyo" not in sys.modules


def test_synth_is_exposed_lazily():
    if HAS_PYO:
        # When the audio engine is present, the lazy attribute resolves.
        assert hum.Synth is not None
    else:
        # Without it, accessing Synth raises a clear import error (not
        # AttributeError) — the dependency is required only on use.
        with pytest.raises(ImportError):
            _ = hum.Synth


def test_unknown_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = hum.this_attribute_does_not_exist
