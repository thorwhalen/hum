# hum — developer context

`hum` is a Python **synthesizer**: a `Synth` wrapper around the [pyo](https://belangeo.github.io/pyo/)
audio engine that adds real-time parameter control ("knobs"), automatic
timestamped **event recording**, **replay**, and **rendering** to audio bytes.

## Architecture

- **CI-safe core (no audio engine):**
  - `hum/util.py` — sequence/chunking helpers (`simple_chunker`, ...), numpy-only.
  - `hum/extra_util.py` — stdlib helpers (e.g. `estimate_frequencies`).
  - `hum/utils/` — `plotting.py` (`plot_wf`, `disp_wf`), `date_ticks.py`.
- **pyo-backed (require the optional `audio` extra):**
  - `hum/pyo_util.py` — the `Synth` class, `ReplayEvents`, `round_event_times`,
    `DFLT_PYO_SR`. Uses `PyoObject` at module level, so it cannot import without pyo.
  - `hum/pyo_synths.py`, `hum/synth_funcs.py` — ready-made synth functions.

`hum/__init__.py` is the public surface: `simple_chunker`, `plot_wf`, `disp_wf`
eagerly; `Synth`, `ReplayEvents`, `round_event_times` **lazily** (PEP 562).

## The import-safety contract (key invariant)

`import hum` and the plotting/chunking utilities MUST work with **no pyo
installed**. Enforced by:

1. `pyo` is an **optional** dependency: `[project.optional-dependencies] audio = ["pyo"]`,
   NOT in core `dependencies` (pyo needs system libs — portaudio/portmidi — and
   has no universal wheels, so it breaks a plain CI install).
2. `Synth` (and the other pyo names) are exposed **lazily** via `__getattr__` in
   `hum/__init__.py` — never import `hum.pyo_util` at top level of `__init__` or
   `hum/utils/__init__.py`.
3. The pyo modules are listed in `[tool.wads.ci.testing].exclude_paths` so CI's
   `--doctest-modules` does not try to import them.
4. pyo tests start with `pytest.importorskip("pyo")`.

`hum/tests/test_import_safety.py` pins this contract — keep it green.

## Tests / CI

```bash
pip install -e ".[audio]"          # full, with the audio engine
python -m pytest hum/ --doctest-modules -q   # locally; pyo tests skip if absent
```

CI is the wads uv workflow; it installs only core deps, so it runs the CI-safe
doctests/tests and skips the pyo ones. The README's code blocks are illustrative
(not `>>>` doctests) and require `pip install hum[audio]`.

Handoffs live in `.claude/handoffs/` (gitignored).
