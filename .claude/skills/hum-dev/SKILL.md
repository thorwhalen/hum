---
name: hum-dev
description: Use when developing or modifying the `hum` package itself — changing the `Synth` engine (`hum/pyo_util.py`), adding synth functions (`hum/pyo_synths.py`, `hum/synth_funcs.py`), touching the plotting/chunking utilities, editing what `hum/__init__.py` exports, or adjusting packaging/CI. Triggers on editing files under the hum repo, "add a synth to hum", "fix the Synth class", or changing hum's dependencies/extras.
---

# Developing `hum`

`hum` is a `pyo`-backed synthesizer (`Synth` = parameter-recording wrapper over a
pyo synthesis graph). See `.claude/CLAUDE.md` for the architecture map.

## The one rule that constrains everything: import-safety

`import hum` MUST work with **no `pyo` installed** (pyo needs system libs and has
no universal wheels, so CI installs core deps only). Concretely:

- Keep `pyo` in `[project.optional-dependencies] audio`, never in core
  `dependencies`.
- Expose pyo-backed names (`Synth`, `ReplayEvents`, `round_event_times`)
  **lazily** via `__getattr__` in `hum/__init__.py`. Never add a top-level
  `from hum.pyo_util import ...` to `hum/__init__.py` or `hum/utils/__init__.py`
  (the latter used to do this and broke `import hum`).
- Any new pyo-importing module: add it to `[tool.wads.ci.testing].exclude_paths`
  (so `--doctest-modules` skips it) and gate its tests with
  `pytest.importorskip("pyo")`.
- `hum/tests/test_import_safety.py` pins all of this — run it, keep it green.

## Adding an optional (heavy / system) dependency

Same pattern as pyo: put it in an extra, import it lazily or guard it, exclude
its modules from CI doctests, and `importorskip` its tests. Don't make a plain
`pip install hum` require system libraries.

## Conventions

Owner style: functional where reasonable, keyword-only args beyond the first
positional, `DFLT_*` constants over magic numbers, a top-level docstring on
every module (ruff `D100`). Result/event data is plain `(timestamp, dict)`
tuples — keep that simple, serializable shape.

## Verify

```bash
python -m pytest hum/ --doctest-modules -q   # pyo tests skip if pyo absent
pip install -e ".[audio]" && python -m pytest hum/  # full, with audio engine
```
