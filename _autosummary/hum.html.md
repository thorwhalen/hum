# hum

Tools to create and manipulate audio.

`import hum` and the plotting/chunking utilities work with no audio engine
installed. The synthesis features (`Synth` and the `hum.pyo_util` /
`hum.pyo_synths` / `hum.synth_funcs` modules) require the optional `pyo`
audio engine — install it with `pip install hum[audio]` (pyo itself needs the
system libraries portaudio and portmidi). `Synth` is exposed lazily here, so
it is only imported (and only requires pyo) when you actually use it.

### Modules

| [`event_params`](hum.event_params.html.md#module-hum.event_params)   | Partition a recorded control-event stream into dials and settings.   |
|-----------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| [`extra_util`](hum.extra_util.html.md#module-hum.extra_util)       | Extra utils that require more dependencies                           |
| [`util`](hum.util.html.md#module-hum.util)                   | A few general utils                                                  |
| [`utils`](hum.utils.html.md#module-hum.utils)                 | General utils                                                        |
