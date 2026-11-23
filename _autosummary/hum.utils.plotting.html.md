# hum.utils.plotting

Plot utils

### Functions

| [`disp_wf`](#hum.utils.plotting.disp_wf)(wf[, sr, autoplay, wf_plot_func])   | Display waveform in Jupyter notebook   |
|----------------------------------------------------------------------------------------------|----------------------------------------|
| [`getmodulename`](#hum.utils.plotting.getmodulename)(obj[, default])               | Get name of module of object           |
| `plot_wf`(wf[, sr, figsize, offset_s, ax])                                                   |                                        |

### hum.utils.plotting.disp_wf(wf, sr=44100, autoplay=False, wf_plot_func=<function plot_wf>)

Display waveform in Jupyter notebook

* **Parameters:**
  * **wf** (*array-like*) – Waveform to display
  * **sr** ([*int*](https://docs.python.org/3/builtins/functions.html#int) *,* *optional*) – Sample rate of waveform, by default 44100
  * **autoplay** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool) *,* *optional*) – Whether to autoplay the audio, by default False
  * **wf_plot_func** (*function* *,* *optional*) – Function to plot the waveform, by default plot_wf (other example: plt.specgram)

### hum.utils.plotting.getmodulename(obj, default='')

Get name of module of object
