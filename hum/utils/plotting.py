"""
Plot utils
"""

from inspect import getmodule
import matplotlib.pylab as plt
from numpy import linspace

from hum.utils.date_ticks import str_ticks
from recode import decode_wav_bytes

DFLT_FIGSIZE_FOR_WF_PLOTS = (22, 5)
DFLT_SR = 44100


def getmodulename(obj, default=""):
    """Get name of module of object"""
    return getattr(getmodule(obj), "__name__", default)


# def plot_wf(wf, sr=None, figsize=(20, 6), **kwargs):
#     if figsize is not None:
#         plt.figure(figsize=figsize)
#     if sr is not None:
#         plt.plot(linspace(start=0, stop=len(wf) / float(sr), num=len(wf)), wf, **kwargs)
#     else:
#         plt.plot(wf, **kwargs)


def plot_wf(
    wf, sr=None, figsize=DFLT_FIGSIZE_FOR_WF_PLOTS, offset_s=0, ax=None, **kwargs
):
    if figsize is not None:
        plt.figure(figsize=figsize)
    _ax = ax or plt
    if sr is not None:
        _ax.plot(
            offset_s + linspace(start=0, stop=len(wf) / float(sr), num=len(wf)),
            wf,
            **kwargs
        )
        plt.margins(x=0)
    else:
        _ax.plot(wf, **kwargs)
        plt.margins(x=0)
        return
    if _ax == plt:
        _xticks, _ = plt.xticks()
        plt.xticks(_xticks, str_ticks(ticks=_xticks, ticks_unit=1))
        plt.margins(x=0)
    else:
        _xticks = _ax.get_xticks()
        _ax.set_xticks(_xticks)
        _ax.set_xticklabels(str_ticks(ticks=_xticks, ticks_unit=1))
        plt.margins(x=0)


def disp_wf(wf, sr=DFLT_SR, autoplay=False, wf_plot_func=plot_wf):
    """
    Display waveform in Jupyter notebook

    Parameters
    ----------
    wf : array-like
        Waveform to display
    sr : int, optional
        Sample rate of waveform, by default 44100
    autoplay : bool, optional
        Whether to autoplay the audio, by default False
    wf_plot_func : function, optional
        Function to plot the waveform, by default plot_wf (other example: plt.specgram)

    """
    if isinstance(wf, bytes):
        # if it's a RIFF file, decode it as WAV
        if wf[:4] == b"RIFF":
            # Decode the WAV bytes
            wav_bytes = wf
            wf, sr = decode_wav_bytes(wav_bytes)
        else:
            raise ValueError("Unsupported audio format. Only WAV bytes are supported.")

    if wf_plot_func is not None:
        if getmodulename(wf_plot_func, "").startswith("matplotlib"):
            plt.figure(figsize=DFLT_FIGSIZE_FOR_WF_PLOTS)
        wf_plot_func(wf, sr)
    try:
        from IPython.display import Audio

        return Audio(data=wf, rate=sr, autoplay=autoplay)
    except:
        pass
