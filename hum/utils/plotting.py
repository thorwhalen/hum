"""
Plot utils
"""
from inspect import getmodule
import matplotlib.pylab as plt
from numpy import linspace
from hum.utils.date_ticks import str_ticks

DFLT_FIGSIZE_FOR_WF_PLOTS = (22, 5)
DFLT_SR = 44100


def getmodulename(obj, default=''):
    """Get name of module of object"""
    return getattr(getmodule(obj), '__name__', default)


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


def disp_wf(wf, sr=DFLT_SR, autoplay=False, wf_plot_func=plt.specgram):
    if wf_plot_func is not None:
        if getmodulename(wf_plot_func, '').startswith('matplotlib'):
            plt.figure(figsize=DFLT_FIGSIZE_FOR_WF_PLOTS)
        wf_plot_func(wf)
    try:
        from IPython.display import Audio

        return Audio(data=wf, rate=sr, autoplay=autoplay)
    except:
        pass
