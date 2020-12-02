from inspect import getmodule
import matplotlib.pylab as plt
from numpy import linspace


def getmodulename(obj, default=''):
    """Get name of module of object"""
    return getattr(getmodule(obj), '__name__', default)


def plot_wf(wf, sr=None, **kwargs):
    if sr is not None:
        plt.plot(linspace(start=0, stop=len(wf) / float(sr), num=len(wf)), wf, **kwargs)
    else:
        plt.plot(wf, **kwargs)


def disp_wf(wf, sr=44100, autoplay=False, wf_plot_func=plt.specgram):
    if wf_plot_func is not None:
        if getmodulename(wf_plot_func, '').startswith('matplotlib'):
            plt.figure(figsize=(16, 5))
        wf_plot_func(wf)
    try:
        from IPython.display import Audio
        return Audio(data=wf, rate=sr, autoplay=autoplay)
    except:
        pass
