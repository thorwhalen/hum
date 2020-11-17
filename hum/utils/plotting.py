import matplotlib.pylab as plt
from numpy import linspace


def plot_wf(wf, sr=None, **kwargs):
    if sr is not None:
        plt.plot(linspace(start=0, stop=len(wf) / float(sr), num=len(wf)), wf, **kwargs)
    else:
        plt.plot(wf, **kwargs)


def disp_wf(wf, sr=44100, autoplay=False):
    plt.figure(figsize=(16, 5))
    plt.plot(wf)
    try:
        from IPython.display import Audio
        return Audio(data=wf, rate=sr, autoplay=autoplay)
    except:
        pass
