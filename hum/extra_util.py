"""Extra utils that require more dependencies"""

from typing import Iterable, Callable
from functools import partial


def estimate_chunk_frequency(chk: Iterable, sr: int):
    from scipy.signal import find_peaks
    from scipy.fft import fft, fftfreq
    import numpy as np

    # Perform FFT
    N = len(chk)
    yf = fft(chk)
    xf = fftfreq(N, 1 / sr)[: N // 2]

    # Find peaks in the FFT result
    peaks, _ = find_peaks(np.abs(yf[: N // 2]), height=0)

    if len(peaks) > 0:
        # Get the frequencies of the peaks
        peak_freqs = xf[peaks]

        # Return the frequency with the highest amplitude
        return peak_freqs[np.argmax(np.abs(yf[peaks]))]
    else:
        return None


def estimate_frequencies(wf: Iterable, sr: int, chunker: int | Callable):
    if isinstance(chunker, int):
        chk_size = chunker
        from hum.util import simple_chunker

        chunker = partial(simple_chunker, chk_size=chk_size, include_tail=True)
    elif not callable(chunker):
        raise ValueError("chunker must be an int or callable")

    for chk in chunker(wf):
        if len(chk) == 0:
            continue
        freq = estimate_chunk_frequency(chk, sr)
        yield freq
