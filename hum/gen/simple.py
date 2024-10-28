"""Waveforms generated only with builtins (no numpy etc.)
Meant to be copy/pasted where needed in tests.
"""


def inverted_tiles_wf(motif=(1, 2, 3, 4, 5), cap=20000, wf_length=21 * 2048):
    """Get a (possibly infinite) waveform through inverted_tiles chunks

    These waveforms are originally meant to be used with [1,...,n] motifs and used for
    diagnosis. Indeed, the samples alternate sign (so waveform can be made audible),
    but checking for the absolute value of the samples, you should see incrementing
    integers.

    :param motif: The iterable to use as the (incremented and inverted) pattern to
        make waveforms out of.
    :param cap: The maximum absolute value to include in a waveform.
        Also determines when we resset the chk to be the motif.
    :param wf_length: Length of the desired waveform, or None for an infinite iterable

    >>> wf = inverted_tiles_wf()
    >>> len(wf)
    43008
    >>> wf[:15]
    [1, 2, 3, 4, 5, -6, -7, -8, -9, -10, 11, 12, 13, 14, 15]
    >>> inverted_tiles_wf(motif=(1, 2, 3), cap=10, wf_length=15)
    [1, 2, 3, -4, -5, -6, 7, 8, 9, -10, -11, -12, 1, 2, 3]

    """
    from itertools import chain, islice

    chks = inverted_tiles(motif=motif, cap=cap)
    it = chain.from_iterable(chks)
    if wf_length is not None:
        return list(islice(it, wf_length))
    else:
        return it


def inverted_tiles(motif=(1, 2, 3, 4, 5), cap=20000):
    """Infinite generator of chunks taken by transforming the input motif.
    The iterative transformation is constituted of a translation by the length of the
    motif and an inversion of the sign.

    :param motif: The iterable to use as the (incremented and inverted) pattern to
        make waveforms out of.
    :param cap: The maximum absolute value to include in a waveform.
        Also determines when we resset the chk to be the motif.

    >>> g = inverted_tiles()
    >>> next(g), next(g), next(g)
    ([1, 2, 3, 4, 5], [-6, -7, -8, -9, -10], [11, 12, 13, 14, 15])
    >>> g = inverted_tiles(motif=(1, 2, 3), cap=10)
    >>> [*next(g), *next(g), *next(g), *next(g)]
    [1, 2, 3, -4, -5, -6, 7, 8, 9, -10, -11, -12]

    """
    # input preprocessing
    if isinstance(motif, int):
        motif = range(1, motif + 1)
    motif = list(motif)
    current_chk = motif
    motif_length = len(current_chk)
    sign = 1
    while True:
        yield current_chk
        if max(abs(x) for x in current_chk) >= cap:  # if we're above the cap
            current_chk = motif  # reset the current_motif
        else:
            sign *= -1  # invert sign
            # and make the next current_chk be
            # the sign-inverted motif_length-incremented version of the current one
            current_chk = [sign * (abs(x) + motif_length) for x in current_chk]
