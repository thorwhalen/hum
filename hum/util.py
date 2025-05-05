"""
A few general utils
"""

from warnings import warn
from inspect import getmodule
from typing import Iterable, Iterator, Tuple, TypeVar, Callable, Iterable, Union
from itertools import zip_longest
from functools import partial

import numpy as np


T = TypeVar("T")


def simple_chunker(
    a: Iterable[T], chk_size: int, *, include_tail: bool = True
) -> Iterator[Tuple[T, ...]]:
    """
    Chunks an iterable into non-overlapping chunks of size `chk_size`.

    Note: This chunker is simpler, but also less efficient than `chunk_iterable`.
    It does have the extra `include_tail` argument, though.
    Though note that you can get the effect of `include_tail=False` in `chunk_iterable`
    by using `filter(lambda x: len(x) == chk_size, chunk_iterable(...))`.

    Args:
        a: The iterable to be chunked.
        chk_size: The size of each chunk.
        include_tail: If True, includes the remaining elements as the last chunk
                      even if they are fewer than `chk_size`. Defaults to True.

    Returns:
        An iterator of tuples, where each tuple is a chunk of size `chk_size`
        (or fewer elements if `include_tail` is True).

    Examples:
        >>> list(simple_chunker(range(8), 3))
        [(0, 1, 2), (3, 4, 5), (6, 7)]
        >>> list(simple_chunker(range(8), 3, include_tail=False))
        [(0, 1, 2), (3, 4, 5)]
    """
    it = iter(a)
    if include_tail:
        sentinel = object()
        for chunk in zip_longest(*([it] * chk_size), fillvalue=sentinel):
            yield tuple(item for item in chunk if item is not sentinel)
    else:
        yield from zip(*([it] * chk_size))


def round_number(number, round_to=0.001):
    """
    Round a number to the nearest multiple of round_to.

    Parameters:
    - number: float, the number to round.
    - round_to: float, the value to round to.

    Returns:
    - float, the rounded number.

    >>> round_number(1.234567, 0.01)
    1.23

    Note, the rounder rounds to nearest multiple of round_to, not to the nearest decimal place.

    >>> round_number(1.234567, 0.001)
    1.235


    """
    digits = len(str(round_to).split(".")[-1])
    return round(round(number / round_to) * round_to, digits)


def round_numbers(items, round_to=0.001, *, index_of_item_number=None, egress=None):
    """
    Round numbers in an iterable, optionally extracting the number to round from an index.

    Parameters:
    - items: iterable of numbers or iterable of tuples/lists containing numbers.
    - round_to: float, round to the nearest multiple of this value.
    - index_of_item_number: int or None. If None, round the item directly;
      otherwise, round the item at this index in each iterable element.

    Returns:
    - generator yielding items with rounded numbers.

    Examples:
    >>> list(round_numbers([1.23, 3.14159], round_to=0.1))
    [1.2, 3.1]

    >>> items = [(1.234, 'one'), (3.14159, 'three')]
    >>> list(round_numbers(items, round_to=0.1, index_of_item_number=0))
    [[1.2, 'one'], [3.1, 'three']]
    """
    digits = len(str(round_to).split(".")[-1])

    for item in items:
        if index_of_item_number is None:
            yield round(round(item / round_to) * round_to, digits)
        else:
            val = item[index_of_item_number]
            rounded_val = round(round(val / round_to) * round_to, digits)
            item = list(item)
            item[index_of_item_number] = rounded_val

            if egress is None:
                yield item
            else:
                yield egress(item)


# ----------------------------------------------------------------------------------
# Frequency snappers
# ----------------------------------------------------------------------------------

from bisect import bisect_left
from math import log2


class ListSnapper:
    """
    Snap to the nearest value in a list.

    >>> snapper = ListSnapper([0, 1, 2, 3, 4, 5])
    >>> snapper(2.3)
    2
    >>> snapper(2.7)
    3
    >>> snapper(2.5)  # exactly the middle of 2 and 3; snaps to 2
    2

    """

    def __init__(self, snap_values):
        self.snap_values = sorted(snap_values)

    def __call__(self, x):
        pos = bisect_left(self.snap_values, x)
        if pos == 0:
            return self.snap_values[0]
        if pos == len(self.snap_values):
            return self.snap_values[-1]
        before = self.snap_values[pos - 1]
        after = self.snap_values[pos]
        return before if abs(x - before) <= abs(after - x) else after


class SnapToChromatic:
    """
    Snap to the nearest chromatic note in a tempered scale.
    """

    def __init__(self, base_tuning=440):
        self.base_tuning = base_tuning

    def __call__(self, x):
        if x <= 0:
            raise ValueError("Frequency must be positive")
        midi_note = round(69 + 12 * log2(x / 440))
        return 440 * 2 ** ((midi_note - 69) / 12)


DFLT_OCTAVE_RANGE = (0, 10)
DFLT_TUNING = 440.0  # A4 tuning frequency
DFLT_SCALE_MIDI_NOTES = (0, 2, 4, 5, 7, 9, 11)  # C major scale


def tempered_semitone_frequencies(
    tuning=DFLT_TUNING, octave_range=DFLT_OCTAVE_RANGE, digits=5
):
    """
    Get a sorted list of frequencies for a given tuning and octave range.

    :param tuning: The tuning frequency (default is 440 Hz).
        It corresponds to A4 (that is, the 69th MIDI note).
    :param octave_range: A tuple of two integers representing the range of octaves.

        The first integer is the starting octave (inclusive), and the second integer is the ending octave (exclusive).
        For example, (0, 10) will generate frequencies from C0 to B9.
        The default is (0, 10), which covers a wide range of frequencies.
    :return: A sorted list of frequencies in the specified octave range. The first (i.e. .[0]) frequency will correspond to a C.

    >>> frequencies = tempered_semitone_frequencies()
    >>> len(frequencies)
    120
    >>> frequencies  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [8.1758, 8.66196, 9.17702, 9.72272, ..., 7040.0, 7458.62018, 7902.13282]

    """

    assert len(octave_range) == 2, "octave_range must be a tuple of two integers"
    n_octave_divisions = 12

    def gen_freqs():
        for octave in range(*octave_range):
            base_midi = n_octave_divisions * octave
            for st in range(n_octave_divisions):
                midi_note = base_midi + st
                freq = tuning * 2 ** ((midi_note - 69) / n_octave_divisions)
                yield round(freq, digits)

    return sorted(gen_freqs())


def scale_frequencies(
    scale: Iterable[Union[float, int]] = DFLT_SCALE_MIDI_NOTES,
    *,
    tuning=DFLT_TUNING,
    octave_range=DFLT_OCTAVE_RANGE,
):
    """
    Get frequencies for a given scale.

    :param scale: A list of integers representing the scale.
        For example, (0, 2, 4, 5, 7, 9, 11) corresponds to the C major scale.
    :param tuning: The tuning frequency (default is 440 Hz).
    :param octave_range: A tuple of two integers representing the range of octaves.
    :return: A list of frequencies corresponding to the notes in the given scale.
    """
    chromatic_scale = tempered_semitone_frequencies(
        tuning=tuning, octave_range=octave_range
    )
    return [chromatic_scale[i] for i in range(len(chromatic_scale)) if i % 12 in scale]


# TODO: Add string_to_scale_map argument to convert string to midi note array scale
def scale_snapper(
    scale: Iterable[Union[float, int]] = DFLT_SCALE_MIDI_NOTES,
    *,
    tuning=DFLT_TUNING,
    octave_range=DFLT_OCTAVE_RANGE,
):
    """
    Snap frequency to the nearest midi note of the midi_nodes list, modulo 12.

    :param scale: A list of integers representing the scale.
        For example, (0, 2, 4, 5, 7, 9, 11) (the default) corresponds to the C major scale.
    :param tuning: The tuning frequency (default is 440 Hz).
        It corresponds to A4 (that is, the 69th MIDI note).
    :param octave_range: A tuple of two integers representing the range of octaves.
        The first integer is the starting octave (inclusive), and the second integer is the ending octave (exclusive).
        For example, (0, 10) will generate frequencies from C0 to B9.
        The default is (0, 10), which covers a wide range of frequencies.

    :return: A ListSnapper object that can be used to snap frequencies to the nearest note in the scale.

    >>> snapper = scale_snapper()
    >>> snapper(440.0)  # A4
    440.0

    If you're near that A4, `snapper` will snap to it:

    >>> snapper(441.0)
    440.0
    >>> snapper(439.0)
    440.0

    But even if you're, say at A#, you'll still snap to A4 (since it's the closest note
    of C major scale):

    >>> snapper(466.16)  # A#4
    440.0

    If you are slightly higher than A#, you'll snap to B4

    >>> snapper(467)  # doctest: +ELLIPSIS
    493.8833...

    """
    _scale_frequencies = scale_frequencies(
        scale=scale, tuning=tuning, octave_range=octave_range
    )
    # create a ListSnapper object
    return ListSnapper(_scale_frequencies)


# ----------------------------------------------------------------------------------


def getmodulename(obj, default=""):
    """Get name of module of object"""
    return getattr(getmodule(obj), "__name__", default)


class ModuleNotFoundErrorNiceMessage:
    def __init__(self, msg=None):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            msg = (
                self.msg
                or f"""
It seems you don't have required `{exc_val.name}` package for this Store.
Try installing it by running:

    pip install {exc_val.name}

in your terminal.
For more information: https://pypi.org/project/{exc_val.name}
            """
            )
            warn(msg)


class ModuleNotFoundIgnore:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            pass
        return True
