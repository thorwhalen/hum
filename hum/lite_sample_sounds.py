"""
Generate simple waveforms
"""
import numpy as np
import random
import itertools
from typing import Mapping, Callable, Sequence, Any, Optional, Iterable, TypeVar
from collections import defaultdict

DFLT_SR = 44100
DFLT_CHK_SIZE = 21 * 2048
DFLT_MAX_AMPLITUDE = 30000
DFLT_PATTERN_LEN = 100
DFLT_PATTERN = [DFLT_MAX_AMPLITUDE] * 10 + [-DFLT_MAX_AMPLITUDE] * 10


def chk_from_pattern(chk_size=DFLT_CHK_SIZE, pattern=None):
    """
    Returns a chk with length chk_size that repeats pattern if given, or creates a random pattern of length 100
    >>> np.random.seed(1)
    >>> assert all(chk_from_pattern(5) == [3003, -17828, -24808, 2511, 20057])
    >>> chk = chk_from_pattern(6, [1,2,3])
    >>> assert all(chk == [1, 2, 3, 1, 2, 3])
    """
    if pattern is None:
        pattern = random_samples(DFLT_PATTERN_LEN, DFLT_MAX_AMPLITUDE)
    return np.tile(pattern, reps=int(np.ceil(chk_size / float(len(pattern)))))[
        :chk_size
    ].astype(np.int16)


def random_samples(chk_size=DFLT_CHK_SIZE, max_amplitude=DFLT_MAX_AMPLITUDE, **kwargs):
    """
    Returns a random sample of integers of length chk_size in the range [-max_amplitude, max_amplitude]
    >>> np.random.seed(1)
    >>> assert all(random_samples(5) == [3003, -17828, -24808, 2511, 20057])
    """
    return np.random.randint(-max_amplitude, max_amplitude, chk_size).astype(np.int16)


def pure_tone(
    chk_size=DFLT_CHK_SIZE, freq=440, sr=DFLT_SR, max_amplitude=DFLT_MAX_AMPLITUDE,
):
    """
    Generates a pure tone using given arguments
    >>> np.random.seed(1)
    >>> assert all(pure_tone(5) == [0, 1902, 3797, 5677, 7534])
    """
    pattern_length = int(sr / freq)
    pattern = max_amplitude * np.sin(np.linspace(0, 2 * np.pi, pattern_length))
    return chk_from_pattern(chk_size, pattern)


def triangular_tone(
    chk_size=DFLT_CHK_SIZE, freq=440, sr=DFLT_SR, max_amplitude=DFLT_MAX_AMPLITUDE,
):
    """
    Generates a triangular tone using given arguments
    >>> np.random.seed(1)
    >>> assert all(triangular_tone(5) == [-30000, -29900, -29800, -29700, -29600])
    """
    pattern_length = int(sr / freq)
    pattern = np.arange(-max_amplitude, max_amplitude, pattern_length)
    return chk_from_pattern(chk_size, pattern)


def square_tone(
    chk_size=DFLT_CHK_SIZE, freq=440, sr=DFLT_SR, max_amplitude=DFLT_MAX_AMPLITUDE,
):
    """
    Generates a square tone using given arguments
    >>> np.random.seed(1)
    >>> assert all(square_tone(5) == [30000, 30000, 30000, 30000, 30000])
    """
    pattern_length = int(sr / freq)
    half_pattern_length = int(pattern_length / 2)  # oh well for the plus minus 1
    pattern = [max_amplitude] * half_pattern_length + [
        -max_amplitude
    ] * half_pattern_length
    return chk_from_pattern(chk_size, pattern)


tag_to_wf_gen_func = {
    'random': random_samples,
    'pure_tone': pure_tone,
    'triangular_tone': triangular_tone,
    'square_tone': square_tone,
}

tag_to_wf_gen_func_items = tuple(tag_to_wf_gen_func.items())


class AnnotatedWaveform(object):
    """
    Creates a waveform with chunks corresponding to the tags in tag_to_wf_gen_func.
    >>> np.random.seed(1)
    >>> annotated_wf = AnnotatedWaveform(chk_size = 4)
    >>> chk_tag_gen = annotated_wf.chk_and_tag_gen()
    >>> wf, annots = annotated_wf.get_wf_and_annots()
    >>> list(wf)
    [3003, -17828, -24808, 2511, 0, 1902, 3797, 5677, -30000, -29900, -29800, -29700, 30000, 30000, 30000, 30000]
    >>> assert list(next(chk_tag_gen)[0]) == [20057, 13723, -22187, 22047]
    """

    def __init__(
        self,
        chk_size=DFLT_CHK_SIZE,
        freq=440,
        sr=DFLT_SR,
        max_amplitude=DFLT_MAX_AMPLITUDE,
    ):
        self.chk_size = chk_size
        self.sr = sr
        self.freq = freq
        self.max_amplitude = max_amplitude
        self._default_kwargs = {
            'chk_size': chk_size,
            'freq': freq,
            'sr': sr,
            'max_amplitude': max_amplitude,
        }

    def chk_and_tag_gen(
        self, chk_tags=('random', 'pure_tone', 'triangular_tone', 'square_tone'),
    ):
        """
        Yields (chk, tag) pairs for each tag given in chk_tags
        """
        for tag in chk_tags:
            yield tag_to_wf_gen_func[tag](**self._default_kwargs), tag

    def get_wf_and_annots(
        self, chk_tags=('random', 'pure_tone', 'triangular_tone', 'square_tone'),
    ):
        """
        Yields (wf, annots) tuple where annots is a dictionary mapping tag to chunk indices
        """
        slice_of_tag = defaultdict(list)
        bt_cursor = 0
        current_tag = None
        wf = list()
        for chk, tag in self.chk_and_tag_gen(chk_tags):
            wf += list(chk)
            if tag == current_tag:
                # TODO: Shame!
                slice_of_tag[tag][-1] = list(slice_of_tag[tag][-1])
                slice_of_tag[tag][-1][1] += self.chk_size
                slice_of_tag[tag][-1] = tuple(slice_of_tag[tag][-1])
            else:
                current_tag = tag
                slice_of_tag[tag].append((bt_cursor, bt_cursor + self.chk_size))

            bt_cursor += self.chk_size

        return np.array(wf), dict(slice_of_tag)


Tag = TypeVar('Tag')  # TODO: Use generic for better typing


def tag_wf_gen(
    tag_wfgen_map: Mapping[Tag, Callable[..., Sequence]] = tag_to_wf_gen_func_items,
    tag_sequence: Optional[Iterable[Tag]] = None,
):
    """Generate tagged waveforms -- i.e. ``(tag, wf)`` pairs.

    :param tag_wfgen_map: A ``{tag: wfgen, ...}`` map where wfgen is a callable taking
        no arguments and returning a sequence
    :param tag_sequence: A sequence of tags (that should all be keys of tag_wfgen_map)
    :return:

    Make a generator (here with default specs):

    >>> gen = tag_wf_gen()

    From it, source ``(tag, wf)`` pairs (by default, indefinitely).

    >>> tag, wf = next(gen)

    Since by default, we're given random ``tag`` and ``wf``, the only thing we can
    verify here is that tag is a string and ``wf`` has the default size of 43008.

    >>> type(tag)
    <class 'str'>
    >>> len(wf)
    43008

    By default the tags are randomly chosen from the ``tag_wfgen_map``, but you can
    also specify a iterable of tags explicitly. Obviously, the elements of this
    sequence must actually be keys of ``tag_wfgen_map`` -- this will be asserted if
    the iterable is a Sequence.

    You can specify a sequence of tags:

    >>> gen = tag_wf_gen(
    ...     tag_sequence=['random', 'pure_tone', 'triangular_tone', 'square_tone']
    ... )
    >>> tag_and_wf_seq = list(gen)  # consume the whole gen
    >>> [x[0] for x in tag_and_wf_seq]
    ['random', 'pure_tone', 'triangular_tone', 'square_tone']
    >>> list(tag_and_wf_seq[2][1][:5])  # the first 5 numbers of the triangular tone
    [-30000, -29900, -29800, -29700, -29600]

    """
    tag_wfgen_map = dict(tag_wfgen_map)

    if tag_sequence is None:
        indefinite_random_choice_from_tags = map(
            random.choice, itertools.repeat(list(tag_wfgen_map))
        )
        tag_sequence = indefinite_random_choice_from_tags
    else:
        if isinstance(tag_sequence, Sequence):
            tag_sequence = list(tag_sequence)
            if not all(tag in tag_wfgen_map for tag in tag_sequence):
                raise ValueError(
                    f'A tag_sequence must only have elements that are keys of '
                    f'tag_wfgen_map, i.e. from {list(tag_wfgen_map)}'
                )

    for tag in tag_sequence:
        yield tag, tag_wfgen_map[tag]()
