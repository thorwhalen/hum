"""
Generate simple waveforms
"""
import numpy as np
from collections import defaultdict

DFLT_SR = 44100
DFLT_CHK_SIZE_FRM = 21 * 2048
DFLT_MAX_AMPLITUDE = 30000
DFLT_PATTERN_LEN = 100
DFLT_PATTERN = [DFLT_MAX_AMPLITUDE] * 10 + [-DFLT_MAX_AMPLITUDE] * 10


def chk_from_pattern(chk_size_frm=DFLT_CHK_SIZE_FRM, pattern=None):
    if pattern is None:
        pattern = np.random.randint(
            -DFLT_MAX_AMPLITUDE, DFLT_MAX_AMPLITUDE, DFLT_PATTERN_LEN
        )
    return np.tile(pattern, reps=int(np.ceil(chk_size_frm / float(len(pattern)))))[
        :chk_size_frm
    ].astype(np.int16)


def random_samples(
    chk_size_frm=DFLT_CHK_SIZE_FRM, max_amplitude=DFLT_MAX_AMPLITUDE, **kwargs
):
    return np.random.randint(-max_amplitude, max_amplitude, chk_size_frm).astype(
        np.int16
    )


def pure_tone(
    chk_size_frm=DFLT_CHK_SIZE_FRM,
    freq=440,
    sr=DFLT_SR,
    max_amplitude=DFLT_MAX_AMPLITUDE,
):
    pattern_length = int(sr / freq)
    pattern = max_amplitude * np.sin(np.linspace(0, 2 * np.pi, pattern_length))
    return chk_from_pattern(chk_size_frm, pattern)


def triangular_tone(
    chk_size_frm=DFLT_CHK_SIZE_FRM,
    freq=440,
    sr=DFLT_SR,
    max_amplitude=DFLT_MAX_AMPLITUDE,
):
    pattern_length = int(sr / freq)
    pattern = np.arange(-max_amplitude, max_amplitude, pattern_length)
    return chk_from_pattern(chk_size_frm, pattern)


def square_tone(
    chk_size_frm=DFLT_CHK_SIZE_FRM,
    freq=440,
    sr=DFLT_SR,
    max_amplitude=DFLT_MAX_AMPLITUDE,
):
    pattern_length = int(sr / freq)
    half_pattern_length = int(pattern_length / 2)  # oh well for the plus minus 1
    pattern = [max_amplitude] * half_pattern_length + [
        -max_amplitude
    ] * half_pattern_length
    return chk_from_pattern(chk_size_frm, pattern)


tag_to_wf_gen_func = {
    'random': random_samples,
    'pure_tone': pure_tone,
    'triangular_tone': triangular_tone,
    'square_tone': square_tone,
}


class AnnotatedWaveform(object):
    def __init__(
        self,
        chk_size_frm=DFLT_CHK_SIZE_FRM,
        freq=440,
        sr=DFLT_SR,
        max_amplitude=DFLT_MAX_AMPLITUDE,
    ):
        self.chk_size_frm = chk_size_frm
        self.sr = sr
        self.freq = freq
        self.max_amplitude = max_amplitude
        self._default_kwargs = {
            'chk_size_frm': chk_size_frm,
            'freq': freq,
            'sr': sr,
            'max_amplitude': max_amplitude,
        }

    def chk_and_tag_gen(
        self, chk_tags=('random', 'pure_tone', 'triangular_tone', 'square_tone'),
    ):
        for tag in chk_tags:
            yield tag_to_wf_gen_func[tag](**self._default_kwargs), tag

    def get_wf_and_annots(
        self, chk_tags=('random', 'pure_tone', 'triangular_tone', 'square_tone'),
    ):
        slice_of_tag = defaultdict(list)
        bt_cursor = 0
        current_tag = None
        wf = list()
        for chk, tag in self.chk_and_tag_gen(chk_tags):
            wf += list(chk)
            if tag == current_tag:
                # TODO: Shame!
                slice_of_tag[tag][-1] = list(slice_of_tag[tag][-1])
                slice_of_tag[tag][-1][1] += self.chk_size_frm
                slice_of_tag[tag][-1] = tuple(slice_of_tag[tag][-1])
            else:
                current_tag = tag
                slice_of_tag[tag].append((bt_cursor, bt_cursor + self.chk_size_frm))

            bt_cursor += self.chk_size_frm

        return np.array(wf), dict(slice_of_tag)


import random
import itertools
from typing import Mapping, Callable, Sequence, Optional


def tag_wf_gen(
    tag_wfgen_map: Optional[Mapping[object, Callable[[], Sequence]]] = None,
    tag_sequence=None,
):
    """Generate (tag, wf) pairs.

    :param tag_wfgen_map: A {tag: wfgen, ...} map where wfgen is a callable taking no arguments and returning a sequence
    :param tag_sequence: A sequence of tags (that should all be keys of tag_wfgen_map)
    :return:
    """
    if tag_wfgen_map is None:
        tag_wfgen_map = tag_to_wf_gen_func
    if tag_sequence is None:
        indefinite_random_choice_from_tags = map(
            random.choice, itertools.repeat(list(tag_wfgen_map))
        )
        tag_sequence = indefinite_random_choice_from_tags

    for tag in tag_sequence:
        yield tag, tag_wfgen_map[tag]()
