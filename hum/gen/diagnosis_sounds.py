# from numpy import *
from numpy import (array, hstack, ones, ceil, zeros, floor, argmin, diff, where, reshape, math,
                   arange, iinfo, int16, linspace, pi, unique, tile, repeat)
from numpy.random import randint, random
from itertools import islice, count

from datetime import datetime as dt

second_ms = 1000.0
epoch = dt.utcfromtimestamp(0)


def utcnow_ms():
    return (dt.utcnow() - epoch).total_seconds() * second_ms


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def ums_to_01_array(ums, n_ums_bits):
    ums_bits_str_format = "{:0" + str(n_ums_bits) + "b}"
    return array([int(x == '1') for x in ums_bits_str_format.format(ums)])


class BinarySound(object):
    def __init__(self, redundancy, repetition, nbits=50, header_size_words=1, header_pattern=None):
        """
        :param redundancy: num of times a word will be repeated (to implement error correction)
        :param repetition: num of times to repeat each element of a pattern.
        :param nbits: num of bits of a pattern. An {0,1}-array to be encoded will have to be this size.
            Also, if a header_pattern is explicitly given, it will also have to be this size.
        :param header_size_words: the size (in num of words) of the header
        :param header_pattern: specifies the header pattern. This pattern will be repeated (i.e. it's elements
            repeated several times (010-->000011110000) and tiled (header_size_words times). Can be:
            * None: Default. Will choose a random array of 0s and 1s
            * an explicit array of nbits 0s and 1s to use for the header
            * a string indicating a method to generate it (for now, choices are "halfhalf" or "alternating")
        >>> from oto.sound.diagnosis_sounds import BinarySound
        >>> from numpy import *
        >>> from numpy.random import randint
        >>>
        >>> nbits=50
        >>> bs = BinarySound(
        ...     nbits=nbits, redundancy=142, repetition=3, header_size_words=1)
        >>> utc = randint(0, 2, nbits)
        >>> wf = bs.mk_phrase(utc)
        >>> print(bs)
        {'repetition': 3, 'redundancy': 142, 'word_size_frm': 150, 'phrase_data_frm': 21300}
        >>> all(utc == bs.decode(wf))
        True
        """
        self.nbits = nbits

        self.header_size_words = header_size_words
        # repetition: how many times to repeat each bit to make a word
        self.repetition = repetition
        # word_size_frm: the size of a word, in frames
        self.word_size_frm = int(self.nbits * self.repetition)
        # redundancy: how many times to repeat a word to make (along with header) a phrase
        self.redundancy = redundancy
        if header_pattern is None:
            header_pattern = randint(0, 2, nbits)
        elif isinstance(header_pattern, str):
            if header_pattern == 'halfhalf':
                header_pattern = hstack((ones(int(ceil(nbits / 2))), zeros(int(floor(nbits / 2))))).astype(int)
            elif header_pattern == 'alternating':
                header_pattern = array([1, 0] * int(ceil(nbits / 2)))[:nbits]
            else:
                raise ValueError("header_pattern not recognized: {}".format(header_pattern))
        else:
            assert len(header_pattern) == nbits, "header_pattern must have nbits={}".format(nbits)
            assert set(unique(header_pattern).astype(int)) == {0, 1}, "header_pattern must be an array of 0s and 1s"
        self.header_pattern = header_pattern
        self.header_word = tile(repeat(self.header_pattern, self.repetition), self.header_size_words)
        self.phrase_data_frm = self.redundancy * self.word_size_frm

    @classmethod
    def for_audio_params(cls, nbits=50, freq=3000, chk_size_frm=43008, sr=44100, header_size_words=1,
                         header_pattern=None):
        """
        Construct a BinarySound object for a set of audio params
        :param nbits: num of bits of a word of data we want to encode
        :param freq: frequency (num of times per second the bits will be repeated -- with sr, will determine repetition)
        :param chk_size_frm: chunk size (in frames) of the sounds we'll be using (determines
        :param sr: sample rate of the targeted sound
        :param header_size_words: header size (increasing it will decrease error rate, but increase computation time)
        :param header_pattern: specifies the header pattern. This pattern will be repeated (i.e. it's elements
            repeated several times (010-->000011110000) and tiled (header_size_words times). Can be:
            * None: Default. Will choose a random array of 0s and 1s
            * an explicit array of nbits 0s and 1s to use for the header
            * a string indicating a method to generate it (for now, choices are "halfhalf" or "alternating")
        :return: a BinarySound object
        >>> from oto.sound.diagnosis_sounds import BinarySound
        >>> from numpy import *
        >>> from numpy.random import randint
        >>>
        >>> nbits=50
        >>> bs = BinarySound.for_audio_params(
        ...     nbits=nbits, freq=6000, chk_size_frm=43008, sr=44100, header_size_words=1)
        >>> utc = randint(0, 2, nbits)
        >>> wf = bs.mk_phrase(utc)
        >>> print(bs)
        {'repetition': 3, 'redundancy': 142, 'word_size_frm': 150, 'phrase_data_frm': 21300}
        >>> all(utc == bs.decode(wf))
        True
        """
        # repetition: how many times to repeat each bit to make a word
        repetition = int(floor(sr / (2 * freq)))
        # word_size_frm: the size of a word, in frames
        word_size_frm = int(nbits * repetition)
        # redundancy: how many times to repeat a word to make (along with header) a phrase
        redundancy = (int(floor((chk_size_frm / 2) / word_size_frm) - header_size_words))

        self = cls(nbits=nbits, redundancy=redundancy, repetition=repetition, header_size_words=header_size_words,
                   header_pattern=header_pattern)
        self.freq = freq
        self.sr = sr
        self.chk_size_frm = chk_size_frm
        self.redundancy = redundancy
        self.repetition = repetition
        return self

    def mk_phrase(self, bit_array):
        wf = hstack((self.header_word,
                     tile(repeat(bit_array, self.repetition), self.redundancy)))
        return (2 * wf) - 1

    def mk_utc_phrases(self, sound_duration_s=12):
        wf = list()

        def wf_seconds(wf):
            return len(wf) / float(self.sr)

        while wf_seconds(wf) < sound_duration_s:
            bit_array = ums_to_01_array(ums=int(utcnow_ms()), n_ums_bits=self.nbits)
            wf += list(self.mk_phrase(bit_array))

        return array(wf)[:int(sound_duration_s * self.sr)]

    def header_position(self, wf):
        wf = wf > 0
        return argmin(slow_mask(wf, self.header_word))

    def decode(self, wf):
        header_pos = self.header_position(wf)
        header_end_idx = header_pos + len(self.header_word)
        wf = wf > 0
        m = wf[header_end_idx:(header_end_idx + self.phrase_data_frm)]
        size_to_make_it_a_multiple_of_word_size = (len(m) - (len(m) % self.word_size_frm))
        m = m[:size_to_make_it_a_multiple_of_word_size]
        m = reshape(m, (-1, self.word_size_frm))
        m = m.sum(axis=0).reshape((-1, self.repetition))
        m = m.sum(axis=1)
        return (m / float(self.repetition * self.redundancy) > 0.5).astype(int)

    def __repr__(self):
        return str({
            'repetition': self.repetition,
            'word_size_frm': self.word_size_frm,
            'redundancy': self.redundancy,
            'phrase_data_frm': self.phrase_data_frm
        })


def zero_crossing_gaps(wf):
    w = wf > 0
    ww = hstack((w, ~w[-1]))
    return diff(hstack(([0], 1 + where(diff(ww))[0])))


def slow_mask(arr, msk):
    msk = array(msk)
    arr_msk_dist = list()
    for w in window(arr, len(msk)):
        arr_msk_dist.append(sum(abs(array(w) - msk)))
    return arr_msk_dist


class WfGen(object):
    def __init__(self, sr=44100, buf_size_frm=2048, amplitude=0.5):
        self.sr = sr
        self.buf_size_frm = buf_size_frm
        self.buf_size_s = buf_size_frm / float(self.sr)
        if amplitude > 1.0: amplitude = 1.0
        if amplitude < 0.0: amplitude = 0.0
        self.amplitude = float(amplitude)
        self.lookup_table_freqs = (arange(int(buf_size_frm / 2)) + 1) / self.buf_size_s
        self.lookup_tables = list(map(self.mk_lookup_table, self.lookup_table_freqs))

    def mk_sine_wave_from_lookup_table(self, lookup_table):
        period = len(lookup_table)
        return (lookup_table[i % period] for i in count(0))

    def mk_sine_wave_iterator(self, freq=440):
        if isinstance(freq, (int, float)):
            table_idx = where(self.lookup_table_freqs == freq)[0]
            if len(table_idx) > 0:
                table_idx = table_idx[0]
                lookup_table = self.lookup_tables[table_idx]
            else:
                lookup_table = self.mk_lookup_table(freq=freq)
            return self.mk_sine_wave_from_lookup_table(lookup_table)
        else:  # consider freq to already be a lookup table
            return self.mk_sine_wave_from_lookup_table(freq)

    def mk_sine_wf(self, n_frm, freq=440):
        it = self.mk_sine_wave_iterator(freq)
        return array([x for x in islice(it, int(n_frm))])

    def mk_lookup_table(self, freq=440):
        freq = float(freq)
        period = int(self.sr / freq)
        lookup_table = [self.amplitude * math.sin(2.0 * math.pi * float(freq) * (float(i % period) / float(self.sr)))
                        for i in range(period)]
        return lookup_table

    def mk_wf_from_freq_weight_array(self, n_frm, freq_weight_array):
        wf = zeros(n_frm)
        for i, w in enumerate(freq_weight_array):
            wf += w * self.mk_sine_wf(n_frm, self.lookup_tables[i])
        return wf


class TimeSound(WfGen):
    def __init__(self, sr=44100, buf_size_frm=2048, amplitude=0.5, n_ums_bits=30):
        super(TimeSound, self).__init__(sr=sr, buf_size_frm=buf_size_frm, amplitude=amplitude)
        self.n_ums_bits = n_ums_bits
        self.ums_bits_str_format = "{:0" + str(n_ums_bits) + "b}"
        self.n_freqs_per_ums_bit = len(self.lookup_tables) // self.n_ums_bits
        self.n_freqs_for_ums = self.n_freqs_per_ums_bit * self.n_ums_bits
        self.buf_size_ms = self.buf_size_s * 1000

    def ums_to_01_array(self, ums):
        return array([int(x == '1') for x in self.ums_bits_str_format.format(ums)])

    def freq_weight_array_for_ums(self, ums):
        return tile(self.ums_to_01_array(ums), self.n_freqs_per_ums_bit)

    def ums_to_wf(self, ums, n_bufs=1):
        return self.mk_wf_from_freq_weight_array(n_frm=n_bufs * self.buf_size_frm,
                                                 freq_weight_array=self.freq_weight_array_for_ums(ums))

    def timestamped_wf(self, offset_ums=0, n_bufs=21, n_bufs_per_tick=1):
        wf = list()
        ums = offset_ums
        for buf_idx in range(n_bufs):
            wf.extend(list(self.ums_to_wf(ums, n_bufs=n_bufs_per_tick)))
            ums = int(ums + n_bufs_per_tick * self.buf_size_ms)
        return array(wf)

    def spectr_of_time(self, offset_ums=0, n_bufs=21, n_bufs_per_tick=1):
        wf = list()
        ums = offset_ums
        for buf_idx in range(n_bufs):
            wf.append(list(self.freq_weight_array_for_ums(ums)))
            if n_bufs_per_tick > 1:
                wf.append(list(zeros(self.n_freqs_for_ums)))
            ums = int(ums + n_bufs_per_tick * self.buf_size_ms)
        wf = array(wf)
        return hstack((wf, zeros((wf.shape[0], int(self.buf_size_frm / 2 - wf.shape[1])))))


import soundfile as sf
from scipy import signal


def mk_some_buzz_wf(sr=44100):
    bleep_wf = (signal.sawtooth(pi * (sr / 10) * linspace(0, 1, int(5 * sr))))
    bleep_wf += randint(-1, 1, len(bleep_wf))
    return ((bleep_wf / 2) * iinfo(int16).max).astype(int16)


def wf_with_timed_bleeps(bleep_loc_ms,
                         bleep_spec=200,
                         sr=6144):
    """Not sure this works as expected. Docs needed."""
    if isinstance(bleep_spec, int):
        bleep_size_ms = bleep_spec
        bleep_size_frm = int(sr * bleep_size_ms / 1000)
        bleep_spec = mk_some_buzz_wf(sr)[:bleep_size_frm]

    bleep_size_frm = len(bleep_spec)
    bleep_loc_frm = (sr * (array(bleep_loc_ms) / 1000)).astype(int)
    max_bleep_loc_frm = max(bleep_loc_frm) + len(bleep_spec)
    wf = zeros(max_bleep_loc_frm)
    for loc_frm in bleep_loc_frm:
        wf[loc_frm:(loc_frm + bleep_size_frm)] = bleep_spec
    return wf


def mk_sounds_with_timed_bleeps(bleep_loc_ms,
                                bleep_spec=200,
                                sr=6144,
                                save_filepath='bleeps.wav'):
    wf = wf_with_timed_bleeps(bleep_loc_ms=bleep_loc_ms, bleep_spec=bleep_spec, sr=sr)
    sf.write(open(save_filepath, 'w'), data=wf, samplerate=sr)
