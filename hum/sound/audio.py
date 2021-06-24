"""
Utils to view, hear, and manipulate audio
"""
from numpy import array, max, log10, ceil, int16, hstack, zeros, argmin, ndim
from numpy.random import randint
import soundfile as sf
import os
import matplotlib.pylab as plt
from IPython.display import Audio

from hum.utils.plotting import plot_wf
from hum.utils.librosa_utils import specshow, melspectrogram, amplitude_to_db


default_sr = 44100
default_wf_type = int16


def subtype_of_wf(wf):
    if len(wf) > 0:
        if isinstance(wf[0], (int, int16)):
            return 'PCM_16'
        else:
            return 'FLOAT'
    else:
        return None


def stereo_to_mono_by_taking_first_channel(wf):
    if ndim(wf) == 1:
        return wf
    else:
        likely_channel_dim = argmin(wf.shape)
        if likely_channel_dim == 0:
            return wf[0, :]
        else:
            return wf[:, 0]


def ensure_channels_in_columns(wf):
    likely_channel_dim = argmin(wf.shape)
    if likely_channel_dim == 0:
        return wf.T
    else:
        return wf


def ensure_mono(wf):
    if ndim(wf) == 1:
        return wf
    else:
        return stereo_to_mono_by_taking_first_channel(wf)


def is_wav_file(filepath):
    return os.path.splitext(filepath)[1] == '.wav'


def plot_melspectrogram(spect_mat, sr=default_sr, hop_length=512, name=None):
    # Make a new figure
    plt.figure(figsize=(12, 4))
    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    specshow(spect_mat, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    # Put a descriptive title on the plot
    if name is not None:
        plt.title('mel power spectrogram of "{}"'.format(name))
    else:
        plt.title('mel power spectrogram')
    # draw a color bar
    plt.colorbar(format='%+02.0f dB')
    # Make the figure layout compact
    plt.tight_layout()


def wf_and_sr(*args, **kwargs):
    """
    :param args: Either the file path to a sound or a tuple of (wf, sr)
    :param kwargs: wf = wf, sr = sr
    :return: wf, sr
    """
    if len(args) > 0:
        args_0 = args[0]
        if isinstance(args_0, str):
            kwargs['filepath'] = args_0
        elif isinstance(args_0, tuple):
            kwargs['wf'], kwargs['sr'] = args_0
    kwargs_keys = list(kwargs.keys())
    if 'wf' in kwargs_keys:
        return kwargs['wf'], kwargs['sr']


class Sound(object):
    def __init__(self, wf=None, sr=default_sr, wf_type=default_wf_type):
        if wf is None:
            wf = array([], dtype=wf_type)
        self.wf = wf
        self.sr = sr

    def copy(self):
        return self.__class__(wf=self.wf.copy(), sr=self.sr)

    def __len__(self):
        return len(self.wf)

    def convert_interval_to_samples_unit(self, interval):
        if isinstance(interval, tuple) and len(interval) <= 3:
            interval = slice(*interval)
        if (
            isinstance(interval.start, float)
            or isinstance(interval.stop, float)
            or isinstance(interval.step, float)
        ):
            if interval.start is not None:
                _start = int(interval.start * self.sr)
            else:
                _start = None
            if interval.stop is not None:
                _stop = int(interval.stop * self.sr)
            else:
                _stop = None
            if interval.step is not None:
                _step = int(interval.step * self.sr)
            else:
                _step = None
            interval = slice(_start, _stop, _step)

        return interval

    def __getitem__(self, item):
        item = self.convert_interval_to_samples_unit(item)
        return self.__class__(self.wf.__getitem__(item), self.sr)

    @property
    def duration_s(self):
        """
        Deprecated: Use chk_size_ms instead. Just included here for back-compatibility.
        :return: duration, in seconds, of the sound
        """
        return len(self.wf) / self.sr

    ####################################################################################################################
    # CREATION

    @classmethod
    def from_file(cls, filepath, wf_type=default_wf_type):
        """
        Construct sound object from a wav audio file
        :param filepath: filepath of the sound file
        :param wf_type: type of the wf numbers
        :return: a Sound object
        """
        # kwargs = dict({'always_2d': False, 'ensure_mono': True}, **kwargs)

        wf, sr = sf.read(filepath, dtype=wf_type)
        wf = ensure_mono(wf)
        return cls(wf=wf, sr=sr)

    @classmethod
    def from_(cls, sound):
        """
        Construct sound object from another sound object
        """
        if isinstance(sound, tuple) and len(sound) == 2:  # then it's a (wf, sr) tuple
            return cls(sound[0], sound[1])
        elif isinstance(sound, str) and os.path.isfile(sound):
            return cls.from_file(sound)
        elif isinstance(sound, dict):
            if 'wf' in sound and 'sr' in sound:
                return cls(sound['wf'], sound['sr'])
            else:
                return cls.from_sref(sound)
        elif hasattr(sound, 'wf') and hasattr(sound, 'sr'):
            return cls(sound.wf, sound.sr)
        else:
            raise TypeError("Couldn't figure out how that format represents sound")

    @classmethod
    def silence(cls, seconds=0.0, sr=default_sr, wf_type=default_wf_type):
        """
        Construct silent sound object with length seconds
        >>> assert sum(sum(Sound.silence(seconds = 1).melspectr_matrix())) == 0.0
        """
        return cls(sr=sr, wf=zeros(int(round(seconds * sr)), wf_type))

    def save_to_wav(self, filepath=None, samplerate=None, **kwargs):
        subtype = kwargs.get('subtype', subtype_of_wf(self.wf))
        samplerate = samplerate or self.sr
        if isinstance(filepath, int):
            rand_range = filepath
            template = 'sound_save_{:0' + str(int(ceil(log10(rand_range)))) + '.0}.wav'
            filepath = template.format(randint(0, rand_range))
        else:
            filepath = filepath or 'sound_save.wav'
        sf.write(filepath, self.wf, samplerate=samplerate, subtype=subtype, **kwargs)

    ####################################################################################################################
    # TRANSFORMATIONS

    def ensure_mono(self):
        self.wf = ensure_mono(self.wf)

    def crop_with_idx(self, first_idx, last_idx):
        cropped_sound = self.copy()
        cropped_sound.wf = cropped_sound.wf[first_idx : (last_idx + 1)]
        return cropped_sound

    def crop_with_seconds(self, first_second, last_second):
        return self.crop_with_idx(
            int(round(first_second * self.sr)), int(round(last_second * self.sr)),
        )

    def melspectr_matrix(self, **mel_kwargs):
        """
        Returns a melspectrogram matrix to be used to display a melspectrogram
        """
        mel_kwargs = dict(
            {'n_fft': 2048, 'hop_length': 512, 'n_mels': 128}, **mel_kwargs
        )
        S = melspectrogram(array(self.wf).astype(float), sr=self.sr, **mel_kwargs)
        # Convert to log scale (dB). We'll use the peak power as reference.
        return amplitude_to_db(S, ref=max)

    def __add__(self, append_sound):
        assert (
            self.sr == append_sound.sr
        ), 'Sounds need to have the same sample rate to be appended'
        return Sound(sr=self.sr, wf=hstack(self.wf, append_sound.wf))

    ####################################################################################################################
    # DISPLAY FUNCTIONS

    def hear(self, autoplay=False, **kwargs):
        """
        Display UI to play sound
        """
        wf = array(ensure_mono(self.wf)).astype(float)
        wf[
            randint(len(wf))
        ] *= 1.001  # hack to avoid having exactly the same sound twice (creates an Audio bug)
        return Audio(data=wf, rate=self.sr, autoplay=autoplay, **kwargs)

    def plot_wf(*args, **kwargs):
        wf, sr = wf_and_sr(*args, **kwargs)
        return plot_wf(wf, sr)

    def display(self, autoplay=False, **kwargs):
        """
        Display a melspectrogram of sound and UI to play sound
        """
        self.melspectrogram(plot_it=True, **kwargs)

        return self.hear(autoplay=autoplay)

    def melspectrogram(self, plot_it=False, **mel_kwargs):
        """
        Returns a melsepectrogram matrix and plots a melspectrogram if plot_it is True
        """
        mel_kwargs = dict(
            {'n_fft': 2048, 'hop_length': 512, 'n_mels': 128}, **mel_kwargs
        )
        log_S = self.melspectr_matrix(**mel_kwargs)
        if plot_it:
            plot_melspectrogram(log_S, sr=self.sr, hop_length=mel_kwargs['hop_length'])
        return log_S
