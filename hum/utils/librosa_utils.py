"""A few utils (specshow, melspectrogram) vendored from librosa.

This code was copied from parts of librosa, and adapted, so as to be able to use
targeted functionality with less dependencies and manual installation
(namely for libsndfile) than librosa has.

Librosa can be found here: https://librosa.org/

Librosa's license follows.

ISC License
Copyright (c) 2013--2017, librosa development team.

Permission to use, copy, modify, and/or distribute this software for any purpose with or
without fee is hereby granted, provided that the above copyright notice and this
permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO
THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE,
DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""
import warnings
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.axes import Axes
from matplotlib.ticker import Formatter, ScalarFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import SymmetricalLogLocator
import matplotlib
from packaging.version import parse as version_parse
import scipy
import scipy.signal
from numpy.lib.stride_tricks import as_strided
import re

MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10

# specshow


def specshow(
    data,
    x_coords=None,
    y_coords=None,
    x_axis=None,
    y_axis=None,
    sr=22050,
    hop_length=512,
    fmin=None,
    fmax=None,
    tuning=0.0,
    bins_per_octave=12,
    key='C:maj',
    Sa=None,
    mela=None,
    thaat=None,
    auto_aspect=True,
    htk=False,
    ax=None,
    **kwargs,
):
    if np.issubdtype(data.dtype, np.complexfloating):
        warnings.warn(
            'Trying to display complex-valued input. ' 'Showing magnitude instead.'
        )
        data = np.abs(data)

    kwargs.setdefault('cmap', cmap(data))
    kwargs.setdefault('rasterized', True)
    kwargs.setdefault('edgecolors', 'None')
    kwargs.setdefault('shading', 'flat')

    all_params = dict(
        kwargs=kwargs,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        tuning=tuning,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length,
        key=key,
        htk=htk,
    )

    # Get the x and y coordinates
    y_coords = __mesh_coords(y_axis, y_coords, data.shape[0], **all_params)
    x_coords = __mesh_coords(x_axis, x_coords, data.shape[1], **all_params)

    axes = __check_axes(ax)

    out = axes.pcolormesh(x_coords, y_coords, data, **kwargs)
    __set_current_image(ax, out)

    axes.set_xlim(x_coords.min(), x_coords.max())
    axes.set_ylim(y_coords.min(), y_coords.max())

    # Set up axis scaling
    __scale_axes(axes, x_axis, 'x')
    __scale_axes(axes, y_axis, 'y')

    # Construct tickers and locators
    __decorate_axis(axes.xaxis, x_axis, key=key, Sa=Sa, mela=mela, thaat=thaat)
    __decorate_axis(axes.yaxis, y_axis, key=key, Sa=Sa, mela=mela, thaat=thaat)

    # If the plot is a self-similarity/covariance etc. plot, square it
    if __same_axes(x_axis, y_axis, axes.get_xlim(), axes.get_ylim()) and auto_aspect:
        axes.set_aspect('equal')
    return out


def cmap(data, robust=True, cmap_seq='magma', cmap_bool='gray_r', cmap_div='coolwarm'):
    data = np.atleast_1d(data)

    if data.dtype == 'bool':
        return get_cmap(cmap_bool, lut=2)

    data = data[np.isfinite(data)]

    if robust:
        min_p, max_p = 2, 98
    else:
        min_p, max_p = 0, 100

    min_val, max_val = np.percentile(data, [min_p, max_p])

    if min_val >= 0 or max_val <= 0:
        return get_cmap(cmap_seq)

    return get_cmap(cmap_div)


def __mesh_coords(ax_type, coords, n, **kwargs):
    """Compute axis coordinates"""

    if coords is not None:
        if len(coords) < n:
            raise Exception(
                'Coordinate shape mismatch: ' '{}<{}'.format(len(coords), n)
            )
        return coords

    coord_map = {
        'linear': __coord_fft_hz,
        'fft': __coord_fft_hz,
        'fft_note': __coord_fft_hz,
        'fft_svara': __coord_fft_hz,
        'hz': __coord_fft_hz,
        'log': __coord_fft_hz,
        'mel': __coord_mel_hz,
        'cqt': __coord_cqt_hz,
        'cqt_hz': __coord_cqt_hz,
        'cqt_note': __coord_cqt_hz,
        'cqt_svara': __coord_cqt_hz,
        'chroma': __coord_chroma,
        'chroma_c': __coord_chroma,
        'chroma_h': __coord_chroma,
        'time': __coord_time,
        's': __coord_time,
        'ms': __coord_time,
        'lag': __coord_time,
        'lag_s': __coord_time,
        'lag_ms': __coord_time,
        'tonnetz': __coord_n,
        'off': __coord_n,
        'tempo': __coord_tempo,
        'fourier_tempo': __coord_fourier_tempo,
        'frames': __coord_n,
        None: __coord_n,
    }

    if ax_type not in coord_map:
        raise Exception('Unknown axis type: {}'.format(ax_type))
    return coord_map[ax_type](n, **kwargs)


def __coord_fourier_tempo(n, sr=22050, hop_length=512, **_kwargs):
    """Fourier tempogram coordinates"""

    n_fft = 2 * (n - 1)
    # The following code centers the FFT bins at their frequencies
    # and clips to the non-negative frequency range [0, nyquist]
    basis = fourier_tempo_frequencies(sr=sr, hop_length=hop_length, win_length=n_fft)
    fmax = basis[-1]
    basis -= 0.5 * (basis[1] - basis[0])
    basis = np.append(np.maximum(0, basis), [fmax])
    return basis


def fourier_tempo_frequencies(sr=22050, win_length=384, hop_length=512):
    # sr / hop_length gets the frame rate
    # multiplying by 60 turns frames / sec into frames / minute
    return fft_frequencies(sr=sr * 60 / float(hop_length), n_fft=win_length)


def frames_to_time(frames, sr=22050, hop_length=512, n_fft=None):
    samples = frames_to_samples(frames, hop_length=hop_length, n_fft=n_fft)

    return samples_to_time(samples, sr=sr)


def samples_to_time(samples, sr=22050):
    return np.asanyarray(samples) / float(sr)


def frames_to_samples(frames, hop_length=512, n_fft=None):
    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    return (np.asanyarray(frames) * hop_length + offset).astype(int)


def __coord_time(n, sr=22050, hop_length=512, **_kwargs):
    """Get time coordinates from frames"""
    return frames_to_time(np.arange(n + 1), sr=sr, hop_length=hop_length)


def __coord_chroma(n, bins_per_octave=12, **_kwargs):
    """Get chroma bin numbers"""
    return np.linspace(0, (12.0 * n) / bins_per_octave, num=n + 1, endpoint=True)


def tempo_frequencies(n_bins, hop_length=512, sr=22050):
    bin_frequencies = np.zeros(int(n_bins), dtype=np.float)

    bin_frequencies[0] = np.inf
    bin_frequencies[1:] = 60.0 * sr / (hop_length * np.arange(1.0, n_bins))

    return bin_frequencies


def __coord_tempo(n, sr=22050, hop_length=512, **_kwargs):
    """Tempo coordinates"""
    basis = tempo_frequencies(n + 2, sr=sr, hop_length=hop_length)[1:]
    edges = np.arange(1, n + 2)
    return basis * (edges + 0.5) / edges


def __coord_cqt_hz(n, fmin=None, bins_per_octave=12, sr=22050, **_kwargs):
    """Get CQT bin frequencies"""
    if fmin is None:
        fmin = note_to_hz('C1')

    # Apply tuning correction
    fmin = fmin * 2.0 ** (_kwargs.get('tuning', 0.0) / bins_per_octave)

    # we drop by half a bin so that CQT bins are centered vertically
    freqs = cqt_frequencies(
        n + 1,
        fmin=fmin / 2.0 ** (0.5 / bins_per_octave),
        bins_per_octave=bins_per_octave,
    )

    if np.any(freqs > 0.5 * sr):
        warnings.warn(
            'Frequency axis exceeds Nyquist. '
            'Did you remember to set all spectrogram parameters in specshow?'
        )

    return freqs


def cqt_frequencies(n_bins, fmin, bins_per_octave=12, tuning=0.0):
    correction = 2.0 ** (float(tuning) / bins_per_octave)
    frequencies = 2.0 ** (np.arange(0, n_bins, dtype=float) / bins_per_octave)

    return correction * fmin * frequencies


def note_to_hz(note, **kwargs):
    return midi_to_hz(note_to_midi(note, **kwargs))


def midi_to_hz(notes):
    return 440.0 * (2.0 ** ((np.asanyarray(notes) - 69.0) / 12.0))


def note_to_midi(note, round_midi=True):
    if not isinstance(note, str):
        return np.array([note_to_midi(n, round_midi=round_midi) for n in note])

    pitch_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    acc_map = {
        '#': 1,
        '': 0,
        'b': -1,
        '!': -1,
        '♯': 1,
        '𝄪': 2,
        '♭': -1,
        '𝄫': -2,
        '♮': 0,
    }

    match = re.match(
        r'^(?P<note>[A-Ga-g])'
        r'(?P<accidental>[#♯𝄪b!♭𝄫♮]*)'
        r'(?P<octave>[+-]?\d+)?'
        r'(?P<cents>[+-]\d+)?$',
        note,
    )
    if not match:
        raise Exception('Improper note format: {:s}'.format(note))

    pitch = match.group('note').upper()
    offset = np.sum([acc_map[o] for o in match.group('accidental')])
    octave = match.group('octave')
    cents = match.group('cents')

    if not octave:
        octave = 0
    else:
        octave = int(octave)

    if not cents:
        cents = 0
    else:
        cents = int(cents) * 1e-2

    note_value = 12 * (octave + 1) + pitch_map[pitch] + offset + cents

    if round_midi:
        note_value = int(np.round(note_value))

    return note_value


def __coord_n(n, **_kwargs):
    """Get bare positions"""
    return np.arange(n + 1)


def __coord_mel_hz(n, fmin=0, fmax=None, sr=22050, htk=False, **_kwargs):
    """Get the frequencies for Mel bins"""

    if fmin is None:
        fmin = 0
    if fmax is None:
        fmax = 0.5 * sr

    basis = mel_frequencies(n, fmin=fmin, fmax=fmax, htk=htk)
    basis[1:] -= 0.5 * np.diff(basis)
    basis = np.append(np.maximum(0, basis), [fmax])
    return basis


def __coord_fft_hz(n, sr=22050, **_kwargs):
    """Get the frequencies for FFT bins"""
    n_fft = 2 * (n - 1)
    # The following code centers the FFT bins at their frequencies
    # and clips to the non-negative frequency range [0, nyquist]
    basis = fft_frequencies(sr=sr, n_fft=n_fft)
    fmax = basis[-1]
    basis -= 0.5 * (basis[1] - basis[0])
    basis = np.append(np.maximum(0, basis), [fmax])
    return basis


def __check_axes(axes):
    """Check if "axes" is an instance of an axis object. If not, use `gca`."""
    if axes is None:
        import matplotlib.pyplot as plt

        axes = plt.gca()
    elif not isinstance(axes, Axes):
        raise Exception(
            '`axes` must be an instance of matplotlib.axes.Axes. '
            'Found type(axes)={}'.format(type(axes))
        )
    return axes


def __set_current_image(ax, img):
    """Helper to set the current image in pyplot mode.
    If the provided ``ax`` is not `None`, then we assume that the user is using the object API.
    In this case, the pyplot current image is not set.
    """

    if ax is None:
        import matplotlib.pyplot as plt

        plt.sci(img)


def __scale_axes(axes, ax_type, which):
    """Set the axis scaling"""

    kwargs = dict()
    if which == 'x':
        if version_parse(matplotlib.__version__) < version_parse('3.3.0'):
            thresh = 'linthreshx'
            base = 'basex'
            scale = 'linscalex'
        else:
            thresh = 'linthresh'
            base = 'base'
            scale = 'linscale'

        scaler = axes.set_xscale
        limit = axes.set_xlim
    else:
        if version_parse(matplotlib.__version__) < version_parse('3.3.0'):
            thresh = 'linthreshy'
            base = 'basey'
            scale = 'linscaley'
        else:
            thresh = 'linthresh'
            base = 'base'
            scale = 'linscale'

        scaler = axes.set_yscale
        limit = axes.set_ylim

    # Map ticker scales
    if ax_type == 'mel':
        mode = 'symlog'
        kwargs[thresh] = 1000.0
        kwargs[base] = 2

    elif ax_type in ['cqt', 'cqt_hz', 'cqt_note', 'cqt_svara']:
        mode = 'log'
        kwargs[base] = 2

    elif ax_type in ['log', 'fft_note', 'fft_svara']:
        mode = 'symlog'
        kwargs[base] = 2
        # kwargs[thresh] = core.note_to_hz(
        #    'C2'
        # )  # in librosa/core.py but I don't think it is needed
        kwargs[scale] = 0.5

    elif ax_type in ['tempo', 'fourier_tempo']:
        mode = 'log'
        kwargs[base] = 2
        limit(16, 480)
    else:
        return

    scaler(mode, **kwargs)


def __decorate_axis(axis, ax_type, key='C:maj', Sa=None, mela=None, thaat=None):
    """Configure axis tickers, locators, and labels"""
    if ax_type == 'time':
        axis.set_major_formatter(TimeFormatter(unit=None, lag=False))
        axis.set_major_locator(MaxNLocator(prune=None, steps=[1, 1.5, 5, 6, 10]))
        axis.set_label_text('Time')
    elif ax_type in ['mel', 'log']:
        axis.set_major_formatter(ScalarFormatter())
        axis.set_major_locator(SymmetricalLogLocator(axis.get_transform()))
        axis.set_label_text('Hz')


class TimeFormatter(Formatter):
    def __init__(self, lag=False, unit=None):

        if unit not in ['s', 'ms', None]:
            raise Exception('Unknown time unit: {}'.format(unit))

        self.unit = unit
        self.lag = lag

    def __call__(self, x, pos=None):
        """Return the time format as pos"""

        _, dmax = self.axis.get_data_interval()
        vmin, vmax = self.axis.get_view_interval()

        # In lag-time axes, anything greater than dmax / 2 is negative time
        if self.lag and x >= dmax * 0.5:
            # In lag mode, don't tick past the limits of the data
            if x > dmax:
                return ''
            value = np.abs(x - dmax)
            # Do we need to tweak vmin/vmax here?
            sign = '-'
        else:
            value = x
            sign = ''

        if self.unit == 's':
            s = '{:.3g}'.format(value)
        elif self.unit == 'ms':
            s = '{:.3g}'.format(value * 1000)
        else:
            if vmax - vmin > 3600:
                # Hours viz
                s = '{:d}:{:02d}:{:02d}'.format(
                    int(value / 3600.0),
                    int(np.mod(value / 60.0, 60)),
                    int(np.mod(value, 60)),
                )
            elif vmax - vmin > 60:
                # Minutes viz
                s = '{:d}:{:02d}'.format(int(value / 60.0), int(np.mod(value, 60)))
            elif vmax - vmin >= 1:
                # Seconds viz
                s = '{:.2g}'.format(value)
            else:
                # Milliseconds viz
                s = '{:.3f}'.format(value)

        return '{:s}{:s}'.format(sign, s)


def __same_axes(x_axis, y_axis, xlim, ylim):
    """Check if two axes are the same, used to determine squared plots"""
    axes_same_and_not_none = (x_axis == y_axis) and (x_axis is not None)
    axes_same_lim = xlim == ylim
    return axes_same_and_not_none and axes_same_lim


# librosa.feature.melspectrogram


def melspectrogram(
    y=None,
    sr=22050,
    S=None,
    n_fft=2048,
    hop_length=512,
    win_length=None,
    window='hann',
    center=True,
    pad_mode='reflect',
    power=2.0,
    **kwargs,
):
    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Build a Mel filter
    mel_basis = mel(sr, n_fft, **kwargs)

    return np.dot(mel_basis, S)


def _spectrogram(
    y=None,
    S=None,
    n_fft=2048,
    hop_length=512,
    power=1,
    win_length=None,
    window='hann',
    center=True,
    pad_mode='reflect',
):
    if S is not None:
        # Infer n_fft from spectrogram shape
        n_fft = 2 * (S.shape[0] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        S = (
            np.abs(
                stft(
                    y,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    center=center,
                    window=window,
                    pad_mode=pad_mode,
                )
            )
            ** power
        )

    return S, n_fft


def stft(
    y,
    n_fft=2048,
    hop_length=None,
    win_length=None,
    window='hann',
    center=True,
    dtype=None,
    pad_mode='reflect',
):
    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
    valid_audio(y)

    # Pad the time series so that frames are centered
    if center:
        if n_fft > y.shape[-1]:
            warnings.warn(
                'n_fft={} is too small for input signal of length={}'.format(
                    n_fft, y.shape[-1]
                )
            )

        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    elif n_fft > y.shape[-1]:
        raise Exception(
            'n_fft={} is too large for input signal of length={}'.format(
                n_fft, y.shape[-1]
            )
        )

    # Window the time series.
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

    if dtype is None:
        dtype = dtype_r2c(y.dtype)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty(
        (int(1 + n_fft // 2), y_frames.shape[1]), dtype=dtype, order='F'
    )

    fft = get_fftlib()

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = MAX_MEM_BLOCK // (stft_matrix.shape[0] * stft_matrix.itemsize)
    n_columns = max(n_columns, 1)

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:, bl_s:bl_t] = fft.rfft(
            fft_window * y_frames[:, bl_s:bl_t], axis=0
        )
    return stft_matrix


def get_window(window, Nx, fftbins=True):
    if callable(window):
        return window(Nx)

    elif isinstance(window, (str, tuple)) or np.isscalar(window):
        # TODO: if we add custom window functions in librosa, call them here

        return scipy.signal.get_window(window, Nx, fftbins=fftbins)

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

        raise Exception('Window size mismatch: ' '{:d} != {:d}'.format(len(window), Nx))
    else:
        raise Exception('Invalid window specification: {}'.format(window))


def pad_center(data, size, axis=-1, **kwargs):
    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise Exception(
            ('Target size ({:d}) must be ' 'at least input size ({:d})').format(size, n)
        )

    return np.pad(data, lengths, **kwargs)


def valid_audio(y, mono=True):
    if not isinstance(y, np.ndarray):
        raise Exception('Audio data must be of type numpy.ndarray')

    if not np.issubdtype(y.dtype, np.floating):
        raise Exception('Audio data must be floating-point')

    if mono and y.ndim != 1:
        raise Exception(
            'Invalid shape for monophonic audio: '
            'ndim={:d}, shape={}'.format(y.ndim, y.shape)
        )

    elif y.ndim > 2 or y.ndim == 0:
        raise Exception(
            'Audio data must have shape (samples,) or (channels, samples). '
            'Received shape={}'.format(y.shape)
        )

    elif y.ndim == 2 and y.shape[0] < 2:
        raise Exception(
            'Mono data must have shape (samples,). ' 'Received shape={}'.format(y.shape)
        )

    if not np.isfinite(y).all():
        raise Exception('Audio buffer is not finite everywhere')

    return True


def frame(x, frame_length, hop_length, axis=-1):
    if not isinstance(x, np.ndarray):
        raise Exception(
            'Input must be of type numpy.ndarray, ' 'given type(x)={}'.format(type(x))
        )

    if x.shape[axis] < frame_length:
        raise Exception(
            'Input is too short (n={:d})'
            ' for frame_length={:d}'.format(x.shape[axis], frame_length)
        )

    if hop_length < 1:
        raise Exception('Invalid hop_length: {:d}'.format(hop_length))

    if axis == -1 and not x.flags['F_CONTIGUOUS']:
        warnings.warn(
            'librosa.util.frame called with axis={} '
            'on a non-contiguous input. This will result in a copy.'.format(axis)
        )
        x = np.asfortranarray(x)
    elif axis == 0 and not x.flags['C_CONTIGUOUS']:
        warnings.warn(
            'librosa.util.frame called with axis={} '
            'on a non-contiguous input. This will result in a copy.'.format(axis)
        )
        x = np.ascontiguousarray(x)

    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    strides = np.asarray(x.strides)

    new_stride = np.prod(strides[strides > 0] // x.itemsize) * x.itemsize

    if axis == -1:
        shape = list(x.shape)[:-1] + [frame_length, n_frames]
        strides = list(strides) + [hop_length * new_stride]

    elif axis == 0:
        shape = [n_frames, frame_length] + list(x.shape)[1:]
        strides = [hop_length * new_stride] + list(strides)

    else:
        raise Exception('Frame axis={} must be either 0 or -1'.format(axis))

    return as_strided(x, shape=shape, strides=strides)


def dtype_r2c(d, default=np.complex64):
    mapping = {
        np.dtype(np.float32): np.complex64,
        np.dtype(np.float64): np.complex128,
        np.dtype(np.float): np.complex,
    }

    # If we're given a complex type already, return it
    dt = np.dtype(d)
    if dt.kind == 'c':
        return dt

    # Otherwise, try to map the dtype.
    # If no match is found, return the default.
    return np.dtype(mapping.get(dt, default))


def get_fftlib():
    global __FFTLIB
    return __FFTLIB


def set_fftlib(lib=None):
    global __FFTLIB
    if lib is None:
        from numpy import fft

        lib = fft

    __FFTLIB = lib


set_fftlib(None)


def mel(
    sr,
    n_fft,
    n_mels=128,
    fmin=0.0,
    fmax=None,
    htk=False,
    norm='slaney',
    dtype=np.float32,
):
    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == 'slaney':
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]
    else:
        weights = normalize(weights, norm=norm, axis=-1)

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn(
            'Empty filters detected in mel frequency basis. '
            'Some channels will produce empty responses. '
            'Try increasing your sampling rate (and fmax) or '
            'reducing n_mels.'
        )

    return weights


def fft_frequencies(sr=22050, n_fft=2048):
    return np.linspace(0, float(sr) / 2, int(1 + n_fft // 2), endpoint=True)


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)


def hz_to_mel(frequencies, htk=False):
    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels, htk=False):
    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def normalize(S, norm=np.inf, axis=0, threshold=None, fill=None):
    # Avoid div-by-zero
    if threshold is None:
        threshold = tiny(S)

    elif threshold <= 0:
        raise Exception('threshold={} must be strictly ' 'positive'.format(threshold))

    if fill not in [None, False, True]:
        raise Exception('fill={} must be None or boolean'.format(fill))

    if not np.all(np.isfinite(S)):
        raise Exception('Input must be finite')

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(np.float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise Exception('Cannot normalize with norm=0 and fill=True')

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag ** norm, axis=axis, keepdims=True) ** (1.0 / norm)

        if axis is None:
            fill_norm = mag.size ** (-1.0 / norm)
        else:
            fill_norm = mag.shape[axis] ** (-1.0 / norm)

    elif norm is None:
        return S

    else:
        raise Exception('Unsupported norm: {}'.format(repr(norm)))

    # indices where norm is below the threshold
    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm


def tiny(x):
    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
        x.dtype, np.complexfloating
    ):
        dtype = x.dtype
    else:
        dtype = np.float32

    return np.finfo(dtype).tiny


# amplitude_to_db


def amplitude_to_db(S, ref=1.0, amin=1e-5, top_db=80.0):
    S = np.asarray(S)

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn(
            'amplitude_to_db was called on complex input so phase '
            'information will be discarded. To suppress this warning, '
            'call amplitude_to_db(np.abs(S)) instead.'
        )

    magnitude = np.abs(S)

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    power = np.square(magnitude, out=magnitude)

    return power_to_db(power, ref=ref_value ** 2, amin=amin ** 2, top_db=top_db)


def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    S = np.asarray(S)

    if amin <= 0:
        raise Exception('amin must be strictly positive')

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn(
            'power_to_db was called on complex input so phase '
            'information will be discarded. To suppress this warning, '
            'call power_to_db(np.abs(D)**2) instead.'
        )
        magnitude = np.abs(S)
    else:
        magnitude = S

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise Exception('top_db must be non-negative')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


# onset_strength


def onset_strength(
    y=None,
    sr=22050,
    S=None,
    lag=1,
    max_size=1,
    ref=None,
    detrend=False,
    center=True,
    feature=None,
    aggregate=None,
    **kwargs,
):
    if aggregate is False:
        raise Exception(
            'aggregate={} cannot be False when computing full-spectrum onset strength.'
        )

    odf_all = onset_strength_multi(
        y=y,
        sr=sr,
        S=S,
        lag=lag,
        max_size=max_size,
        ref=ref,
        detrend=detrend,
        center=center,
        feature=feature,
        aggregate=aggregate,
        channels=None,
        **kwargs,
    )

    return odf_all[0]


def onset_strength_multi(
    y=None,
    sr=22050,
    S=None,
    n_fft=2048,
    hop_length=512,
    lag=1,
    max_size=1,
    ref=None,
    detrend=False,
    center=True,
    feature=None,
    aggregate=None,
    channels=None,
    **kwargs,
):
    if feature is None:
        feature = melspectrogram
        kwargs.setdefault('fmax', 11025.0)

    if aggregate is None:
        aggregate = np.mean

    if lag < 1 or not isinstance(lag, (int, np.integer)):
        raise Exception('lag must be a positive integer')

    if max_size < 1 or not isinstance(max_size, (int, np.integer)):
        raise Exception('max_size must be a positive integer')

    # First, compute mel spectrogram
    if S is None:
        S = np.abs(feature(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, **kwargs))

        # Convert to dBs
        S = power_to_db(S)

    # Ensure that S is at least 2-d
    S = np.atleast_2d(S)

    # Compute the reference spectrogram.
    # Efficiency hack: skip filtering step and pass by reference
    # if max_size will produce a no-op.
    if ref is None:
        if max_size == 1:
            ref = S
        else:
            ref = scipy.ndimage.maximum_filter1d(S, max_size, axis=0)
    elif ref.shape != S.shape:
        raise Exception(
            'Reference spectrum shape {} must match input spectrum {}'.format(
                ref.shape, S.shape
            )
        )

    # Compute difference to the reference, spaced by lag
    onset_env = S[:, lag:] - ref[:, :-lag]

    # Discard negatives (decreasing amplitude)
    onset_env = np.maximum(0.0, onset_env)

    # Aggregate within channels
    pad = True
    if channels is None:
        channels = [slice(None)]
    else:
        pad = False

    if aggregate:
        onset_env = sync(onset_env, channels, aggregate=aggregate, pad=pad, axis=0)

    # compensate for lag
    pad_width = lag
    if center:
        # Counter-act framing effects. Shift the onsets by n_fft / hop_length
        pad_width += n_fft // (2 * hop_length)

    onset_env = np.pad(onset_env, ([0, 0], [int(pad_width), 0]), mode='constant')

    # remove the DC component
    if detrend:
        onset_env = scipy.signal.lfilter([1.0, -1.0], [1.0, -0.99], onset_env, axis=-1)

    # Trim to match the input duration
    if center:
        onset_env = onset_env[:, : S.shape[1]]

    return onset_env


def sync(data, idx, aggregate=None, pad=True, axis=-1):
    if aggregate is None:
        aggregate = np.mean

    shape = list(data.shape)

    if np.all([isinstance(_, slice) for _ in idx]):
        slices = idx
    elif np.all([np.issubdtype(type(_), np.integer) for _ in idx]):
        slices = index_to_slice(np.asarray(idx), 0, shape[axis], pad=pad)
    else:
        raise Exception('Invalid index set: {}'.format(idx))

    agg_shape = list(shape)
    agg_shape[axis] = len(slices)

    data_agg = np.empty(
        agg_shape, order='F' if np.isfortran(data) else 'C', dtype=data.dtype
    )

    idx_in = [slice(None)] * data.ndim
    idx_agg = [slice(None)] * data_agg.ndim

    for (i, segment) in enumerate(slices):
        idx_in[axis] = segment
        idx_agg[axis] = i
        data_agg[tuple(idx_agg)] = aggregate(data[tuple(idx_in)], axis=axis)

    return data_agg


def index_to_slice(idx, idx_min=None, idx_max=None, step=None, pad=True):
    # First, normalize the index set
    idx_fixed = fix_frames(idx, idx_min, idx_max, pad=pad)

    # Now convert the indices to slices
    return [slice(start, end, step) for (start, end) in zip(idx_fixed, idx_fixed[1:])]


def fix_frames(frames, x_min=0, x_max=None, pad=True):
    frames = np.asarray(frames)

    if np.any(frames < 0):
        raise Exception('Negative frame index detected')

    if pad and (x_min is not None or x_max is not None):
        frames = np.clip(frames, x_min, x_max)

    if pad:
        pad_data = []
        if x_min is not None:
            pad_data.append(x_min)
        if x_max is not None:
            pad_data.append(x_max)
        frames = np.concatenate((pad_data, frames))

    if x_min is not None:
        frames = frames[frames >= x_min]

    if x_max is not None:
        frames = frames[frames <= x_max]

    return np.unique(frames).astype(int)


def spectral_contrast(
    y=None,
    sr=22050,
    S=None,
    n_fft=2048,
    hop_length=512,
    win_length=None,
    window='hann',
    center=True,
    pad_mode='reflect',
    freq=None,
    fmin=200.0,
    n_bands=6,
    quantile=0.02,
    linear=False,
):
    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Compute the center frequencies of each bin
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)

    freq = np.atleast_1d(freq)

    if freq.ndim != 1 or len(freq) != S.shape[0]:
        raise Exception('freq.shape mismatch: expected ' '({:d},)'.format(S.shape[0]))

    if n_bands < 1 or not isinstance(n_bands, int):
        raise Exception('n_bands must be a positive integer')

    if not 0.0 < quantile < 1.0:
        raise Exception('quantile must lie in the range (0, 1)')

    if fmin <= 0:
        raise Exception('fmin must be a positive number')

    octa = np.zeros(n_bands + 2)
    octa[1:] = fmin * (2.0 ** np.arange(0, n_bands + 1))

    if np.any(octa[:-1] >= 0.5 * sr):
        raise Exception(
            'Frequency band exceeds Nyquist. ' 'Reduce either fmin or n_bands.'
        )

    valley = np.zeros((n_bands + 1, S.shape[1]))
    peak = np.zeros_like(valley)

    for k, (f_low, f_high) in enumerate(zip(octa[:-1], octa[1:])):
        current_band = np.logical_and(freq >= f_low, freq <= f_high)

        idx = np.flatnonzero(current_band)

        if k > 0:
            current_band[idx[0] - 1] = True

        if k == n_bands:
            current_band[idx[-1] + 1 :] = True

        sub_band = S[current_band]

        if k < n_bands:
            sub_band = sub_band[:-1]

        # Always take at least one bin from each side
        idx = np.rint(quantile * np.sum(current_band))
        idx = int(np.maximum(idx, 1))

        sortedr = np.sort(sub_band, axis=0)

        valley[k] = np.mean(sortedr[:idx], axis=0)
        peak[k] = np.mean(sortedr[-idx:], axis=0)

    if linear:
        return peak - valley
    else:
        return power_to_db(peak) - power_to_db(valley)


# db_to_amplitude


def db_to_amplitude(S_db, ref=1.0):
    return db_to_power(S_db, ref=ref ** 2) ** 0.5


def db_to_power(S_db, ref=1.0):
    return ref * np.power(10.0, 0.1 * S_db)


# spectral_centroid


def spectral_centroid(
    y=None,
    sr=22050,
    S=None,
    n_fft=2048,
    hop_length=512,
    freq=None,
    win_length=None,
    window='hann',
    center=True,
    pad_mode='reflect',
):
    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    if not np.isrealobj(S):
        raise Exception('Spectral centroid is only defined ' 'with real-valued input')
    elif np.any(S < 0):
        raise Exception(
            'Spectral centroid is only defined ' 'with non-negative energies'
        )

    # Compute the center frequencies of each bin
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)

    if freq.ndim == 1:
        freq = freq.reshape((-1, 1))

    # Column-normalize S
    return np.sum(freq * normalize(S, norm=1, axis=0), axis=0, keepdims=True)
