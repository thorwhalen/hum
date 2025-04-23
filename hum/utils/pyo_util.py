"""pyo utils"""

import os
import tempfile
import inspect

from pyo import *

DFLT_PYO_SR = 44100
DFLT_PYO_NCHNLS = 1
DFLT_PYO_AUDIO = "portaudio"
DFLT_PYO_VERBOSITY = 1


class PyoServer(Server):
    """
    A subclass of the pyo Server class that makes it a context manager.

    See pyo issue #300: https://github.com/belangeo/pyo/issues/300

    """

    _boot_with_new_buffer = True

    def __enter__(self):
        self.boot(newBuffer=self._boot_with_new_buffer)
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.shutdown()


class SigControl:
    """
    A class to hold the value and time of a signal.
    This is used to explicitly specify that a number is to be treated as a value, or as a time,
    or both.
    """

    def __init__(self, value=None, time=None):
        self.value = value
        self.time = time


class SigWrapper:
    """
    A wrapper around a pyo SigTo object to allow for easier access to its attributes.
    """

    def __init__(self, sigto_obj):
        self._sig = sigto_obj

    def __call__(self, val):
        self._sig.value = val

    def update(self, val):
        if isinstance(val, SigControl):
            if val.time is not None:
                self._sig.time = val.time
            if val.value is not None:
                self._sig.value = val.value
        else:
            self._sig.value = val

    def __getattr__(self, name):
        return getattr(self._sig, name)

    def __setattr__(self, name, val):
        if name in {"value", "time"}:
            setattr(self._sig, name, val)
        elif name == "_sig":
            super().__setattr__(name, val)
        else:
            raise AttributeError(f"Cannot set {name} on SigWrapper")

    def __repr__(self):
        return f"<SigWrapper value={self._sig.value}, time={self._sig.time}>"


class ParamSet:
    """
    A class to hold a set of parameters for a PyoObject.
    This is used to allow for easier access to the parameters of a PyoObject.
    """

    def __init__(self, param_dict):
        self._params = {k: SigWrapper(v) for k, v in param_dict.items()}

    def __getitem__(self, key):
        return self._params[key]

    def __setitem__(self, key, value):
        self._params[key].update(value)


# TODO: Add validation of values
def get_pyoobj_params(pyoobj):
    """
    Get the parameters of a PyoObject subclass.
    """
    # get the signature of the function
    signa = inspect.signature(pyoobj)
    # get the dict of parameters, using names as keys and .default as values if provided, and None if not
    specs = {
        k: (v.default if v.default is not inspect.Parameter.empty else None)
        for k, v in signa.parameters.items()
    }
    return specs


class RealTimeSynth:
    default_time = 0.05

    def __init__(
        self,
        graph_func,
        *,
        sr=DFLT_PYO_SR,
        nchnls=DFLT_PYO_NCHNLS,
        audio='portaudio',
        verbosity=DFLT_PYO_VERBOSITY,
        **server_kwargs,
    ):
        server_kwargs = dict(
            server_kwargs, sr=sr, nchnls=nchnls, verbosity=verbosity, audio=audio
        )
        self._server_kwargs = server_kwargs
        self.server = Server(**self._server_kwargs).boot()
        self.server.start()

        param_specs = get_pyoobj_params(graph_func)

        # resolve the parameters to SigTo objects
        # if the parameter is a dict, use the value and time keys
        # if the parameter is a number, use it as the value and the default time
        raw_params = {
            name: SigTo(
                value=(spec['value'] if isinstance(spec, dict) else spec),
                time=(
                    spec.get('time', self.default_time)
                    if isinstance(spec, dict)
                    else self.default_time
                ),
            )
            for name, spec in param_specs.items()
        }

        self.sound_params = ParamSet(raw_params)
        self.output = graph_func(**raw_params).out()

    def start(self):
        self.output.out()
        self.server.start()

    def stop(self):
        self.output.stop()
        self.server.stop()

    def __del__(self):
        self.server.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


class OfflineSynthRenderer:
    def __init__(
        self,
        graph_func,
        parameter_frames,
        *,
        frame_durations=1.0,
        output_filepath=None,
        egress=lambda x: x,
        default_time=0.05,
        sr=DFLT_PYO_SR,
        nchnls=DFLT_PYO_NCHNLS,
        verbosity=DFLT_PYO_VERBOSITY,
        **server_kwargs,
    ):
        server_kwargs = dict(
            server_kwargs, sr=sr, nchnls=nchnls, verbosity=verbosity, audio='offline'
        )
        self._server_kwargs = server_kwargs
        self.server = Server(**self._server_kwargs).boot()

        self.graph_func = graph_func
        self.parameter_frames = parameter_frames
        self.output_filepath = output_filepath

        if isinstance(frame_durations, (float, int)):
            self.durations = [frame_durations] * len(parameter_frames)
        elif len(frame_durations) != len(parameter_frames):
            raise ValueError(
                "`frame_durations` must be a float or a list of the same length as `parameter_frames`."
            )
        else:
            self.durations = frame_durations

        self.total_duration = sum(self.durations)

        self.table = NewTable(length=self.total_duration)
        self.egress = egress
        self.default_time = default_time

    def render(self):
        first_frame = self.parameter_frames[0]
        raw_params = {
            key: SigTo(value=first_frame[key], time=self.default_time)
            for key in first_frame
        }

        graph_output = self.graph_func(**raw_params)

        try:
            recorder = TableRec(graph_output, table=self.table).play()

            for frame, dur in zip(self.parameter_frames, self.durations):
                for key, value in frame.items():
                    raw_params[key].value = value
                self.server.recordOptions(dur=dur)
                self.server.start()
        finally:
            recorder.stop()

        # Handle file saving and return bytes
        if self.output_filepath is None:
            tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            filepath = tmpfile.name
            tmpfile.close()
        else:
            filepath = self.output_filepath

        self.table.save(filepath)

        if self.egress is None:
            return None
        elif callable(self.egress):
            with open(filepath, "rb") as f:
                data = f.read()
            return self.egress(data)
        else:
            raise ValueError("egress must be a callable or None")
