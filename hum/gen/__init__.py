from hum.util import ModuleNotFoundIgnore

with ModuleNotFoundIgnore():
    from hum.gen.diagnosis_sounds import (
        WfGen, TimeSound, BinarySound, slow_mask, wf_with_timed_bleeps, mk_sounds_with_timed_bleeps
    )

with ModuleNotFoundIgnore():
    from hum.gen.sine_mix import mk_sine_wf, freq_based_stationary_wf

with ModuleNotFoundIgnore():
    from hum.gen.voiced_time import Voicer, tell_time_continuously
