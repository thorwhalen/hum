"""
Generate Audio
"""
from contextlib import suppress

with suppress(ModuleNotFoundError, ImportError):
    from hum.gen.diagnosis_sounds import (
        WfGen,
        TimeSound,
        BinarySound,
        slow_mask,
        wf_with_timed_bleeps,
        mk_sounds_with_timed_bleeps,
    )

with suppress(ModuleNotFoundError, ImportError):
    from hum.gen.sine_mix import mk_sine_wf, freq_based_stationary_wf

with suppress(ModuleNotFoundError, ImportError):
    from hum.gen.voiced_time import Voicer, tell_time_continuously

with suppress(ModuleNotFoundError, ImportError):
    from hum.gen.signal_generators import (
        normal_dist,
        gen_words,
        categorical_gen,
        alphabet_to_bins,
        call_repeatedly,
        bernoulli,
        bernoulli_gen,
        inlier_outlier,
        signal,
        create_session,
        string_to_num,
        session_to_df,
    )
