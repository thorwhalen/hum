import random
from more_itertools import distribute
from hum.gen.sine_mix import dflt_wf_params_to_wf
from more_itertools import minmax
import numpy as np
from i2 import Pipe
from functools import partial

DFLT_SR = 44100
DFLT_FACTOR = 100
DFLT_TAG_MODEL = {
    "normal": lambda last_item: {
        "bt": (last_item["tt"] if last_item else 0)
        + abs(random.gauss(mu=0, sigma=0.5)),
        "duration": 0.1 + abs(random.gauss(mu=0, sigma=1)),
    },
    "abnormal": lambda last_item: {
        "bt": (last_item["tt"] if last_item else 0)
        + abs(random.gauss(mu=0.5, sigma=1)),
        "duration": 0.05 + abs(random.gauss(mu=0, sigma=0.25)),
    },
}

rpm = Pipe(partial(random.uniform, 400, 800), int)


def frequs_from_cats(cats, num_freq_per_cat=2):
    groups = distribute(
        num_freq_per_cat,
        tuple(range(200, num_freq_per_cat * (len(cats) + 1) * 200, 200)),
    )

    return {cat: list(group) for cat, group in zip(cats, groups)}


def tag_model_to_params(tag_model):
    keys = list(tag_model.keys())
    return frequs_from_cats(keys)


def interval_to_duration(interval, sr=DFLT_SR):
    return (interval["tt"] - interval["bt"]) * sr


def tagged_intervals_gen(tag_model=None, n_items=None, start_bt_s=0):
    """
    Generates a sequence of tagged intervals in the form of dicts
    of the form {"tag": str, "bt": numerical, "tt": numerical} where bt is
    the lower bound of the interval and tt is the upper bound of the interval.

    :param tag_model: A dict whose keys are the tags and values are the generative
                      model for that tag.
    :param n_items: Number of items to be generated for each tag. If None, one
                    item for each tag will be generated.
    :param start_bt_s: The start time in seconds.
    :return: A generator yielding dicts representing tagged intervals.
    """
    if tag_model is None:
        tag_model = DFLT_TAG_MODEL

    if n_items is None:
        n_items = {tag: 1 for tag in tag_model}
    elif isinstance(n_items, int):
        n_items = {tag: n_items for tag in tag_model}

    last_items = {tag: None for tag in tag_model}

    for tag, generator in tag_model.items():
        for _ in range(n_items[tag]):
            generated = generator(last_items[tag])
            bt = max(
                generated["bt"],
                (last_items[tag]["tt"] if last_items[tag] else start_bt_s),
            )
            tt = bt + generated["duration"]
            last_items[tag] = {"bt": bt, "tt": tt}
            yield {"tag": tag, "bt": bt, "tt": tt}


def intervals_to_wf(intervals, sr=DFLT_SR):
    _, max_tt = minmax(intervals, key=lambda x: x['tt'])
    end_wf = int(max_tt['tt'] * sr)
    wf = np.zeros(end_wf + 1)
    for interval in intervals:
        wf_params = tag_model_to_params(tag_model=DFLT_TAG_MODEL)[interval["tag"]]
        n_samples = int(interval_to_duration(interval))
        bt = int(interval['bt'] * sr)
        tt = bt + n_samples
        wf[bt:tt] = dflt_wf_params_to_wf(wf_params, n_samples=n_samples, sr=sr)
    return wf


def intervals_to_plc(intervals, sr=DFLT_SR, factor=DFLT_FACTOR):
    _, max_tt = minmax(intervals, key=lambda x: x['tt'])
    end_wf = int(max_tt['tt'] * sr)
    wf = np.zeros(end_wf + 1)
    for interval in intervals:
        delta = 0
        n_samples = int(interval_to_duration(interval))
        bt = int(interval['bt'] * sr)
        if interval['tag'] == 'normal':
            delta = 400
        vals = ([delta + rpm()] * factor for _ in range(n_samples // factor))
        flattened = [item for sublist in vals for item in sublist]
        wf[bt : bt + len(flattened)] = flattened
    return wf


def intervals_to_json(intervals):
    import json

    result = dict()
    result['data'] = list()
    for interval in intervals:
        wf = intervals_to_wf(intervals)
        plc = intervals_to_plc(intervals)
        result['data'].append({'channel': 'annot', 'data': interval})
        result['data'].append(
            {'channel': 'wf', 'data': list(wf), 'ts': interval['bt'], 'sr': DFLT_SR}
        )
        result['data'].append(
            {
                'channel': 'plc',
                'data': list(plc),
                'ts': interval['bt'],
                'sr': DFLT_SR // DFLT_FACTOR,
            }
        )
    return json.dumps(result)


if __name__ == '__main__':
    # Example usage:
    intervals = list(
        tagged_intervals_gen(tag_model=DFLT_TAG_MODEL, n_items=3, start_bt_s=0)
    )
    json_str = intervals_to_json(intervals)
    #print(json_str)
