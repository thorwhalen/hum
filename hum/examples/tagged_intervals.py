import random
from more_itertools import distribute
from hum.gen.sine_mix import dflt_wf_params_to_wf
from hum.util import simple_chunker

from more_itertools import minmax
import numpy as np
from i2 import Pipe
from functools import partial

DFLT_SR = 44100
DFLT_FACTOR = 100
DFLT_TAG_MODEL = {
    'normal': lambda last_item: {
        'bt': (last_item['tt'] if last_item else 0)
        + abs(random.gauss(mu=0, sigma=0.5)),
        'duration': 0.1 + abs(random.gauss(mu=0, sigma=1)),
    },
    'abnormal': lambda last_item: {
        'bt': (last_item['tt'] if last_item else 0)
        + abs(random.gauss(mu=0.5, sigma=1)),
        'duration': 0.05 + abs(random.gauss(mu=0, sigma=0.25)),
    },
}

rpm = Pipe(partial(random.uniform, 400, 800), int)


def mean_crossing_count(arr):
    mean_ = np.mean(arr)
    return int(sum(np.diff(np.array(arr) > mean_)))


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
    return (interval['tt'] - interval['bt']) * sr


def identity(x):
    return x


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
                generated['bt'],
                (last_items[tag]['tt'] if last_items[tag] else start_bt_s),
            )
            tt = bt + generated['duration']
            last_items[tag] = {'bt': bt, 'tt': tt}
            yield {'tag': tag, 'bt': bt, 'tt': tt}


def intervals_to_wf(intervals, sr=DFLT_SR, rescaler=identity):
    _, max_tt = minmax(intervals, key=lambda x: x['tt'])
    end_wf = int(max_tt['tt'] * sr)
    wf = np.zeros(end_wf + 1)
    for interval in intervals:
        wf_params = tag_model_to_params(tag_model=DFLT_TAG_MODEL)[interval['tag']]
        n_samples = int(interval_to_duration(interval))
        bt = int(interval['bt'] * sr)
        tt = bt + n_samples
        wf[bt:tt] = dflt_wf_params_to_wf(wf_params, n_samples=n_samples, sr=sr)
    return wf


def intervals_to_plc(intervals, sr=DFLT_SR, factor=DFLT_FACTOR, rescaler=identity):
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

    # TODO: This is just a hack to make the PLC have a different sr than wf
    #  --> Remove when the proper PLC generator is implemented
    chks = simple_chunker(wf, chk_size=factor)
    wf = list(map(mean_crossing_count, chks))

    return wf


def mk_channel_data(channel, data, ts, sr):
    return {'channel': channel, 'data': list(data), 'ts': ts, 'sr': sr}


def rescale_intervals(intervals, rescaler):
    return [
        {
            'tag': interval['tag'],
            'bt': rescaler(interval['bt']),
            'tt': rescaler(interval['tt']),
        }
        for interval in intervals
    ]


def intervals_to_json(intervals, sr=DFLT_SR, factor=DFLT_FACTOR, rescaler=identity):
    import json

    result = dict()
    result['data'] = list()
    wf = intervals_to_wf(intervals, rescaler=rescaler)
    plc = intervals_to_plc(intervals, rescaler=rescaler)

    first_ts = rescaler(intervals[0]['bt'])
    result['data'].append(mk_channel_data('wf', wf, first_ts, sr))
    result['data'].append(mk_channel_data('plc', plc, first_ts, sr // factor))
    result['data'].append(
        {'channel': 'annot', 'data': rescale_intervals(intervals, rescaler),}
    )
    volume_list, mixed_list = mk_vol_mixed_data(intervals, wf, rescaler)
    result['data'].append({'channel': 'volume', 'data': volume_list})
    result['data'].append({'channel': 'mixed', 'data': mixed_list})

    return json.dumps(result)


def mk_vol_mixed_data(intervals, wf, rescaler=identity):
    volume_list = list()
    mixed_list = list()
    for interval in intervals:
        wf_chunk = read_chk(wf, interval)
        volume = np.std(wf_chunk)
        mean = np.mean(wf_chunk)
        volume_list.append({'value': volume, 'ts': rescaler(interval['bt'])})
        mixed_list.append(
            {'values': {'mean': mean, 'std': volume}, 'ts': rescaler(interval['bt'])}
        )

    return volume_list, mixed_list


def read_chk(wf, interval, sr=DFLT_SR):
    wf_chunk = wf[int(interval['bt'] * sr) : int(interval['tt'] * sr)]
    return wf_chunk


if __name__ == '__main__':
    intervals = list(
        tagged_intervals_gen(tag_model=DFLT_TAG_MODEL, n_items=3, start_bt_s=0)
    )
    RESCALER = lambda x: int(x * DFLT_SR)

    json_str = intervals_to_json(intervals, rescaler=RESCALER)
    print(json_str)
