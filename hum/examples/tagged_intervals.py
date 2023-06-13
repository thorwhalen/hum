import random
from more_itertools import distribute
from hum.gen.sine_mix import dflt_wf_params_to_wf


DFLT_SR = 44100
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


if __name__ == '__main__':
    # Example usage:
    intervals = tagged_intervals_gen(tag_model=DFLT_TAG_MODEL, n_items=3, start_bt_s=0)

    for interval in intervals:
        duration = interval_to_duration(interval)
        wf_params = tag_model_to_params(tag_model=DFLT_TAG_MODEL)[interval["tag"]]
        wf = dflt_wf_params_to_wf(wf_params)
        print(interval, duration, wf_params, wf)
