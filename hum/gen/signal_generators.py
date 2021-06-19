"""
Generating Signals
"""
import numpy as np
import random
from itertools import chain
import pandas as pd

DFLT_WORD_LENGTH = 30
DFLT_ALPHABET = list('abcde')


def normal_dist(mu, sigma):
    return sigma * np.random.randn() + mu


def gen_words(
    N=DFLT_WORD_LENGTH, alphabet=DFLT_ALPHABET, spread_pct=0.01, proba_dist='normal',
):
    if proba_dist == 'normal':
        sigma = N * spread_pct
        mu = N
        dist = normal_dist(mu, sigma)
    else:
        raise NotImplementedError(
            f'Probability distribution {proba_dist} not implemented'
        )
    while True:
        length = dist
        word = [random.choice(alphabet)] * int(length)
        yield word


def categorical_gen(gen_it):
    yield from chain.from_iterable(gen_it())


def alphabet_to_bins(alphabet=list('abcde')):
    length = len(alphabet)
    low = 0.0
    high = 10.0 * length
    bins = np.linspace(low, high, length + 1)
    # print(bins)
    return {letter: (bins[i], bins[i + 1]) for i, letter in enumerate(alphabet)}


# def context_to_signal(cat_gen, dict_symbol_to_interval)
#   symbol = next(cat_gen)
#   low, hi = dict_symbol_to_interval[symbol]
#   res = np.random.uniform(low=0.0, high=1.0)
def call_repeatedly(func, *args, **kwargs):
    while True:
        yield func(*args, **kwargs)


def bernoulli(p_out=0.1):
    a = [0, 1]
    p = [1.0 - p_out, p_out]
    return int(np.random.choice(a, size=1, replace=True, p=p))


def bernoulli_gen(p_out=0.5):
    return call_repeatedly(bernoulli, p_out=p_out)


def inlier_outlier(segment, interval_size, outlier_status):
    low, high = segment
    if outlier_status == 0:
        return np.random.uniform(low, high)
    else:
        low, high = high, low + interval_size
        return np.random.uniform(low, high) % interval_size


def signal(symbol_gen, outlier_gen, alphabet):
    while True:
        symb = next(symbol_gen)
        outlier = next(outlier_gen)
        length = len(alphabet)
        low, high = alphabet_to_bins(alphabet)[symb]
        yield inlier_outlier((low, high), length * 10, outlier)


def create_session(symbol_gen, outlier_gen, alphabet, session_length=50):
    symbs = []
    outliers = []
    sigs = []
    for _ in range(session_length):
        symb = next(symbol_gen)
        symbs.append(symb)
        outlier = next(outlier_gen)
        outliers.append(outlier)
        length = len(alphabet)
        low, high = alphabet_to_bins(alphabet)[symb]
        sigs.append(inlier_outlier((low, high), length * 10, outlier))
    return symbs, outliers, sigs


def string_to_num(word):
    all_letters = sorted(list(set(word)))
    return [all_letters.index(letter) for letter in word]


def session_to_df(session):
    symbs, outliers, sigs = session
    df = pd.DataFrame()
    df['symbols'] = string_to_num(''.join(symbs))
    df['outliers'] = outliers
    df['signal'] = sigs
    return df
