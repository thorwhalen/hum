"""
A few general utils
"""

from warnings import warn
from inspect import getmodule
from typing import Iterable, Iterator, Tuple, TypeVar, Callable
from itertools import zip_longest
from functools import partial

import numpy as np


T = TypeVar("T")


def simple_chunker(
    a: Iterable[T], chk_size: int, *, include_tail: bool = True
) -> Iterator[Tuple[T, ...]]:
    """
    Chunks an iterable into non-overlapping chunks of size `chk_size`.

    Note: This chunker is simpler, but also less efficient than `chunk_iterable`.
    It does have the extra `include_tail` argument, though.
    Though note that you can get the effect of `include_tail=False` in `chunk_iterable`
    by using `filter(lambda x: len(x) == chk_size, chunk_iterable(...))`.

    Args:
        a: The iterable to be chunked.
        chk_size: The size of each chunk.
        include_tail: If True, includes the remaining elements as the last chunk
                      even if they are fewer than `chk_size`. Defaults to True.

    Returns:
        An iterator of tuples, where each tuple is a chunk of size `chk_size`
        (or fewer elements if `include_tail` is True).

    Examples:
        >>> list(simple_chunker(range(8), 3))
        [(0, 1, 2), (3, 4, 5), (6, 7)]
        >>> list(simple_chunker(range(8), 3, include_tail=False))
        [(0, 1, 2), (3, 4, 5)]
    """
    it = iter(a)
    if include_tail:
        sentinel = object()
        for chunk in zip_longest(*([it] * chk_size), fillvalue=sentinel):
            yield tuple(item for item in chunk if item is not sentinel)
    else:
        yield from zip(*([it] * chk_size))


from functools import partial
from collections.abc import Iterable


def round_numbers(items, round_to=0.001, *, index_of_item_number=None, egress=None):
    """
    Round numbers in an iterable, optionally extracting the number to round from an index.

    Parameters:
    - items: iterable of numbers or iterable of tuples/lists containing numbers.
    - round_to: float, round to the nearest multiple of this value.
    - index_of_item_number: int or None. If None, round the item directly;
      otherwise, round the item at this index in each iterable element.

    Returns:
    - generator yielding items with rounded numbers.

    Examples:
    >>> list(round_numbers([1.23, 3.14159], round_to=0.1))
    [1.2, 3.1]

    >>> items = [(1.234, 'one'), (3.14159, 'three')]
    >>> list(round_numbers(items, round_to=0.1, index_of_item_number=0))
    [[1.2, 'one'], [3.1, 'three']]
    """
    digits = len(str(round_to).split(".")[-1])

    for item in items:
        if index_of_item_number is None:
            yield round(round(item / round_to) * round_to, digits)
        else:
            val = item[index_of_item_number]
            rounded_val = round(round(val / round_to) * round_to, digits)
            item = list(item)
            item[index_of_item_number] = rounded_val

            if egress is None:
                yield item
            else:
                yield egress(item)


def getmodulename(obj, default=""):
    """Get name of module of object"""
    return getattr(getmodule(obj), "__name__", default)


class ModuleNotFoundErrorNiceMessage:
    def __init__(self, msg=None):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            msg = (
                self.msg
                or f"""
It seems you don't have required `{exc_val.name}` package for this Store.
Try installing it by running:

    pip install {exc_val.name}

in your terminal.
For more information: https://pypi.org/project/{exc_val.name}
            """
            )
            warn(msg)


class ModuleNotFoundIgnore:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            pass
        return True
