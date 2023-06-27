"""
A few general utils
"""
from warnings import warn
from inspect import getmodule
from typing import Iterable


def simple_chunker(a: Iterable, chk_size: int):
    """Generate fixed sized non-overlapping chunks of an iterable ``a``.

    >>> list(simple_chunker(range(7), 3))
    [(0, 1, 2), (3, 4, 5)]
    """
    return zip(*([iter(a)] * chk_size))


def getmodulename(obj, default=''):
    """Get name of module of object"""
    return getattr(getmodule(obj), '__name__', default)


class ModuleNotFoundErrorNiceMessage:
    def __init__(self, msg=None):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            msg = (
                self.msg
                or f'''
It seems you don't have required `{exc_val.name}` package for this Store.
Try installing it by running:

    pip install {exc_val.name}

in your terminal.
For more information: https://pypi.org/project/{exc_val.name}
            '''
            )
            warn(msg)


class ModuleNotFoundIgnore:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            pass
        return True
