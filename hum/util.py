"""
A few general utils
"""
from warnings import warn
from inspect import getmodule


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
