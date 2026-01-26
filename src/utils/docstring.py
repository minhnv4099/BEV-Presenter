#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from typing import Optional, Callable
import inspect
import functools


def auto_docstring(custom_intro: Optional[str] = ''):
    def docstring_adder(func):
        @functools.wraps(func)
        def wrapper(*arg, **kwargs):
            pass

        return wrapper

    if inspect.isclass(custom_intro) or inspect.ismethod(custom_intro):
        return docstring_adder(custom_intro)

    return docstring_adder
