#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
import sys
import time
import inspect
from typing import Optional, Callable, Union, Any
import functools

from .logging import getLogger

logger = getLogger(__name__)


def timing(func: Callable[..., Any]):
    @functools.wraps(func)
    def timing_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = int(time.time() - start_time)

        module = inspect.getmodule(sys._getframe())
        # print(inspect.getmodule(sys._getframe(0)))
        # print(inspect.getmodule(sys._getframe(1)))
        # print(inspect.getmodule(sys._getframe(2)))
        # print(inspect.getmodule(sys._getframe(3)))
        # print(inspect.getmodule(sys._getframe(4)))
        logger.info(f"{func.__name__!r} executed in {elapsed_time}s.")

        return result

    return timing_wrapper
