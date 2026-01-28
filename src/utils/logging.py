#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from typing import Optional, Union
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][%(name)s] - %(message)s #%(lineno)d',
)


# avoid conflicting name with get_logger from built-in logging
def getLogger(name: Optional[str] = None, level: Union[int, str] = logging.INFO) -> logging.Logger:

    if name is None:
        name = __name__

    return logging.getLogger(name)
