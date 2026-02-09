#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from typing import Iterable, Union, Any, Sequence, Optional


def is_sequence(seq):
    return isinstance(seq, list)


def construct_list(
    data: Union[Any, Sequence[Any]],
    length: Optional[int] = None
) -> list:
    if is_sequence(data):
        return data

    return [data] * length
