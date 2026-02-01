#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from typing import Callable, Any, Optional
import functools
from torch import Tensor, FloatTensor
import inspect

def add_required_attention_args(func: Callable[..., Any]):
    @functools.wraps(func)
    def func_wrapper(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        identity: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        key_pos: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        head_mask: Optional[FloatTensor] = None,
        **kwargs
    ):
        ...
    