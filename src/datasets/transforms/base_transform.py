#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from __future__ import annotations

import copy
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, final
import functools

import numpy as np
import torch
from collections import defaultdict
from enum import Enum
from src.utils.logging import getLogger
from src.utils.fileio import dump


logger = getLogger(__name__)
HasShapeType = (np.ndarray, torch.Tensor)


def has_shape(data: Any):
    return isinstance(data, HasShapeType)


class UpdateType(Enum):
    CHANGE_TYPE = 'CHANGE_TYPE'
    CHANGE_SHAPE = 'CHANGE_SHAPE'
    CHANGE_VALUE = 'CHANGE_VALUE'
    UNCHANGED = 'UNCHANGED'


def track_update_dict(func: Callable[[dict], dict]):
    @functools.wraps(func)
    def wrapped_func(obj: "BaseTransform", input_dict: dict, debug: bool = True, *args, **kwargs) -> dict:
        ori_input_dict = copy.deepcopy(input_dict)
        output_dict = obj.transform(input_dict)
        added_keys = []
        kept_keys = []
        eliminated_keys = []
        updated_keys = defaultdict(list)

        for k in output_dict:
            if k not in ori_input_dict:
                added_keys.append(k)

        for k in ori_input_dict:
            if k not in output_dict:
                eliminated_keys.append(k)
            else:
                kept_keys.append(k)
                in_value, out_value = ori_input_dict[k], output_dict[k]

                if not type(in_value) is type(out_value):
                    updated_keys[k].append(UpdateType.CHANGE_TYPE.value)
                if has_shape(in_value) and has_shape(out_value):
                    if in_value.shape != out_value.shape:
                        updated_keys[k].append(UpdateType.CHANGE_SHAPE.value)
                # elif in_value != out_value:
                #     updated_keys[k].append(UpdateType.CHANGE_VALUE.value)

                if not updated_keys[k]:
                    updated_keys[k].append(UpdateType.UNCHANGED.value)

        setattr(obj, 'kept_keys', kept_keys)
        setattr(obj, 'added_keys', added_keys)
        setattr(obj, 'eliminated_keys', eliminated_keys)
        setattr(obj, 'updated_keys', dump(updated_keys, indent=3))

        if debug:
            indent_str = '     '
            logger.info(
                f"{obj.__class__.__name__!r}:\n"
                f"{indent_str}keep: {obj.kept_keys}\n"
                f"{indent_str}removed: {obj.eliminated_keys}\n"
                f"{indent_str}added: {obj.added_keys}\n"
                f"{indent_str}updated: {obj.updated_keys}"
            )

        return output_dict

    return wrapped_func


class BaseTransform(metaclass=ABCMeta):
    """Base class for all transformations.
    With debug mode to check dict which keys are added, removed, keep or updated.
    """

    kept_keys: list[str]
    added_keys: list[str]
    eliminated_keys: list[str]
    updated_keys: dict[str, list[UpdateType]]

    @final
    def __call__(self, results: Dict, debug: bool = True) -> Optional[Union[Dict, Tuple[List, List]]]:
        input_dict = copy.deepcopy(results)
        output_dict = self.transform(results)

        if debug:
            self.check_update_key(input_dict, output_dict)
            indent_str = '     '
            logger.info(
                f"{self.__class__.__name__!r}:\n"
                f"{indent_str}keep: {self.kept_keys}\n"
                f"{indent_str}removed: {self.eliminated_keys}\n"
                f"{indent_str}added: {self.added_keys}\n"
                f"{indent_str}updated: {self.updated_keys}"
            )

        return output_dict

    @abstractmethod
    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function. All subclass of BaseTransform should
        override this method.

        This function takes the result dict as the input, and can add new
        items to the dict or modify existing items in the dict. And the result
        dict will be returned in the end, which allows to concate multiple
        transforms into a pipeline.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """

    def check_update_key(self, ori_input_dict: dict, output_dict: dict):
        added_keys = []
        kept_keys = []
        eliminated_keys = []
        updated_keys = defaultdict(list)

        for k in output_dict:
            if k not in ori_input_dict:
                added_keys.append(k)

        for k in ori_input_dict:
            if k not in output_dict:
                eliminated_keys.append(k)
            else:
                kept_keys.append(k)
                in_value, out_value = ori_input_dict[k], output_dict[k]

                if not type(in_value) is type(out_value):
                    updated_keys[k].append(UpdateType.CHANGE_TYPE.value)
                if has_shape(in_value) and has_shape(out_value):
                    if in_value.shape != out_value.shape:
                        updated_keys[k].append(UpdateType.CHANGE_SHAPE.value)
                # elif in_value != out_value:
                #     updated_keys[k].append(UpdateType.CHANGE_VALUE.value)

                if not updated_keys[k]:
                    updated_keys.pop(k)

        self.kept_keys = kept_keys
        self.added_keys = added_keys
        self.eliminated_keys = eliminated_keys
        self.updated_keys = dict(updated_keys)
