#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from mmengine.fileio import dump as mmdump
from mmengine.fileio import load as mmload
from typing import Any
from os import path as osp


def dump(obj,
         file=None,
         file_format='json',
         file_client_args=None,
         backend_args=None,
         indent: int = None,
         **kwargs):

    if file is None:
        file_format = file_format or 'json'

    indent = indent or 3

    return mmdump(
        obj,
        file,
        file_format,
        file_client_args,
        backend_args,
        indent=indent,
        **kwargs
    )


def load(file,
         file_format=None,
         file_client_args=None,
         backend_args=None,
         **kwargs):
    return mmload(file, file_format, file_client_args, backend_args, **kwargs)


def dump_check_exist(data: Any, file_path: str, replace_exist: bool = False):
    if not osp.exists(file_path):
        dump(data, file_path)
        return True

    return False
