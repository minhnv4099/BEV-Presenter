#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from typing import Any, Optional
from mmengine.fileio import dump as mmdump
from mmengine.fileio import load as mmload
from os import path as osp


def dump(obj: Any,
         file: Optional[str] = None,
         file_format: Optional[str] = None,
         file_client_args: Optional[dict] = None,
         backend_args: Optional[dict] = None,
         **kwargs):
    if file:
        base_name, ext = osp.splitext(file)
        if ext:
            file_format = ext[1:]
        else:
            # file with no extension
            file_format = 'json'
    else:
        # dump as json
        file_format = 'json'

    if file_format == 'json':
        kwargs.setdefault('indent', 2)
    else:
        # others don't have indent
        kwargs.pop('indent', None)

    return mmdump(
        obj,
        file,
        file_format,
        file_client_args,
        backend_args,
        **kwargs
    )


def load(file,
         file_format: Optional[str] = None,
         file_client_args: Optional[dict] = None,
         backend_args: Optional[dict] = None,
         **kwargs):
    return mmload(file, file_format, file_client_args, backend_args, **kwargs)
