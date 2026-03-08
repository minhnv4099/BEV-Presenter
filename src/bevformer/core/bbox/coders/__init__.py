#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from .nms_free_coder import NMSFreeCoder
from src.registry import BBOX_CODERS
from src.typing_ import ConfigType
from src.bevformer.builder import _build_with_fallback


def build_bbox_coder(config: ConfigType):
    return _build_with_fallback(config, BBOX_CODERS, BBOX_CODERS)


__all__ = ["NMSFreeCoder", 'build_bbox_coder']
