import time
from collections import defaultdict
import torch
from src.typing import ConfigType
from src.utils.logging import getLogger
from mmcv.cnn.bricks.norm import build_norm_layer
from src.bevformer.builder import _build_with_fallback
from src.registry import (
    MODELS,
    ATTENTIONS,
    TRANSFORMER_LAYERS,
    TRANSFORMER_BLOCKS,
    TRANSFORMERS,
)
from contextlib import suppress

logger = getLogger(name="BricksBuilder")

time_maps = defaultdict(lambda: 0.)
count_maps = defaultdict(lambda: 0.)


def run_time(name):
    def middle(fn):
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.time()
            res = fn(*args, **kwargs)
            torch.cuda.synchronize()
            time_maps['%s : %s' % (name, fn.__name__)] += time.time() - start
            count_maps['%s : %s' % (name, fn.__name__)] += 1
            print("%s : %s takes up %f " % (name, fn.__name__, time_maps['%s : %s' % (name, fn.__name__)] / count_maps[
                '%s : %s' % (name, fn.__name__)]))
            return res

        return wrapper

    return middle


def build_feedforward_network(cfg: ConfigType, default_args=None):
    """Builder for feed-forward network (FFN)."""
    return _build_with_fallback(cfg, MODELS, MODELS, default_args, main_first=False)


def build_attention(cfg: ConfigType, default_args=None):
    """Builder for attention."""
    return _build_with_fallback(cfg, MODELS, ATTENTIONS, default_args, main_first=False)


def build_transformer_layer(cfg: ConfigType, default_args=None):
    """Builder for transformer layer."""
    return _build_with_fallback(cfg, MODELS, TRANSFORMER_LAYERS, default_args, main_first=False)


def build_transformer_block(cfg: ConfigType, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    return _build_with_fallback(cfg, MODELS, TRANSFORMER_BLOCKS, default_args, main_first=False)


def build_transformer(cfg: ConfigType, default_args=None):
    """Builder for transformer."""
    return _build_with_fallback(cfg, MODELS, TRANSFORMERS, default_args, main_first=False)
