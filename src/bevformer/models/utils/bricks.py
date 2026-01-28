import functools
import time
import warnings
from collections import defaultdict
import torch
from typing import Optional, Any
from src.typing import ConfigType
from src.utils.logging import getLogger
from mmcv.cnn.bricks.norm import build_norm_layer
from src.registry import (
    Registry,
    MODELS,
    CONFIGS,
    ATTENTIONS,
    TRANSFORMER_LAYERS,
    TRANSFORMER_BLOCKS,
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


def _build_with_fallback(
    cfg: ConfigType,
    main_registry: Registry,
    fallback_registry: Optional[Registry] = None,
    default_args: Optional[Any] = None
):
    obj = None
    try:
        obj = main_registry.build(cfg=cfg, default_args=default_args)
        if obj is None:
            logger.info(
                f"Failed build object of {cfg['type']!r} by the main registry {main_registry.name!r}. "
                f"Try build from fallback registry."
            )
            raise ValueError
    except Exception as e:
        # warnings.warn(str(e))
        if fallback_registry is None:
            raise ValueError("Fallback register is not set. Cannot build.")

        try:
            warnings.warn(f"Building {cfg['type']!r} by fallback registry {fallback_registry.name!r}.")
            obj = fallback_registry.build(cfg, default_args=default_args)
        except Exception as e:
            raise ValueError(e)

    if obj is None:
        # logger.info(f"Failed build from {cfg['type']!r} by fallback registry {fallback_registry.__class__.__name__}.")
        raise ValueError(f"Failed build from {cfg['type']!r} by main and fallback registry {fallback_registry.name}.")

    return obj


def build_positional_encoding(cfg: ConfigType, default_args=None):
    """Builder for Position Encoding."""
    return _build_with_fallback(cfg, MODELS, MODELS, default_args)


def build_feedforward_network(cfg: ConfigType, default_args=None):
    """Builder for feed-forward network (FFN)."""
    return _build_with_fallback(cfg, MODELS, MODELS, default_args)


def build_attention(cfg: ConfigType, default_args=None):
    """Builder for attention."""
    return _build_with_fallback(cfg, MODELS, ATTENTIONS, default_args)


def build_transformer_layer(cfg: ConfigType, default_args=None):
    """Builder for transformer layer."""
    return _build_with_fallback(cfg, MODELS, TRANSFORMER_LAYERS, default_args)


def build_transformer_block(cfg: ConfigType, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    return _build_with_fallback(cfg, MODELS, TRANSFORMER_BLOCKS, default_args)


def build_config(cfg: ConfigType, default_args=None):
    return _build_with_fallback(cfg, CONFIGS, None, default_args)
