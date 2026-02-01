#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
import traceback
import warnings
from typing import Optional, Any

from src.typing import ConfigType
from src.utils.logging import getLogger

from src.registry import (
    Registry,
    MODELS,
    BACKBONES,
    NECKS,
    HEADS,
    CONFIGS,
    POSITION_ENCODINGS,
    ASSIGNERS,
    BBOX_CODERS,
    MATCH_COST,
    LOSSES,
    DETECTORS
)

logger = getLogger(__name__)


def _build_with_fallback(
    cfg: ConfigType,
    main_registry: Registry,
    fallback_registry: Optional[Registry] = None,
    default_args: Optional[dict] = None,
    main_first: bool = False
):
    """Build a registered object from config. Use `main registry` (typically most-parent registry) first.
    If any error occur, use the `fallback registry`.

    Args:
        cfg (`dict` or `ConfigDict`): Config.
        main_registry (Registry): Main registry to build pre-defined object (e.g. `MODELS`).
        fallback_registry (Registry): Fallback registry, usually particular (e.g. `BACKBONES`, `NECKS`, ...):
        default_args (dict): Arguments to override.
    """
    def _build_by_main():
        try:
            print(main_registry.get("BEVFormer"))
            print(main_registry)

            _obj = main_registry.build(cfg=cfg, default_args=default_args)
            warnings.warn(f"Built object of {cfg['type']!r} by main registry {main_registry.name!r}.")
            return _obj
        except KeyError as e:
            traceback.print_exc(5)
            logger.info(f"Failed build object of {cfg['type']!r} by main registry {main_registry.name!r}.")

        return None

    def _build_by_fallback():
        try:
            _obj = fallback_registry.build(cfg, default_args=default_args)
            warnings.warn(f"Built object of {cfg['type']!r} by fallback registry {fallback_registry.name!r}.")
            return _obj
        except Exception as e:
            traceback.print_last()
            logger.info(f"Failed build object of {cfg['type']!r} by fallback registry {fallback_registry.name!r}.")
        return None

    if main_first:
        obj = _build_by_main() or _build_by_fallback()
    else:
        obj = _build_by_fallback() or _build_by_main()

    if obj is None:
        raise ValueError("Cannot build.")
    return obj


def build_config(cfg: ConfigType, default_args=None):
    """Builder config."""
    return _build_with_fallback(cfg, CONFIGS, None, default_args)


def build_backbone(config: ConfigType, default_args: Optional[ConfigType] = None):
    """Build backbone."""
    return _build_with_fallback(config, MODELS, BACKBONES, default_args)


def build_neck(config: ConfigType, default_args: Optional[ConfigType] = None):
    """Build neck."""
    return _build_with_fallback(config, MODELS, NECKS, default_args)


def build_head(cfg: ConfigType, default_args=None):
    """Build head."""
    return _build_with_fallback(cfg, MODELS, HEADS, default_args)


def build_detector(cfg: ConfigType, default_args=None):
    """Build detector."""
    return _build_with_fallback(cfg, MODELS, DETECTORS, default_args)


def build_loss(cfg: ConfigType, default_args=None):
    """Builder config."""
    return _build_with_fallback(cfg, MODELS, LOSSES, default_args)


def build_positional_encoding(cfg: ConfigType, default_args=None):
    """Builder for Position Encoding."""
    return _build_with_fallback(cfg, MODELS, POSITION_ENCODINGS, default_args)