from .hungarian_assigner_3d import HungarianAssigner3D
from .hungarian_assigner import HungarianAssigner
from .base import BaseAssigner
from src.registry import ASSIGNERS
from src.typing import ConfigType
from src.bevformer.builder import _build_with_fallback


def build_assigner(config: ConfigType) -> BaseAssigner:
    """Build assigner."""
    return _build_with_fallback(config, ASSIGNERS, ASSIGNERS)


__all__ = ['HungarianAssigner3D', 'HungarianAssigner', 'build_assigner']
