from .match_cost import BBox3DL1Cost, SmoothL1Cost
from src.registry import MATCH_COST
from src.typing import ConfigType
from src.bevformer.builder import _build_with_fallback


def build_match_cost(config: ConfigType):
    """Build a match cost based on config.
    Args:
        config: Config to build.
    """
    return _build_with_fallback(config, MATCH_COST, MATCH_COST)


__all__ = ['BBox3DL1Cost', 'SmoothL1Cost', 'build_match_cost']
