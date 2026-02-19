from typing import Callable, List, Optional, Sequence, Union, TYPE_CHECKING
from src.bevformer.builder import build_compose_component
from src.utils.logging import getLogger
if TYPE_CHECKING:
    from src.datasets.pipelines import BaseTransform

logger = getLogger(__name__)


class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable], optional): Sequence of transform
            object or config dict to be composed.
    """

    def __init__(self,
                 transforms: Optional[Sequence[Union[dict, Callable]]],
                 debug: bool = False):
        self.transforms: List[BaseTransform] = []
        self.debug = debug

        if transforms is None:
            transforms = []

        for transform in transforms:
            # `Compose` can be built with config dict with type and
            # corresponding arguments.
            if isinstance(transform, dict):
                transform: BaseTransform = build_compose_component(transform)
                if not callable(transform):
                    raise TypeError(f'transform should be a callable object, but got {type(transform)}')
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(
                    f'transform must be a callable object or dict, '
                    f'but got {type(transform)}')

    def __call__(self, data: dict) -> Union[dict, list[dict], None]:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for i, t in enumerate(self.transforms):
            if self.debug:
                # logger.info(f"Step {i}. {t.__class__.__name__}")
                ...
            data = t(data, self.debug)
            # The transform will return None when it failed to load images or
            # cannot find suitable augmentation parameters to augment the data.
            # Here we simply return None if the transform returns None and the
            # dataset will handle it by randomly selecting another data sample.
            if data is None:
                return None

        # only need debug one time
        self.debug = False
        return data

    def __repr__(self):
        """Print ``self.transforms`` in sequence.

        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
