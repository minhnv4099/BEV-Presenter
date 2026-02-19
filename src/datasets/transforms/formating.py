#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from typing import Union, Sequence, Dict, Optional, Tuple, List
import torch
import numpy as np
from src.registry import PIPELINES
from src.structures.data_container import DataContainer as DC
from mmcv.transforms.formatting import to_tensor
from .base_transform import BaseTransform


@PIPELINES.register_module()
class CustomDefaultFormatBundle3D(BaseTransform):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names: list[str]):
        self.class_names = class_names

    def transform(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        # results = super(CustomDefaultFormatBundle3D, self).__call__(results, )
        # results['gt_map_masks'] = DC(
        #     to_tensor(results['gt_map_masks']), stack=True)

        return results


@PIPELINES.register_module()
class TypeConverter(BaseTransform):

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        results['img'] = np.array(results['img'], dtype=np.float32)
        results['gt_labels_3d'] = torch.as_tensor(results['gt_labels_3d'], dtype=torch.long)

        return results
