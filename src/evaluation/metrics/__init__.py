#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from .nuscenes_metric import NuScenesMetric
from mmengine.registry import METRICS

METRICS.register_module(module=NuScenesMetric)
