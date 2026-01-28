#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from typing import Union, Optional, TypeAlias, List
from mmengine.config import ConfigDict


ConfigType: TypeAlias = Optional[Union[ConfigDict, dict, list[ConfigDict], list[dict]]]
# ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
# Type hint of one or more config data
MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]