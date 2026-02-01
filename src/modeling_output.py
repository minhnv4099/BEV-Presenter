#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from torch import Tensor
from typing import Optional, Union
from typing_extensions import override
from dataclasses import dataclass, is_dataclass
from .dataclass_as_dict import DataclassAsDict


class ModelOutput(DataclassAsDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Subclasses of ModelOutput must use the @dataclass decorator
        # This check is done in __init__ because the @dataclass decorator operates after __init_subclass__
        # issubclass() would return True for issubclass(ModelOutput, ModelOutput) when False is needed
        # Just need to check that the current class is not ModelOutput
        is_modeloutput_subclass = self.__class__ != ModelOutput

        if is_modeloutput_subclass and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} is not a dataclass."
                " This is a subclass of ModelOutput and so must use the @dataclass decorator."
            )

    @override
    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    @override
    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    @override
    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    @override
    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")


@dataclass
class BaseModelOutput(ModelOutput):
    last_hidden_state: Optional[Tensor] = None


@dataclass
class BaseModelOutputWithNoAttention(ModelOutput):
    """"""
    last_hidden_state: Optional[Tensor] = None
    hidden_states: Optional[tuple[Tensor]] = None


@dataclass
class BaseModelOutputWithPoolingAndNoAttention(ModelOutput):
    """"""
    last_hidden_state: Optional[Tensor] = None
    pooler_output: Optional[Tensor] = None
    hidden_states: Optional[tuple[Tensor]] = None


@dataclass
class ImageClassifierOutputWithNoAttention(ModelOutput):
    """"""
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None
    hidden_states: Optional[tuple[Tensor]] = None


@dataclass
class BackboneOutput(ModelOutput):
    """"""
    feature_maps: Optional[tuple[Tensor]] = None
    hidden_states: Optional[tuple[Tensor]] = None
    attentions: Optional[Tensor] = None


@dataclass
class BEVFormerEncoderOutput(ModelOutput):
    bev_feat: Optional[Tensor] = None
