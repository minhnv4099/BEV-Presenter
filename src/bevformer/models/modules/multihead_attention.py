#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from typing import Optional
import math
import torch
import torch.nn as nn
from torch import Tensor

from mmengine.model import BaseModule, xavier_init
from src.registry import ATTENTIONS
from src.utils.logging import getLogger
from src.bevformer.transformers.attentions import eager_attention_forward

logger = getLogger(__name__)


@ATTENTIONS.register_module()
class MultiheadAttention(BaseModule):

    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 6,
                 dropout: Optional[float] = None,
                 init_cfg: Optional[dict] = None,
                 batch_first: bool = True,
                 ):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(
                "embed_dims should divide by num_head, but got"
                f"embed_dims={embed_dims}, "
                f"num_heads{num_heads}, "
                f"embed_dims/num_heads={embed_dims / num_heads:.3f}"
            )

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = int(embed_dims // num_heads)
        self.all_head_size = int(self.head_dims * self.num_heads)

        self.q_proj = nn.Linear(in_features=embed_dims, out_features=embed_dims)
        self.k_proj = nn.Linear(in_features=embed_dims, out_features=embed_dims)
        self.v_proj = nn.Linear(in_features=embed_dims, out_features=embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        self.init_weights()

    def init_weights(self):
        xavier_init(self.q_proj, distribution='uniform', bias=0.)
        xavier_init(self.k_proj, distribution='uniform', bias=0.)
        xavier_init(self.v_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        identity: Optional[Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
    ):
        if key is None:
            key = query
        if value is None:
            value = key
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        if identity is None:
            identity = query

        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        bs, n_query, hidden_size = query.shape
        view_shape = (bs, -1, self.num_heads, self.head_dims)

        query: torch.Tensor = self.q_proj(query).view(*view_shape).transpose(1, 2)
        key: torch.Tensor = self.k_proj(key).view(*view_shape).transpose(1, 2)
        value: torch.Tensor = self.v_proj(value).view(*view_shape).transpose(1, 2)

        context, _ = eager_attention_forward(
            module=self,
            query=query,
            key=key,
            value=value,
            attention_mask=attn_mask,
            head_mask=head_mask,
            scaling=math.sqrt(hidden_size),
            dropout=0.0
        )
        context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(context_shape)

        output = self.output_proj(context)
        output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
