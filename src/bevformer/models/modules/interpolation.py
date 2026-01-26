#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
from typing import Optional

import torch
import torch.nn as nn


class InterpolateInitialPositionEmbeddings(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def forward(self, pos_embed: torch.Tensor, img_size: Optional[tuple[int]] = (800, 1344)) -> torch.Tensor:
        """Forward of [`InterpolateInitialPositionEmbeddings`]. It interpolates the initial embeddings (i.e., after getting patch embeddings.
        It do to change to num_patches inferred from ``image_size`.

        Args:
            pos_embed (`torch.Tensor`):
                Position embeddings to interpolate of shape `(1, seq_len, hidden_dim)`.
            img_size (`tuple[int]`, *optional*, default to `(800, 1344)`):
                Desired size of image after interpolation. Typically size of pixel values.
        Returns:
            Position embeddings of shape `(1, new_seq_len, hidden_dim)`.
            Used to plus to embeddings to indicate positions.
        """
        batch_size, seq_len, hidden_size = pos_embed.shape
        num_patch_height, num_patch_width = (
            self.config.image_size[0] // self.config.patch_size,
            self.config.image_size[1] // self.config.patch_size,
        )
        height, width = img_size
        new_num_patch_height, new_num_patch_width = height // self.config.patch_size, width // self.config.patch_size

        num_patches = num_patch_height * num_patch_width
        new_num_patches = new_num_patch_height * new_num_patch_width
        if num_patches == new_num_patches:
            return pos_embed

        patch_pos_embed = pos_embed.transpose(1, 2)
        patch_pos_embed = patch_pos_embed.view(batch_size, hidden_size, num_patch_height, num_patch_width)
        patch_pos_embed = nn.functional.interpolate(
            input=patch_pos_embed,
            size=(new_num_patch_height, new_num_patch_width),
            mode="bicubic",
            align_corners=False
        )
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)

        return patch_pos_embed


class InterpolateMidPositionEmbeddings(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def forward(self, pos_embed: torch.Tensor, img_size=(800, 1344)) -> torch.Tensor:
        """Forward of `[InterpolateMidPositionEmbeddings]`. It interpolates the middle embeddings (i.e., hidden states after each encoder layer).
        It does to change to num_patches inferred from ``image_size`.

        Args:
            pos_embed (`torch.Tensor`):
                Position embeddings to interpolate of shap `(num_layer - 1, 1 [bs], seq_len, hidden_dim)`.
            img_size (`tuple[int]`, *optional*, default to `(800, 1344)`):
                Desired size of image.
        Returns:
            Position embeddings of shape `(num_layer, 1, new_seq_len, hidden_dim)`.
            Used to plus to embeddings to indicate positions.
        """
        num_layer, batch_size, seq_len, hidden_size = pos_embed.shape
        num_patch_height, num_patch_width = (
            self.config.image_size[0] // self.config.patch_size,
            self.config.image_size[1] // self.config.patch_size,
        )
        height, width = img_size
        new_num_patch_height, new_num_patch_width = height // self.config.patch_size, width // self.config.patch_size

        num_patches = num_patch_height * num_patch_width
        new_num_patches = new_num_patch_height * new_num_patch_width

        if num_patches == new_num_patches:
            return pos_embed

        patch_pos_embed = pos_embed.transpose(2, 3)
        patch_pos_embed = patch_pos_embed.view(num_layer * batch_size, hidden_size, num_patch_height, num_patch_width)
        patch_pos_embed = nn.functional.interpolate(
            input=patch_pos_embed,
            size=(new_num_patch_height, new_num_patch_width),
            mode="bicubic",
            align_corners=False
        )
        patch_pos_embed = (
            patch_pos_embed.flatten(2)
            .transpose(1, 2)
            .contiguous()
            .view(num_layer, batch_size, new_num_patch_height * new_num_patch_width, hidden_size)
        )

        return patch_pos_embed
