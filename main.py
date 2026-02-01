#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
import torch
import inspect
from src.registry import MODELS
from src.bevformer.builder import (
    build_config,
    build_head,
    build_detector,
)
from src.bevformer.models.utils.bricks import build_transformer_block
from src.bevformer.configs.bevformer_tiny_test import encoder as encoder_cfg
from src.bevformer.configs.bevformer_tiny_test import model as detector_cfg


def debug_reference_points():
    # bev_h, bev_w, height, bs, num_points = 5, 5, 2, 1, 4
    # img_metas = [
    #     {"lidar2img":  torch.rand(6, 4, 4)}
    # ] * bs
    #
    # ref_points_3d = BEVFormerEncoder.get_reference_points(bev_h, bev_w, height, bs, dim='2d', num_points_in_pillar=num_points)
    # ref_points_3d = BEVFormerEncoder.get_reference_points(bev_h, bev_w, height, bs, dim='3d', num_points_in_pillar=num_points)
    #
    # BEVFormerEncoder.point_sampling(ref_points_3d, pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], img_metas=img_metas)
    pass


def debug_flow():
    bev_h, bev_w, height, bs, dim = 50, 50, 8, 1, 256
    num_cam, num_levels, num_query = 6, 4, bev_h*bev_w
    image_shape = 320, 320

    level_size_1 = 32
    level_size_2 = 16
    level_size_3 = 8
    level_size_4 = 4

    level_shape_1 = [level_size_1, level_size_1]
    level_shape_2 = [level_size_2, level_size_2]
    level_shape_3 = [level_size_3, level_size_3]
    level_shape_4 = [level_size_4, level_size_4]

    num_value = level_size_1**2 + level_size_2**2 + level_size_3**2 + level_size_4**2

    img_metas = [
        {
            "lidar2img":  torch.rand(num_cam, 4, 4),
            "img_shape": [
                image_shape * num_cam
            ]
        },

    ] * bs

    bev_query = torch.rand(num_query, bs, dim)
    prev_bev = torch.rand(num_query, bs, dim)

    key = torch.rand(num_cam, num_value, bs, dim)
    value = torch.rand(num_cam, num_value, bs, dim)

    spatial_shapes = torch.tensor([level_shape_1, level_shape_2, level_shape_3, level_shape_4])

    encoder = build_transformer_block(encoder_cfg)

    output = encoder(
        bev_query,
        key,
        value,
        bev_h=bev_h,
        bev_w=bev_w,
        prev_bev=None,
        spatial_shapes=spatial_shapes,
        img_metas=img_metas,
    )

    print(output.shape)

    # decoder = build_transformer_block(decoder_cfg)


def main2():
    model = build_detector(detector_cfg)
    print(model)


if __name__ == "__main__":
    debug_flow()
