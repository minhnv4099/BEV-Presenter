#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
import glob
import numpy as np
import torch
from mmengine.config import ConfigDict
from src.bevformer.builder import (
    build_config,
    build_head,
    build_detector,
    build_dataset
)
from src.utils.fileio import dump
from torch.utils.data import DataLoader
from src.bevformer.models.utils.bricks import build_transformer_block
from src.bevformer.configs.bevformer_tiny_test import encoder as encoder_cfg
from src.bevformer.configs.bevformer_tiny_test import model as model_cfg
from src.bevformer.configs.bevformer_tiny_test import data as data_cfg


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


def create_dummy_img_metas(bs=2, num_cam=6, H=224, W=224):
    img_metas = []

    for b in range(bs):
        meta = {}

        # 1. lidar2img: (num_cam, 4, 4)
        lidar2img = []
        for _ in range(num_cam):
            mat = np.eye(4, dtype=np.float32)
            lidar2img.append(mat)
        meta['lidar2img'] = np.stack(lidar2img, axis=0).tolist()

        # 2. img_shape: list per camera
        meta['img_shape'] = [(H, W, 3) for _ in range(num_cam)]
        meta['ori_shape'] = [(H, W, 3) for _ in range(num_cam)]
        meta['pad_shape'] = [(H, W, 3) for _ in range(num_cam)]

        # 3. pc_range (BEVFormer default nuScenes)
        meta['pc_range'] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

        # 4. scene_token
        meta['scene_token'] = f"scene_dummy_{b}"

        # 5. prev_bev_exists (cho temporal)
        meta['prev_bev_exists'] = True

        # 6. can_bus (ego pose: x, y, z, roll, pitch, yaw,...)
        # BEVFormer chỉ quan tâm [:3] và [-1]
        can_bus = np.zeros(18, dtype=np.float32).tolist()
        can_bus[:3] = np.array([0.0, 0.0, 0.0])  # position
        can_bus[-1] = 0.0  # yaw angle
        meta['can_bus'] = can_bus

        img_metas.append(meta)

    return img_metas


def main():
    from src.datasets.builder import build_dataloader

    data_config = ConfigDict(data_cfg)
    detector_cfg = ConfigDict(model_cfg)

    dataset = build_dataset(data_config.train)
    detector = build_detector(detector_cfg)

    sample = dataset[10]
    outs = detector(**sample)
    print(outs)


if __name__ == "__main__":
    main()
