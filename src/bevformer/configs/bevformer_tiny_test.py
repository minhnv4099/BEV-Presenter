#
#  Copyright (c) 2026
#  Minh NGUYEN <vnguyen9@lakeheadu.ca>
#
# BEvFormer-tiny consumes at lease 6700M GPU memory
# compared to bevformer_base, bevformer_tiny has
# smaller backbone: R101-DCN -> R50
# smaller BEV: 200*200 -> 50*50
# less encoder layers: 6 -> 3
# smaller input size: 1600*900 -> 800*450
# multi-scale feautres -> single scale features (C5)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True
)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 50
bev_w_ = 50
queue_length = 3


encoder = dict(
    type='BEVFormerEncoder',
    num_layers=2,
    pc_range=point_cloud_range,
    num_points_in_pillar=8,
    return_intermediate=False,
    transformerlayers=dict(
        type='BEVFormerLayer',
        attn_cfgs=[
            dict(
                type='TemporalSelfAttention',
                embed_dims=_dim_,
                num_levels=1
            ),
            dict(
                type='SpatialCrossAttention',
                pc_range=point_cloud_range,
                deformable_attention=dict(
                    type='MSDeformableAttention3D',
                    embed_dims=_dim_,
                    num_points=8,
                    num_levels=_num_levels_),
                embed_dims=_dim_,
            )
        ],
        ffn_cfgs=dict(
            type="FFN",
            feedforward_channels=_ffn_dim_,
            ffn_drop=0.1,
            num_fcs=2,
        ),
        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                         'ffn', 'norm')))


decoder = dict(
    type='DetectionTransformerDecoder',
    num_layers=2,
    return_intermediate=True,
    transformerlayers=dict(
        type='DetrTransformerDecoderLayer',
        attn_cfgs=[
            dict(
                type='CustomMSDeformableAttention',
                embed_dims=_dim_,
                num_levels=1,
            ),
            dict(
                type='CustomMSDeformableAttention',
                embed_dims=_dim_,
                num_levels=1,
            )
        ],
        ffn_cfgs=dict(
            type="FFN",
            feedforward_channels=_ffn_dim_,
            ffn_drop=0.1,
            num_fcs=2,
        ),
        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                         'ffn', 'norm')))

transformer = dict(
    type='PerceptionTransformer',
    rotate_prev_bev=True,
    use_shift=True,
    use_can_bus=True,
    embed_dims=_dim_,
    encoder=encoder,
    decoder=decoder
)

pts_bbox_head = dict(
    type='BEVFormerHead',
    bev_h=bev_h_,
    bev_w=bev_w_,
    num_query=900,
    num_classes=10,
    in_channels=_dim_,
    sync_cls_avg_factor=True,
    with_box_refine=False,
    as_two_stage=False,
    transformer=transformer,
    bbox_coder=dict(
        type='NMSFreeCoder',
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=point_cloud_range,
        max_num=300,
        voxel_size=voxel_size,
        num_classes=10),
    positional_encoding=dict(
        type='LearnedPositionalEncoding',
        num_feats=_pos_dim_,
        row_num_embed=bev_h_,
        col_num_embed=bev_w_,
    ),
    loss_cls=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=2.0),
    loss_bbox=dict(type='L1Loss', loss_weight=0.25),
    loss_iou=dict(type='GIoULoss', loss_weight=0.0),
    train_cfg=None
)

model = dict(
    type='BEVFormerDetector',
    use_grid_mask=True,
    video_test_mode=True,
    pretrained=dict(img='torchvision://resnet50'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=pts_bbox_head,
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='SmoothL1Cost', weight=0.0),  # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range))))

dataset_type = 'CustomNuScenesDataset'
version = "v1.0-mini"
data_root = f'data/nuscenes/{version}/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),  # loading.py
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),  # loading.py
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),  # transform_3d.py
    # dict(type='ObjectNameFilter', classes=class_names),  # transform_3d.py
    dict(type='PhotoMetricDistortionMultiViewImage'),  # transform_3d.py
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),  # transform_3d.py
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),   # transform_3d.py
    dict(type='PadMultiViewImage', size_divisor=32),  # transform_3d.py
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names),  # formating.py
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img']),   # transform_3d.py
    dict(type='TypeConverter')  # formating.py
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        classes=class_names,
        modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)
