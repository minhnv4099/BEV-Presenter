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

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 50
bev_w_ = 50
queue_length = 3


encoder = dict(
    type='BEVFormerEncoder',
    num_layers=3,
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
            # config=None,
            feedforward_channels=_ffn_dim_,
            ffn_drop=0.1,
            num_fcs=2,
        ),
        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                         'ffn', 'norm')))


decoder = dict(
    type='DetectionTransformerDecoder',
    num_layers=6,
    return_intermediate=True,
    transformerlayers=dict(
        type='DetrTransformerDecoderLayer',
        attn_cfgs=[
            dict(
                type='BaseAttention',
                embed_dims=_dim_,
                num_heads=8,
                dropout=0.1),
            dict(
                type='CustomMSDeformableAttention',
                embed_dims=_dim_,
                num_levels=1
            )
        ],
        ffn_cfgs=dict(
            type="FFN",
            # config=None,
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
        # depth=50,
        # num_stages=4,
        # out_indices=(3,),
        # frozen_stages=1,
        # norm_cfg=dict(type='BN', requires_grad=False),
        # norm_eval=True,
        # style='pytorch'
    ),
    img_neck=dict(
        type='BaseDownChannel',
        in_channels=(512, 1024, 2048),
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True
    ),
    pts_bbox_head=pts_bbox_head,
    train_cfg=dict(
            pts=dict(
                grid_size=[512, 512, 1],
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range,
                out_size_factor=4,
                assigner=dict(
                    type='HungarianAssigner3D',
                    cls_cost=dict(type='BBox3DL1Cost', weight=2.0),
                    reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                    iou_cost=dict(type='SmoothL1Cost', weight=0.0),  # Fake cost. This is just to make it compatible with DETR head.
                    pc_range=point_cloud_range)))
)
