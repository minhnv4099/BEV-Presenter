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
_dim_ = 256
_ffn_dim_ = _dim_*2
_num_levels_ = 1

model = dict(
    type='BEVFormerEncoder',
    num_layers=3,
    pc_range=point_cloud_range,
    num_points_in_pillar=8,
    return_intermediate=False,
    transformerlayers_cfg=dict(
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
                         'ffn', 'norm'))),
