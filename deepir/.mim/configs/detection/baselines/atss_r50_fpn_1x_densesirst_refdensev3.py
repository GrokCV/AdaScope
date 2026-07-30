_base_ = ["./atss_r50_fpn_1x_densesirst_refdense.py"]

model = dict(
    backbone=dict(
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=False),
        style="pytorch",
    ),
    neck=dict(
        start_level=0,
        end_level=-1,
        add_extra_convs="on_output",
        num_outs=5,
    ),
    bbox_head=dict(
        stacked_convs=2,
        feat_channels=128,
        anchor_generator=dict(
            type="AnchorGenerator",
            ratios=[1.0],
            octave_base_scale=1,
            scales_per_octave=4,
            strides=[4, 8, 16, 32, 64],
        ),
    ),
    train_cfg=dict(assigner=dict(type="ATSSAssigner", topk=9), allowed_border=-1, pos_weight=-1, debug=False),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type="nms", iou_threshold=0.6),
        max_per_img=300,
    ),
)

optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0001))
