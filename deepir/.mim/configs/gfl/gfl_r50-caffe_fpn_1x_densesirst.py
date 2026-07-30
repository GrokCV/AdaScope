_base_ = [
    '../detection/_base_/datasets/sirst_det_seg_voc.py',
    '../detection/_base_/schedules/schedule_1x.py',
    '../detection/_base_/default_runtime.py',
]

# DenseSIRST objects are much smaller and weaker than visible-light COCO
# objects after the fixed 512x512 resize used by the dataset base config.
# Vanilla GFL works well on visible imagery because its anchor scale / FPN
# hierarchy / assignment prior are tuned for richer textures and larger boxes.
# On DenseSIRST, those priors either diverge numerically or converge to dense
# clutter responses with near-zero IoU. This variant only changes model-side
# parameters so GFL better matches infrared tiny targets.
model = dict(
    type='GFL',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        end_level=0,
        add_extra_convs='on_output',
        num_outs=1,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='GFLHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=2,
        feat_channels=128,
        init_cfg=dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
            override=dict(
                type='Normal',
                name='gfl_cls',
                std=0.01,
                bias_prob=0.003)),
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[0.75],
            center_offset=0.5,
            strides=[4]),
        reg_max=4,
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=4.0,
            loss_weight=0.1),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=3),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
