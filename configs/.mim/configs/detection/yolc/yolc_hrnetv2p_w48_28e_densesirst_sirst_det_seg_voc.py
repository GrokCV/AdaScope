import os.path as osp

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/sirst_det_seg_voc.py',
]

custom_imports = dict(
    imports=[
        'deepir.models.losses.yolc_gwd_loss',
        'deepir.models.dense_heads.yolc_head',
        'deepir.models.detectors.yolc',
    ],
    allow_failed_imports=False,
)

REPO_ROOT = '/root/data-tmp/BAFE-Net'
DATA_MEAN = [101.0, 101.0, 101.0]
DATA_STD = [38.5, 38.5, 38.5]

model = dict(
    type='YOLC',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=DATA_MEAN,
        std=DATA_STD,
        bgr_to_rgb=False,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://msra/hrnetv2_w48',
        ),
    ),
    neck=dict(
        type='HRFPN',
        in_channels=[48, 96, 192, 384],
        out_channels=384,
        num_outs=1,
    ),
    bbox_head=dict(
        type='YOLCHead',
        num_classes=1,
        in_channels=384,
        feat_channels=96,
        loss_center_local=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_xywh=dict(type='YOLCGWDLoss', loss_weight=2.0),
    ),
    train_cfg=None,
    test_cfg=dict(
        topk=1500,
        local_maximum_kernel=3,
        max_per_img=500,
        nms_cfg=dict(type='nms', iou_threshold=0.5),
    ),
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35.0, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=28,
        by_epoch=True,
        milestones=[20, 26],
        gamma=0.1,
    ),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=28, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='pascal_voc/mAP',
        rule='greater',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

work_dir = osp.join(
    REPO_ROOT,
    'work_dirs',
    'yolc_hrnetv2p_w48_28e_densesirst_sirst_det_seg_voc',
)
