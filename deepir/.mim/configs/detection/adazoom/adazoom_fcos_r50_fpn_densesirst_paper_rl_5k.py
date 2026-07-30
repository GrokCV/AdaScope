import os.path as osp

_base_ = [
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'deepir.datasets',
        'deepir.models.detectors.adazoom_fcos',
    ],
    allow_failed_imports=False,
)

REPO_ROOT = '/root/data-tmp/BAFE-Net'
DATA_ROOT = osp.join(REPO_ROOT, 'data', 'DenseSIRST', 'SIRSTdevkit')
FCOS_CKPT = '/root/data-tmp/BAFE-Net/cluster_visual/checkpoint/fcos_best.pth'
dataset_type = 'deepir.SIRSTVOCDetSegDataset'
backend_args = None
device_cpn = 'cuda:0'

load_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_seg=True,
        imdecode_backend='pillow'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='Pad', size=(512, 512)),
    dict(prob=0.5, type='RandomFlip'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_seg=True,
        imdecode_backend='pillow'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_pipeline = [
    dict(paste_by_box=True, selected=True, type='deepir.SkyCopyPaste'),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='RepeatDataset',
            times=3,
            dataset=dict(
                type=dataset_type,
                data_root=DATA_ROOT,
                ann_file='Splits/trainval_v2.txt',
                data_prefix=dict(sub_data_root=''),
                filter_cfg=dict(filter_empty_gt=False, min_size=0, bbox_min_size=0),
                pipeline=load_pipeline,
                backend_args=backend_args,
            ),
        ),
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=DATA_ROOT,
        ann_file='Splits/test_v2.txt',
        data_prefix=dict(sub_data_root=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='VOCMetric',
    iou_thrs=0.5,
    metric='mAP',
    eval_mode='11points',
    prefix='merged',
)
test_evaluator = val_evaluator

INF = 100000000.0
base_lr = 2e-5

model = dict(
    type='AdaZoomFCOS',
    pretrained_detector_ckpt=FCOS_CKPT,
    episode_len=7,
    gamma=0.5,
    patch_short_edge=800,
    candidate_scales=(240.0**2, 350.0**2, 420.0**2),
    desired_object_scale_ranges=((0.0, 40.0**2), (30.0**2, 60.0**2), (50.0**2, 1e12)),
    candidate_ratios=(0.7, 1.0, 1.5),
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32,
        pad_seg=True,
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        num_outs=3,
        relu_before_extra_convs=True,
    ),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        regress_ranges=((-1, 64), (1, 128), (1, 256)),
        strides=[8, 16, 32],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=0.5,
        ),
    ),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
    ),
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='Adam',
        lr=base_lr,
        weight_decay=0.0,
    ),
)

param_scheduler = []

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=5000,
    val_interval=500,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=500, save_best='auto', by_epoch=False),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'),
)

log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)

work_dir = '/root/data-tmp/BAFE-Net/work_dirs/adazoom_fcos_r50_fpn_densesirst_paper_rl_5k'
