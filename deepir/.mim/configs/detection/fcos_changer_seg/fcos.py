INF = 100000000.0
auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
base_lr = 1.0
data_root = 'data/DenseSIRST/SIRSTdevkit/'
dataset_type = 'deepir.SIRSTVOCDetSegDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, save_best='auto', type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
device_cpn = 'cuda:0'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
load_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        imdecode_backend='pillow',
        type='LoadAnnotations',
        with_bbox=True,
        with_seg=True),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(size=(
        512,
        512,
    ), type='Pad'),
    dict(prob=0.5, type='RandomFlip'),
]
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=-1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='caffe',
        type='ResNet'),
    bbox_head=dict(
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='IoULoss'),
        loss_centerness=dict(
            loss_weight=0.5, type='CrossEntropyLoss', use_sigmoid=True),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=1,
        regress_ranges=(
            (
                -1,
                64,
            ),
            (
                1,
                128,
            ),
            (
                1,
                256,
            ),
        ),
        stacked_convs=4,
        strides=[
            8,
            16,
            32,
        ],
        type='FCOSHead'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            102.9801,
            115.9465,
            122.7717,
        ],
        pad_seg=True,
        pad_size_divisor=32,
        std=[
            1.0,
            1.0,
            1.0,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=3,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=1,
        type='FPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.5, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    type='FCOS')
optim_wrapper = dict(
    optimizer=dict(
        decouple=True, lr=1.0, type='DAdaptAdam', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        factor=0.3333333333333333,
        type='ConstantLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=20,
        gamma=0.1,
        milestones=[
            8,
            16,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='Splits/test_v2.txt',
        backend_args=None,
        data_prefix=dict(sub_data_root=''),
        data_root='data/DenseSIRST/SIRSTdevkit/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(
                imdecode_backend='pillow',
                type='LoadAnnotations',
                with_bbox=True,
                with_seg=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='deepir.SIRSTVOCDetSegDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(eval_mode='11points', metric='mAP', type='VOCMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(
        imdecode_backend='pillow',
        type='LoadAnnotations',
        with_bbox=True,
        with_seg=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=20, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        dataset=dict(
            dataset=dict(
                ann_file='Splits/trainval_v2.txt',
                backend_args=None,
                data_prefix=dict(sub_data_root=''),
                data_root='data/DenseSIRST/SIRSTdevkit/',
                filter_cfg=dict(
                    bbox_min_size=0, filter_empty_gt=False, min_size=0),
                pipeline=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(
                        imdecode_backend='pillow',
                        type='LoadAnnotations',
                        with_bbox=True,
                        with_seg=True),
                    dict(keep_ratio=False, scale=(
                        512,
                        512,
                    ), type='Resize'),
                    dict(size=(
                        512,
                        512,
                    ), type='Pad'),
                    dict(prob=0.5, type='RandomFlip'),
                ],
                type='deepir.SIRSTVOCDetSegDataset'),
            times=3,
            type='RepeatDataset'),
        pipeline=[
            dict(paste_by_box=True, selected=True, type='deepir.SkyCopyPaste'),
            dict(type='PackDetInputs'),
        ],
        type='MultiImageMixDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(paste_by_box=True, selected=True, type='deepir.SkyCopyPaste'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='Splits/test_v2.txt',
        backend_args=None,
        data_prefix=dict(sub_data_root=''),
        data_root='data/DenseSIRST/SIRSTdevkit/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(
                imdecode_backend='pillow',
                type='LoadAnnotations',
                with_bbox=True,
                with_seg=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='deepir.SIRSTVOCDetSegDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(eval_mode='11points', metric='mAP', type='VOCMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/fcos_r50-caffe_fpn_gn-head_1x_densesirst'
