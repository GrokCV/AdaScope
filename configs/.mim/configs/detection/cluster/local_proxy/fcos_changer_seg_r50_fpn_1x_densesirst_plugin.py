_base_ = [
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py',
    '../../_base_/datasets/sirst_det_cls_c5_refiner.py',
]

custom_imports = dict(
    imports=[
        'deepir.models.dense_heads.fcos_seg_head',
        'deepir.models.dense_heads.fcos_changer_seg_head',
    ],
    allow_failed_imports=False,
)

dataset_type = 'deepir.SIRSTVOCDetSegDataset'
data_root = '/root/data-tmp/data/DenseSIRST/SIRSTdevkit'
backend_args = None

load_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_seg=True,
        imdecode_backend='pillow',
    ),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
]

train_pipeline = [dict(type='PackDetInputs')]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_seg=True,
        imdecode_backend='pillow',
    ),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        _delete_=True,
        type='MultiImageMixDataset',
        dataset=dict(
            type='RepeatDataset',
            times=3,
            dataset=dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='Splits/trainval_v2.txt',
                data_prefix=dict(sub_data_root=''),
                filter_cfg=dict(filter_empty_gt=False, min_size=0, bbox_min_size=0),
                pipeline=load_pipeline,
                backend_args=backend_args,
            ),
        ),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='Splits/test_v2.txt',
        data_prefix=dict(sub_data_root=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator

model = dict(
    type='FCOS',
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
        type='deepir.FCOSChangerSegHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32],
        regress_ranges=((-1, 64), (64, 128), (128, 256)),
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
            loss_weight=1.0,
        ),
        loss_seg=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            ignore_index=255,
        ),
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
    ),
)
