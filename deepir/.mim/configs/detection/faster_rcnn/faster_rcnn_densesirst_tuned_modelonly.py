import os.path as osp

_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['deepir.datasets'],
    allow_failed_imports=False,
)

REPO_ROOT = '/root/data-tmp/BAFE-Net'
DATA_ROOT = osp.join(REPO_ROOT, 'data', 'DenseSIRST', 'SIRSTdevkit')
dataset_type = 'deepir.SIRSTVOCDetSegDataset'
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_seg=False,
        imdecode_backend='pillow'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(512, 512)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_seg=False,
        imdecode_backend='pillow'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=DATA_ROOT,
            ann_file='Splits/trainval_v2.txt',
            data_prefix=dict(sub_data_root=''),
            filter_cfg=dict(filter_empty_gt=False, min_size=0, bbox_min_size=0),
            pipeline=train_pipeline,
            backend_args=backend_args,
        ),
    ),
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

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points', prefix='voc')
test_evaluator = val_evaluator

model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32,
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
        num_outs=5,
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[1, 2, 4, 8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=True,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        ),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=4000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=200,
        ),
    ),
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True, milestones=[8, 11], gamma=0.1),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

work_dir = '/root/data-tmp/BAFE-Net/work_dirs/faster_rcnn_densesirst_tuned_modelonly'
