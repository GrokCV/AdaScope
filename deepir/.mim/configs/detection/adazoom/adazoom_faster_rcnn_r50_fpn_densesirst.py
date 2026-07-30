import os.path as osp

_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'deepir.datasets',
        'deepir.models.detectors.adazoom_faster_rcnn',
        'deepir.engine.hooks.adazoom_stage_hook',
    ],
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
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='RepeatDataset',
        times=3,
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

val_evaluator = dict(
    type='VOCMetric',
    iou_thrs=0.5,
    metric='mAP',
    eval_mode='11points',
    prefix='merged',
)
test_evaluator = val_evaluator

model = dict(
    type='AdaZoomFasterRCNN',
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
        out_channels=96,
        num_outs=5,
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=96,
        feat_channels=96,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[1, 2, 4],
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
            out_channels=96,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=96,
            fc_out_channels=256,
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
            nms_pre=300,
            max_per_img=150,
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
            nms_pre=300,
            max_per_img=150,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=50,
        ),
    ),
    episode_len=7,
    feature_downsample_stride=32,
    gamma=0.5,
    state_decay=0.1,
    reward_beta=1.5,
    patch_resize=(800, 800),
    candidate_scales=(240.0, 350.0, 420.0),
    desired_object_scale_ranges=((0.0, 40.0), (30.0, 60.0), (50.0, 1e6)),
    candidate_ratios=(0.7, 1.0, 1.5),
    use_collaborative_reward=True,
)

base_lr = 1.0
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='DAdaptAdam',
        lr=base_lr,
        weight_decay=0.05,
        decouple=True,
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
    ),
)

param_scheduler = [
    dict(type='ConstantLR', factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=20, by_epoch=True, milestones=[8, 16], gamma=0.1),
]

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=20,
    val_interval=1,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

custom_hooks = [
    dict(
        type='AdaZoomStageHook',
        detector_iters=90000,
        policy_iters=5000,
        collaborative_policy_iters=500,
        collaborative_detector_iters=1000,
        detector_lr=base_lr,
        policy_lr=2e-5,
        collaborative_detector_lr=base_lr * 0.1,
        collaborative_policy_lr=2e-6,
    ),
]

work_dir = '/root/data-tmp/BAFE-Net/work_dirs/adazoom_faster_rcnn_r50_fpn_densesirst'
