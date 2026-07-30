import os.path as osp

_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'deepir.datasets.transforms.cluster_gt_targets',
        'deepir.engine.hooks.syn_warmup_sup_grpo_stage_hook',
        'deepir.evaluation.metrics.selective_voc_metric',
        'deepir.models.detectors.plain_flat_sync_rl_detector',
        'deepir.models.refine.plain_template_three_action_rl_refiner',
    ],
    allow_failed_imports=False,
)

REPO_ROOT = '/root/data-tmp/BAFE-Net'
DATA_ROOT = osp.join(REPO_ROOT, 'data', 'DenseSIRST', 'SIRSTdevkit')
CLUSTER_DIR = osp.join(DATA_ROOT, 'SIRST', 'Cluster_relabel_graph_v2')
WORK_DIR = osp.join(REPO_ROOT, 'work_dirs', 'adazoom_ppo_r50_fpn_1x_densesirst')

dataset_type = 'deepir.SIRSTVOCDetSegDataset'
backend_args = None

core_scale = 512
core_stride = 1
stride = 32 * core_stride
rf_scale_bins = (1.5, 3.0)
inf = 1e8

load_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(core_scale, core_scale), keep_ratio=False),
    dict(
        type='GenerateC5TargetsFromClusterGT',
        stride=stride,
        rf_scale_bins=rf_scale_bins,
        cluster_xml_suffix='_with_clusters.xml',
        cluster_tag='cluster',
        cluster_name='Target',
        cluster_xml_subdir=CLUSTER_DIR,
        missing_policy='empty',
        minus_one=True,
    ),
]

train_pipeline = [
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'gt_cls_map',
            'gt_offset_map',
            'gt_offset_weight',
            'gt_scale_map',
            'gt_scale_weight',
            'gt_rf_level',
            'gt_rf_weight',
            'gt_cluster_bboxes',
        ),
    ),
]

test_pipeline = [
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'gt_cls_map',
            'gt_offset_map',
            'gt_offset_weight',
            'gt_scale_map',
            'gt_scale_weight',
            'gt_rf_level',
            'gt_rf_weight',
            'gt_cluster_bboxes',
        ),
    ),
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
            pipeline=load_pipeline + train_pipeline,
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
        pipeline=load_pipeline + test_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = val_dataloader

model = dict(
    type='PlainFlatSyncPPODetector',
    rl_algorithm='ppo',
    auto_sync_external_from_global=True,
    use_refiner=True,
    use_local_head=False,
    share_local_with_global=True,
    train_global_head=True,
    train_local_head=False,
    local_loss_weight=1.0,
    local_train_use_gt_rois=True,
    refiner_handoff_alpha=0.0,
    cluster_feature_gate_strength=0.0,
    cluster_feature_gate_detach=True,
    cluster_score_fusion_weight=0.0,
    local_roi_source='blend',
    global_local_suppress_iou_thr=0.0,
    roi_topk=20,
    roi_score_thr=0.05,
    merge_nms_iou_thr=0.5,
    cluster_connectivity='disconnected',
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
        start_level=1,
        num_outs=5,
        relu_before_extra_convs=True,
    ),
    cluster_head=dict(
        type='C5ClusterHead',
        in_channels=2048,
        core_size=core_scale,
        core_stride=core_stride,
        feat_channels=256,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.75,
            loss_weight=50.0,
        ),
        threshold=0.3,
    ),
    global_bbox_head=dict(
        type='FCOSHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, inf)),
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
    ),
    refiner=dict(
        type='PlainTemplateThreeActionPPORefiner',
        in_channels=256,
        feat_channels=256,
        num_ins=5,
        fusion_level=2,
        fusion_type='concat',
        resize_mode='bilinear',
        state_roi_size=(7, 7),
        state_sampling_ratio=2,
        hidden_dim=256,
        min_box_size=2.0,
        template_shift_values=(-0.5, -0.25, 0.0, 0.25, 0.5),
        template_shape_width_values=(0.75, 1.0, 1.25, 1.5),
        template_shape_height_values=(0.75, 1.0, 1.25, 1.5),
        template_rf_expand_values=(1.0, 1.25, 1.5, 2.0, 2.5),
    ),
    proposal_score_reduction='max',
    proposal_train_topk=32,
    proposal_train_score_thr=0.05,
    proposal_match_iou_thr=0.1,
    proposal_bbox_loss_weight=0.25,
    proposal_iou_loss_weight=0.5,
    proposal_smooth_l1_beta=1.0,
    proposal_min_box_size=2.0,
    proposal_use_detached_logits=True,
    proposal_offset_clamp=1.5,
    proposal_scale_log_clamp=1.0,
    train_cfg=None,
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
    ),
    enable_external_local=True,
    external_local_frozen=True,
    external_local_device='cuda:0',
    external_local_cfg='configs/detection/fcos/fcos_r50-caffe_fpn_gn-head_1x_densesirst.py',
    external_local_ckpt='',
    roi_source='refined',
    raw_roi_expand_ratio=1.0,
    ext_local_score_thr=0.05,
    ext_local_score_scale=1.0,
    ext_local_batch_size=8,
    ext_local_max_per_patch=100,
    stitched_cell_size=(256, 256),
    stitched_cell_gap=8,
    stitched_max_cols=4,
    policy_reward_det_weight=1.25,
    policy_reward_geo_weight=0.05,
    policy_reward_area_weight=0.02,
    policy_reward_fp_weight=0.15,
    policy_reward_peak_weight=1.0,
    policy_reward_mass_weight=0.3,
    policy_reward_count_weight=0.25,
    policy_reward_purity_weight=0.1,
    policy_reward_outside_mass_weight=0.05,
    policy_cover_penalty_weight=0.05,
    policy_use_gtcluster_det_baseline=False,
    policy_fp_iou_thr=0.1,
    policy_reward_clip=2.0,
    policy_train_topk=8,
    rl_group_size=4,
    rl_update_steps=4,
    ppo_clip_eps=0.1,
    rl_entropy_weight=0.0005,
    rl_advantage_eps=1e-6,
    rl_use_reference_policy=False,
    ppo_value_loss_weight=0.5,
    refiner_sup_center_weight=1.0,
    refiner_sup_use_policy_topk=False,
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
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

val_evaluator = [
    dict(
        type='SelectiveVOCMetric',
        metric='mAP',
        eval_mode='11points',
        gt_source='instance',
        pred_key='pred_global_instances',
        prefix='global_voc',
    ),
    dict(
        type='SelectiveVOCMetric',
        metric='mAP',
        eval_mode='11points',
        gt_source='cluster',
        pred_key='pred_cluster_instances',
        prefix='raw_cluster_voc',
    ),
    dict(
        type='SelectiveVOCMetric',
        metric='mAP',
        eval_mode='11points',
        gt_source='cluster',
        pred_key='pred_refined_cluster_instances',
        prefix='refined_cluster_voc',
    ),
    dict(
        type='SelectiveVOCMetric',
        metric='mAP',
        eval_mode='11points',
        gt_source='instance',
        pred_key='pred_local_instances',
        prefix='local_voc',
    ),
    dict(type='VOCMetric', metric='mAP', eval_mode='11points', prefix='merged_voc'),
]

test_evaluator = val_evaluator

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='merged_voc/mAP',
        rule='greater',
    ),
)

custom_hooks = [
    dict(
        type='SynWarmupSupGRPOStageHook',
        warmup_epochs=12,
        refiner_supervised_epochs=4,
        reference_metric_key='merged_voc/mAP',
    ),
]

work_dir = WORK_DIR
