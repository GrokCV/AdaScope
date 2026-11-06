"""AdaScope-TOOD: AdaScope with a TOOD detection head on DenseSIRST.

Usage:
    python tools/test_det.py configs/detection/cluster/AdaScope_TOOD.py CHECKPOINT.pth
    python tools/train_det.py configs/detection/cluster/AdaScope_TOOD.py
"""
import os.path as osp

_base_ = [
    '../_base_/datasets/adascope_densesirst.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'deepir.datasets.transforms.adascope_cluster_targets',
        'deepir.engine.hooks.adascope_stage_hook',
        'deepir.evaluation.metrics.selective_voc_metric',
        'deepir.models.detectors.adascope_detector',
        'deepir.models.refine.adascope_refiner',
    ],
    allow_failed_imports=False,
)

WORK_DIR = 'work_dirs/adascope_tood_densesirst'

# ── Model shortcuts ──────────────────────────────────────────────
MAX_PATCHES_PER_IMG = 20
LOCAL_SCORE_THR = 0.05
MERGE_NMS_IOU_THR = 0.5
INF = 1e8

# ── Model ────────────────────────────────────────────────────────
model = dict(
    type='AdaScopeDetector',
    auto_sync_external_from_global=True,
    use_refiner=True, use_local_head=False, share_local_with_global=True,
    train_global_head=True, train_local_head=False,
    local_loss_weight=1.0, local_train_use_gt_rois=True,
    refiner_handoff_alpha=0.0,
    cluster_feature_gate_strength=0.0, cluster_feature_gate_detach=True,
    cluster_score_fusion_weight=0.0, local_roi_source='blend',
    global_local_suppress_iou_thr=0.0, roi_topk=MAX_PATCHES_PER_IMG,
    roi_score_thr=LOCAL_SCORE_THR, merge_nms_iou_thr=MERGE_NMS_IOU_THR,
    cluster_connectivity='disconnected',
    data_preprocessor=dict(type='DetDataPreprocessor',
        mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False, pad_size_divisor=32),
    backbone=dict(type='ResNet', depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
        frozen_stages=-1, norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True, style='caffe',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(type='FPN', in_channels=[256, 512, 1024, 2048], out_channels=256,
        start_level=1, num_outs=5, relu_before_extra_convs=True),
    cluster_head=dict(type='AdaScopeClusterHead', in_channels=2048, core_size=512,
        core_stride=1, feat_channels=256,
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.75,
            loss_weight=50.0), threshold=0.3),
    global_bbox_head=dict(
        type='TOODHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=6,
        feat_channels=256,
        anchor_type='anchor_free',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
        ),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,
            beta=2.0,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        train_cfg=dict(
            initial_epoch=4,
            initial_assigner=dict(type='ATSSAssigner', topk=9),
            assigner=dict(type='TaskAlignedAssigner', topk=13),
            alpha=1.0,
            beta=6.0,
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
    ),
    refiner=dict(type='AdaScopeRefiner', in_channels=256,
        feat_channels=256, num_ins=5, fusion_level=2, fusion_type='concat',
        resize_mode='bilinear', state_roi_size=(7, 7), state_sampling_ratio=2,
        hidden_dim=256, min_box_size=2.0,
        template_shift_values=(-0.5, -0.25, 0.0, 0.25, 0.5),
        template_shape_width_values=(0.75, 1.0, 1.25, 1.5),
        template_shape_height_values=(0.75, 1.0, 1.25, 1.5),
        template_rf_expand_values=(1.0, 1.25, 1.5, 2.0, 2.5)),
    proposal_score_reduction='max', proposal_train_topk=32,
    proposal_train_score_thr=0.05, proposal_match_iou_thr=0.1,
    proposal_bbox_loss_weight=0.25, proposal_iou_loss_weight=0.5,
    proposal_smooth_l1_beta=1.0, proposal_min_box_size=2.0,
    proposal_use_detached_logits=True, proposal_offset_clamp=1.5,
    proposal_scale_log_clamp=1.0,
    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(type='ATSSAssigner', topk=9),
        assigner=dict(type='TaskAlignedAssigner', topk=13),
        alpha=1.0,
        beta=6.0,
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(nms_pre=1000, min_bbox_size=0, score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5), max_per_img=100),
    enable_external_local=True, external_local_frozen=True,
    external_local_device='cuda:0',
    external_local_cfg='configs/detection/cluster/adascope_fcos_local.py',
    external_local_ckpt='', roi_source='refined', raw_roi_expand_ratio=1.0,
    ext_local_score_thr=0.05, ext_local_score_scale=1.0,
    ext_local_batch_size=8, ext_local_max_per_patch=100,
    stitched_cell_size=(256, 256), stitched_cell_gap=8, stitched_max_cols=4,
    policy_reward_det_weight=1.0, policy_reward_geo_weight=0.2,
    policy_reward_area_weight=0.1, policy_reward_fp_weight=0.15,
    policy_reward_peak_weight=1.0, policy_reward_mass_weight=0.25,
    policy_reward_count_weight=0.15, policy_reward_purity_weight=0.1,
    policy_reward_outside_mass_weight=0.05, policy_cover_penalty_weight=0.05,
    policy_use_gtcluster_det_baseline=False, policy_fp_iou_thr=0.1,
    policy_margin_reg_weight=0.01, policy_reward_clip=2.0, policy_train_topk=8,
    grpo_group_size=4, grpo_update_steps=2, grpo_clip_eps=0.2,
    grpo_policy_clip_enabled=False, grpo_entropy_weight=0.001,
    grpo_shift_reg_weight=None, grpo_margin_reg_weight=0.25,
    grpo_cover_reg_weight=0.5, grpo_area_reg_weight=0.1,
    grpo_shift_mag_reg_weight=0.02, grpo_area_budget=2.0,
    grpo_cover_keep_ratio=0.9, grpo_dense_min_gt=2, grpo_advantage_eps=1e-6,
    grpo_ref_kl_weight=0.0, grpo_ref_kl_loss_scale=0.0,
    grpo_ref_kl_safe_max=0.0, grpo_use_reference_policy=False,
    refiner_sup_center_weight=1.0, refiner_sup_use_policy_topk=False,
)

# ── Dense training budget ────────────────────────────────────────
# The TOOD schedule uses 20 epochs; repeat the 768-image trainval split
# three times so each epoch sees the same number of iterations as the
# reported experiments.
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='deepir.SIRSTVOCDetDataset',
            data_root='/path/to/your/DenseSIRST/SIRSTdevkit',
            ann_file='Splits/trainval_v2.txt',
            data_prefix=dict(sub_data_root=''),
            filter_cfg=dict(filter_empty_gt=False, min_size=0, bbox_min_size=0),
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', scale=(512, 512), keep_ratio=False),
                dict(
                    type='AdaScopeClusterTargets',
                    stride=32,
                    rf_scale_bins=(1.5, 3.0),
                    cluster_xml_suffix='_with_clusters.xml',
                    cluster_tag='cluster',
                    cluster_name='Target',
                    cluster_xml_subdir='/path/to/your/DenseSIRST/SIRSTdevkit/SIRST/Cluster_relabel_graph_v2',
                    missing_policy='empty',
                    minus_one=True,
                ),
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'gt_cls_map', 'gt_offset_map',
                               'gt_offset_weight', 'gt_scale_map',
                               'gt_scale_weight', 'gt_rf_level', 'gt_rf_weight',
                               'gt_cluster_bboxes'),
                ),
            ],
            backend_args=None,
        ),
    ),
)

# ── Optimizer ────────────────────────────────────────────────────
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0/3, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=20, by_epoch=True,
         milestones=[12, 18], gamma=0.1),
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ── Evaluation ───────────────────────────────────────────────────
# By default only the merged metric is evaluated/reported.
# Set SHOW_ALL_METRICS = True to also report per-branch metrics
# (global / raw_cluster / refined_cluster / local).
SHOW_ALL_METRICS = False

_merged_metric = dict(type='VOCMetric', metric='mAP', eval_mode='11points', prefix='merged_voc')

_branch_metrics = [
    dict(type='SelectiveVOCMetric', metric='mAP', eval_mode='11points',
         gt_source='instance', pred_key='pred_global_instances', prefix='global_voc'),
    dict(type='SelectiveVOCMetric', metric='mAP', eval_mode='11points',
         gt_source='cluster', pred_key='pred_cluster_instances', prefix='raw_cluster_voc'),
    dict(type='SelectiveVOCMetric', metric='mAP', eval_mode='11points',
         gt_source='cluster', pred_key='pred_refined_cluster_instances',
         prefix='refined_cluster_voc'),
    dict(type='SelectiveVOCMetric', metric='mAP', eval_mode='11points',
         gt_source='instance', pred_key='pred_local_instances', prefix='local_voc'),
]

val_evaluator = [_merged_metric]
test_evaluator = val_evaluator
if SHOW_ALL_METRICS:
    val_evaluator = _branch_metrics + val_evaluator
    test_evaluator = val_evaluator

# ── Hooks ────────────────────────────────────────────────────────
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1,
                    save_best='merged_voc/mAP', rule='greater'),
)
custom_hooks = [
    dict(type='AdaScopeGRPOStageHook', warmup_epochs=12,
         refiner_supervised_epochs=4, reference_metric_key='merged_voc/mAP'),
]

work_dir = WORK_DIR
