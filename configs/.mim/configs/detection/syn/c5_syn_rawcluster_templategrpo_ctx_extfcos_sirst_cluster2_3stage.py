_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/sirst_det_cls_c5_refiner.py',
]

custom_imports = dict(
    imports=[
        'deepir.datasets.transforms.instance_grid_cluster_gt_targets',
        'deepir.engine.hooks.syn_warmup_sup_grpo_stage_hook',
        'deepir.models.detectors.syn_single_stage_rawcluster_grpo_detector',
        'deepir.models.refine.fpn_template_grpo_refiner',
        'deepir.evaluation.metrics.selective_voc_metric',
    ],
    allow_failed_imports=False,
)

cluster_dir = '/opt/data/private/cjt/data/DenseSIRST/SIRSTdevkit/SIRST/Cluster2'

CORE_SCALE = 512
CORE_STRIDE = 1
STRIDE = 32 * CORE_STRIDE

MAX_PATCHES_PER_IMG = 20
LOCAL_SCORE_THR = 0.05
MERGE_NMS_IOU_THR = 0.5

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(CORE_SCALE, CORE_SCALE), keep_ratio=False),
    dict(
        type='GenerateC5InstanceGridTargetsFromClusterGT',
        stride=STRIDE,
        cluster_xml_suffix='_with_clusters.xml',
        cluster_tag='cluster',
        cluster_name='Target',
        cluster_xml_subdir=cluster_dir,
        missing_policy='empty',
        minus_one=True,
    ),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'gt_cls_map', 'gt_offset_map', 'gt_offset_weight',
            'gt_scale_map', 'gt_scale_weight',
            'gt_rf_level', 'gt_rf_weight',
            'gt_cluster_bboxes',
        ),
    ),
]

val_pipeline = train_pipeline
test_pipeline = val_pipeline

train_dataloader = dict(batch_size=2, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

model = dict(
    type='SynSingleStageRawClusterGRPODetector',
    use_refiner=True,
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
        core_size=CORE_SCALE,
        core_stride=CORE_STRIDE,
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
    refiner=dict(
        type='FPNTemplateGRPORefiner',
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
        template_shift_values=(-0.25, 0.0, 0.25),
        template_shrink_values=(0.0, 0.15, 0.30, 0.45, 0.60),
        template_context_values=(0.0, 0.25, 0.5, 0.75, 1.0),
    ),
    roi_topk=MAX_PATCHES_PER_IMG,
    roi_score_thr=LOCAL_SCORE_THR,
    merge_nms_iou_thr=MERGE_NMS_IOU_THR,
    proposal_score_reduction='max',
    proposal_train_topk=32,
    proposal_train_score_thr=0.05,
    proposal_match_iou_thr=0.1,
    proposal_min_box_size=2.0,
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
    external_local_ckpt='/opt/data/private/cjt/BAFE-Net/checkpoints/fcos_globalonly_20260311_best.pth',
    roi_source='refined',
    raw_roi_expand_ratio=1.0,
    ext_local_score_thr=0.05,
    ext_local_score_scale=1.0,
    ext_local_batch_size=8,
    ext_local_max_per_patch=100,
    stitched_cell_size=(256, 256),
    stitched_cell_gap=8,
    stitched_max_cols=4,
    policy_reward_det_weight=1.0,
    policy_reward_geo_weight=0.1,
    policy_reward_fp_weight=0.2,
    policy_reward_area_weight=0.25,
    policy_reward_area_budget=2.25,
    policy_reward_context_weight=0.05,
    policy_fp_iou_thr=0.1,
    policy_margin_reg_weight=0.0,
    policy_reward_clip=2.0,
    policy_train_topk=4,
    grpo_group_size=4,
    grpo_update_steps=2,
    grpo_clip_eps=0.05,
    grpo_policy_clip_enabled=True,
    grpo_entropy_weight=0.0002,
    grpo_shift_reg_weight=0.0,
    grpo_margin_reg_weight=0.0,
    grpo_cover_reg_weight=0.0,
    grpo_area_reg_weight=0.0,
    grpo_geo_reg_weight=0.0,
    grpo_geo_degrade_reg_weight=0.0,
    grpo_shift_mag_reg_weight=0.0,
    grpo_area_budget=1.5,
    grpo_dense_min_gt=0,
    grpo_advantage_eps=1e-6,
    grpo_ref_kl_weight=0.1,
    grpo_ref_kl_loss_scale=10.0,
    grpo_ref_kl_safe_max=200.0,
    grpo_use_reference_policy=True,
    grpo_reference_metric_key='merged_voc/mAP',
    grpo_seed_reference_on_stage_enable=True,
    grpo_reg_target_mode='envelope',
    refiner_sup_box_l1_weight=1.0,
    refiner_sup_iou_weight=2.0,
    refiner_sup_center_weight=1.0,
    refiner_sup_use_policy_topk=False,
    refiner_sup_target_mode='envelope',
    grpo_context_roi_limit_factor=2.0,
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=5.0, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=24, by_epoch=True, milestones=[16, 22], gamma=0.1),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = [
    dict(
        type='SelectiveVOCMetric',
        metric='mAP',
        eval_mode='11points',
        gt_source='cluster',
        pred_key='pred_refined_cluster_instances',
        prefix='cluster_voc',
    ),
    dict(
        type='VOCMetric',
        metric='mAP',
        eval_mode='11points',
        prefix='merged_voc',
    ),
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

work_dir = '/opt/data/private/cjt/BAFE-Net/work_dirs/c5_syn_rawcluster_templategrpo_ctx_extfcos_sirst_cluster2_3stage_syncfix'
