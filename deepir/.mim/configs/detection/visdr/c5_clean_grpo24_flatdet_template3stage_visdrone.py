_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/visdrone_det_c5_refiner.py',
]

custom_imports = dict(
    imports=[
        'deepir.datasets.transforms.cluster_json_targets',
        'deepir.evaluation.metrics.selective_coco_metric',
        'deepir.engine.hooks.syn_warmup_sup_grpo_stage_hook',
        'deepir.models.detectors.flat_sync_clean_grpo_detector',
        'deepir.models.detectors.flat_sync_clean_grpo_detector_visdrone',
        'deepir.models.refine.fpn_template_grpo_refiner',
    ],
    allow_failed_imports=False,
)

VISDRONE_DATA_ROOT = '/opt/data/private/cjt/data/VisDrone'
CORE_SCALE = 1536
CORE_STRIDE = 1
INF = 1e8

MAX_PATCHES_PER_IMG = 16
LOCAL_SCORE_THR = 0.05
MERGE_NMS_IOU_THR = 0.5

model = dict(
    type='FlatSyncCleanGRPODetectorVisDrone',
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
    roi_topk=MAX_PATCHES_PER_IMG,
    roi_score_thr=LOCAL_SCORE_THR,
    merge_nms_iou_thr=MERGE_NMS_IOU_THR,
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
    global_bbox_head=dict(
        type='FCOSHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)),
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
        template_shift_values=(-0.5, -0.25, 0.0, 0.25, 0.5),
        template_shrink_values=(0.0, 0.1, 0.2, 0.333333),
        template_context_values=(0.0, 0.25, 0.5, 1.0, 1.5),
    ),
    proposal_score_reduction='max',
    proposal_train_topk=20,
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
        max_per_img=300,
    ),
    enable_external_local=True,
    external_local_frozen=True,
    external_local_device='cuda:0',
    external_local_cfg='/opt/data/private/cjt/BAFE-Net/configs/detection/fcos/fcos_r50-caffe_fpn_gn-head_48e_visdrone_plain_noflip_mscoco.py',
    external_local_ckpt='',
    roi_source='refined',
    raw_roi_expand_ratio=1.0,
    ext_local_score_thr=0.05,
    ext_local_score_scale=1.0,
    ext_local_batch_size=2,
    ext_local_max_per_patch=150,
    stitched_cell_size=(384, 384),
    stitched_cell_gap=8,
    stitched_max_cols=4,
    policy_reward_det_weight=1.0,
    policy_reward_geo_weight=0.2,
    policy_reward_area_weight=0.1,
    policy_reward_fp_weight=0.15,
    policy_reward_peak_weight=1.0,
    policy_reward_mass_weight=0.25,
    policy_reward_count_weight=0.2,
    policy_reward_purity_weight=0.1,
    policy_reward_outside_mass_weight=0.05,
    policy_use_gtcluster_det_baseline=False,
    policy_fp_iou_thr=0.1,
    policy_margin_reg_weight=0.0,
    policy_reward_clip=2.0,
    policy_train_topk=6,
    grpo_group_size=4,
    grpo_update_steps=2,
    grpo_clip_eps=0.05,
    grpo_policy_clip_enabled=True,
    grpo_entropy_weight=0.0002,
    grpo_shift_reg_weight=0.0,
    grpo_margin_reg_weight=0.0,
    grpo_cover_reg_weight=0.0,
    grpo_area_reg_weight=0.0,
    grpo_shift_mag_reg_weight=0.0,
    grpo_area_budget=1.9,
    grpo_dense_min_gt=0,
    grpo_advantage_eps=1e-6,
    grpo_ref_kl_weight=0.1,
    grpo_ref_kl_loss_scale=10.0,
    grpo_ref_kl_safe_max=200.0,
    grpo_use_reference_policy=True,
    grpo_reference_metric_key='merged_coco/bbox_mAP',
    grpo_seed_reference_on_stage_enable=True,
    refiner_sup_center_weight=1.0,
    refiner_sup_use_policy_topk=False,
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35.0, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', begin=0, end=60, by_epoch=True, milestones=[48, 56], gamma=0.1),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=60, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = [
    dict(
        type='SelectiveCocoMetric',
        ann_file=f'{VISDRONE_DATA_ROOT}/annotations/VisDrone2019-DET_val_coco.json',
        metric='bbox',
        proposal_nums=(1, 10, 100),
        metric_items=['mAP', 'mAP_50', 'mAP_75'],
        pred_key='pred_global_instances',
        gt_source='instance',
        prefix='global_coco',
    ),
    dict(
        type='SelectiveCocoMetric',
        ann_file=f'{VISDRONE_DATA_ROOT}/annotations/VisDrone2019-DET_val_coco.json',
        metric='bbox',
        proposal_nums=(1, 10, 100),
        metric_items=['mAP', 'mAP_50', 'mAP_75'],
        pred_key='pred_local_instances',
        gt_source='instance',
        prefix='local_coco',
    ),
    dict(
        type='CocoMetric',
        ann_file=f'{VISDRONE_DATA_ROOT}/annotations/VisDrone2019-DET_val_coco.json',
        metric='bbox',
        proposal_nums=(1, 10, 100),
        metric_items=['mAP', 'mAP_50', 'mAP_75'],
        prefix='merged_coco',
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
        save_best='merged_coco/bbox_mAP',
        rule='greater',
    ),
)

custom_hooks = [
    dict(
        type='SynWarmupSupGRPOStageHook',
        warmup_epochs=48,
        refiner_supervised_epochs=4,
        reference_metric_key='merged_coco/bbox_mAP',
    ),
]

work_dir = '/opt/data/private/cjt/BAFE-Net/work_dirs/c5_clean_grpo24_flatdet_template3stage_visdrone'
