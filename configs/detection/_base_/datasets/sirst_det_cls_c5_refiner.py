# C5 refiner dataset base: use single-object annotations, then build cluster GT in pipeline.

data_root = '/data/DenseSIRST/SIRSTdevkit'
dataset_type = 'deepir.SIRSTVOCDetSegDataset'
backend_args = None

core_scale = 512
stride = 32

cluster_grid_size = (64, 40)
cluster_topk = 15
cluster_thresh_ratio = 10.0 / 11.0
cluster_min_overlap = 0.3
cluster_min_radius = 1
rf_scale_bins = (1.5, 3.0)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(core_scale, core_scale), keep_ratio=False),
    dict(
        type='GenerateC5Targets',
        stride=stride,
        cluster_gt=True,
        cluster_grid_size=cluster_grid_size,
        cluster_topk=cluster_topk,
        cluster_thresh_ratio=cluster_thresh_ratio,
        cluster_min_overlap=cluster_min_overlap,
        cluster_min_radius=cluster_min_radius,
        rf_scale_bins=rf_scale_bins),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'gt_cls_map', 'gt_offset_map', 'gt_offset_weight',
            'gt_scale_map', 'gt_scale_weight',
            'gt_rf_level', 'gt_rf_weight',
            'gt_cluster_bboxes'))
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(core_scale, core_scale), keep_ratio=False),
    dict(
        type='GenerateC5Targets',
        stride=stride,
        cluster_gt=True,
        cluster_grid_size=cluster_grid_size,
        cluster_topk=cluster_topk,
        cluster_thresh_ratio=cluster_thresh_ratio,
        cluster_min_overlap=cluster_min_overlap,
        cluster_min_radius=cluster_min_radius,
        rf_scale_bins=rf_scale_bins),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'gt_cls_map', 'gt_offset_map', 'gt_offset_weight',
            'gt_scale_map', 'gt_scale_weight',
            'gt_rf_level', 'gt_rf_weight',
            'gt_cluster_bboxes'))
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='Splits/trainval_v2.txt',
        data_prefix=dict(sub_data_root=''),
        filter_cfg=dict(filter_empty_gt=False, min_size=0, bbox_min_size=0),
        pipeline=train_pipeline,
        backend_args=backend_args,
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
        data_root=data_root,
        ann_file='Splits/test_v2.txt',
        data_prefix=dict(sub_data_root=''),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = val_dataloader
