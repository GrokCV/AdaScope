# VisDrone detection dataset base for C5 + refiner models.

data_root = '/opt/data/private/cjt/data/VisDrone'
dataset_type = 'CocoDataset'
backend_args = None

classes = (
    'pedestrian',
    'people',
    'bicycle',
    'car',
    'van',
    'truck',
    'tricycle',
    'awning-tricycle',
    'bus',
    'motor',
)
metainfo = dict(classes=classes)

input_size = (1536, 960)
stride = 32

train_cluster_json = '/opt/data/private/cjt/data/VisDrone/annotations/VisDrone2019-DET_train_clusters_yolc16x10.json'
val_cluster_json = '/opt/data/private/cjt/data/VisDrone/annotations/VisDrone2019-DET_val_clusters_yolc16x10.json'
rf_scale_bins = (10.0, 18.0)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=input_size, keep_ratio=True),
    dict(type='Pad', size=input_size),
    dict(
        type='GenerateC5TargetsFromClusterJSON',
        cluster_json=train_cluster_json,
        stride=stride,
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
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=input_size, keep_ratio=True),
    dict(type='Pad', size=input_size),
    dict(
        type='GenerateC5TargetsFromClusterJSON',
        cluster_json=val_cluster_json,
        stride=stride,
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
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/VisDrone2019-DET_train_coco.json',
        data_prefix=dict(img='VisDrone2019-DET-train/images/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/VisDrone2019-DET_val_coco.json',
        data_prefix=dict(img='VisDrone2019-DET-val/images/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/VisDrone2019-DET_val_coco.json',
        data_prefix=dict(img='VisDrone2019-DET-val/images/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
