# VisDrone detection dataset base for plain detector baselines.

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

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=input_size, keep_ratio=True),
    dict(type='Pad', size=input_size),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=input_size, keep_ratio=True),
    dict(type='Pad', size=input_size),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

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
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root}/annotations/VisDrone2019-DET_val_coco.json',
    metric='bbox',
    proposal_nums=(1, 10, 100),
    metric_items=['mAP', 'mAP_50', 'mAP_75'],
    prefix='merged_coco',
)
test_evaluator = val_evaluator
