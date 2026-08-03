"""DenseSIRST dataset config for AdaScope.

Override DATA_ROOT in your experiment config to point to your local DenseSIRST directory.
Example: _base_ = [...], then DATA_ROOT = '/path/to/SIRSTdevkit'
"""
import os.path as osp

# Default DATA_ROOT — users should override this
DATA_ROOT = 'data/DenseSIRST/SIRSTdevkit'
CLUSTER_DIR = osp.join(DATA_ROOT, 'SIRST', 'Cluster_relabel_graph_v2')

DATASET_TYPE = 'deepir.SIRSTVOCDetDataset'
BACKEND_ARGS = None
RF_SCALE_BINS = (1.5, 3.0)
CORE_SCALE = 512
CORE_STRIDE = 1
STRIDE = 32 * CORE_STRIDE

# ── Pipelines ────────────────────────────────────────────────────
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=BACKEND_ARGS),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(CORE_SCALE, CORE_SCALE), keep_ratio=False),
    dict(
        type='GenerateC5TargetsFromClusterGT',
        stride=STRIDE,
        rf_scale_bins=RF_SCALE_BINS,
        cluster_xml_suffix='_with_clusters.xml',
        cluster_tag='cluster',
        cluster_name='Target',
        cluster_xml_subdir=CLUSTER_DIR,
        missing_policy='empty',
        minus_one=True,
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                   'gt_cls_map', 'gt_offset_map', 'gt_offset_weight',
                   'gt_scale_map', 'gt_scale_weight',
                   'gt_rf_level', 'gt_rf_weight', 'gt_cluster_bboxes'),
    ),
]
val_pipeline = train_pipeline
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=2, num_workers=2, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type=DATASET_TYPE, data_root=DATA_ROOT,
                 ann_file='Splits/trainval_v2.txt', data_prefix=dict(sub_data_root=''),
                 filter_cfg=dict(filter_empty_gt=False, min_size=0, bbox_min_size=0),
                 pipeline=train_pipeline, backend_args=BACKEND_ARGS),
)
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=DATASET_TYPE, data_root=DATA_ROOT,
                 ann_file='Splits/test_v2.txt', data_prefix=dict(sub_data_root=''),
                 test_mode=True, pipeline=val_pipeline, backend_args=BACKEND_ARGS),
)
test_dataloader = val_dataloader
