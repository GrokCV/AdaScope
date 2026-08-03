_base_ = ["./gfl_r50_fpn_1x_densesirst_refdense.py"]

dataset_type = "deepir.SIRSTVOCDetDataset"
data_root = "/root/data-tmp/BAFE-Net/data/DenseSIRST/SIRSTdevkit/"
backend_args = None

load_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_seg=False, imdecode_backend="pillow"),
]

train_pipeline = [
    dict(type="Mosaic", img_scale=(256, 256), pad_val=114.0),
    dict(type="RandomFlip", prob=0.5),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_seg=False, imdecode_backend="pillow"),
    dict(type="Resize", scale=(512, 512), keep_ratio=False),
    dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type="MultiImageMixDataset",
        dataset=dict(
            type="RepeatDataset",
            times=3,
            dataset=dict(
                type=dataset_type,
                data_root=data_root,
                ann_file="Splits/trainval_v2.txt",
                data_prefix=dict(sub_data_root=""),
                filter_cfg=dict(filter_empty_gt=False, min_size=0, bbox_min_size=0),
                pipeline=load_pipeline,
                backend_args=backend_args,
            ),
        ),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="Splits/test_v2.txt",
        data_prefix=dict(sub_data_root=""),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="VOCMetric", metric="mAP", eval_mode="11points")
test_evaluator = val_evaluator

model = dict(
    backbone=dict(
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=False),
        style="pytorch",
    ),
    neck=dict(
        start_level=0,
        end_level=0,
        add_extra_convs="on_output",
        num_outs=1,
        relu_before_extra_convs=True,
    ),
    bbox_head=dict(
        stacked_convs=2,
        feat_channels=128,
        anchor_generator=dict(
            type="AnchorGenerator",
            ratios=[1.0],
            scales=[0.75],
            center_offset=0.5,
            strides=[4],
        ),
        reg_max=4,
        loss_cls=dict(type="QualityFocalLoss", use_sigmoid=True, beta=2.0, loss_weight=0.5),
        loss_dfl=dict(type="DistributionFocalLoss", loss_weight=0.25),
        loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
    ),
    train_cfg=dict(assigner=dict(type="ATSSAssigner", topk=3), allowed_border=-1, pos_weight=-1, debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type="nms", iou_threshold=0.6),
        max_per_img=300,
    ),
)

optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0001))

load_from = "https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r50_fpn_1x_coco/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth"
