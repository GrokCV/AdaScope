_base_ = [
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

dataset_type = "deepir.SIRSTVOCDetSegDataset"
data_root = "/root/data-tmp/BAFE-Net/data/DenseSIRST/SIRSTdevkit/"
backend_args = None

load_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_seg=False, imdecode_backend="pillow"),
    dict(type="Resize", scale=(512, 512), keep_ratio=False),
    dict(type="RandomFlip", prob=0.5),
]

train_pipeline = [
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_seg=False, imdecode_backend="pillow"),
    dict(type="Resize", scale=(512, 512), keep_ratio=False),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
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
    type="YOLOV3",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[0, 0, 0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type="Darknet",
        depth=53,
        out_indices=(2, 3, 4),
        init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://darknet53"),
    ),
    neck=dict(type="YOLOV3Neck", num_scales=3, in_channels=[512, 256, 128], out_channels=[256, 128, 64]),
    bbox_head=dict(
        type="YOLOV3Head",
        num_classes=1,
        in_channels=[256, 128, 64],
        out_channels=[512, 256, 128],
        anchor_generator=dict(
            type="YOLOAnchorGenerator",
            base_sizes=[[(12, 12), (16, 16), (20, 20)], [(6, 6), (8, 8), (10, 10)], [(2, 2), (3, 3), (4, 4)]],
            strides=[16, 8, 4],
        ),
        bbox_coder=dict(type="YOLOBBoxCoder"),
        featmap_strides=[16, 8, 4],
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0, reduction="sum"),
        loss_conf=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0, reduction="sum"),
        loss_xy=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=2.0, reduction="sum"),
        loss_wh=dict(type="MSELoss", loss_weight=2.0, reduction="sum"),
    ),
    train_cfg=dict(assigner=dict(type="GridAssigner", pos_iou_thr=0.3, neg_iou_thr=0.3, min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=5000,
        min_bbox_size=0,
        score_thr=0.0001,
        conf_thr=0.0001,
        nms=dict(type="nms", iou_threshold=0.45),
        max_per_img=1000,
    ),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=35, norm_type=2),
)

load_from = "https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth"
