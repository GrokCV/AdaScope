_base_ = ["./yolov3_d53_1x_densesirst_refdensev3.py"]

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
    backbone=dict(out_indices=(2, 3, 4)),
    neck=dict(type="YOLOV3Neck", num_scales=3, in_channels=[512, 256, 128], out_channels=[256, 128, 64]),
    bbox_head=dict(
        in_channels=[256, 128, 64],
        out_channels=[512, 256, 128],
        anchor_generator=dict(
            type="YOLOAnchorGenerator",
            base_sizes=[[(8, 8), (10, 10), (12, 12)], [(4, 4), (6, 6), (8, 8)], [(2, 2), (3, 3), (4, 4)]],
            strides=[16, 8, 4],
        ),
        featmap_strides=[16, 8, 4],
    ),
    train_cfg=dict(assigner=dict(type="GridAssigner", pos_iou_thr=0.3, neg_iou_thr=0.3, min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.001,
        conf_thr=0.001,
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
