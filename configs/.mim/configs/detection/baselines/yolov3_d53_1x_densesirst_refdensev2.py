_base_ = ["./yolov3_d53_1x_densesirst_refdense.py"]

model = dict(
    backbone=dict(out_indices=(2, 3, 4)),
    neck=dict(type="YOLOV3Neck", num_scales=3, in_channels=[512, 256, 128], out_channels=[512, 256, 128]),
    bbox_head=dict(
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type="YOLOAnchorGenerator",
            base_sizes=[[(24, 24), (32, 32), (40, 40)], [(12, 12), (16, 16), (20, 20)], [(4, 4), (6, 6), (8, 8)]],
            strides=[16, 8, 4],
        ),
        featmap_strides=[16, 8, 4],
    ),
    train_cfg=dict(assigner=dict(type="GridAssigner", pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type="nms", iou_threshold=0.45),
        max_per_img=100,
    ),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=35, norm_type=2),
)

