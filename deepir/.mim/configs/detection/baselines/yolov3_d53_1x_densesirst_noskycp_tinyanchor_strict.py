_base_ = ["./yolov3_d53_1x_densesirst_noskycp_tuned.py"]

model = dict(
    bbox_head=dict(
        anchor_generator=dict(
            base_sizes=[
                [(12, 12), (16, 16), (20, 20)],
                [(6, 6), (8, 8), (10, 10)],
                [(2, 2), (3, 3), (4, 4)],
            ],
        ),
    ),
    train_cfg=dict(assigner=dict(type="GridAssigner", pos_iou_thr=0.3, neg_iou_thr=0.3, min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.001,
        conf_thr=0.001,
        nms=dict(type="nms", iou_threshold=0.45),
        max_per_img=300,
    ),
)
