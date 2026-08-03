_base_ = ["./gfl_r50_fpn_1x_densesirst_noskycp_tuned.py"]

model = dict(
    bbox_head=dict(
        reg_max=4,
        loss_cls=dict(beta=2.0, loss_weight=0.5),
    ),
    train_cfg=dict(assigner=dict(type="ATSSAssigner", topk=3)),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.001,
        max_per_img=300,
        nms=dict(type="nms", iou_threshold=0.6),
    ),
)
