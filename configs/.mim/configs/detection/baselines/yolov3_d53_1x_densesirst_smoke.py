_base_ = ["./yolov3_d53_1x_densesirst_refdensev3.py"]

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            times=1,
            dataset=dict(ann_file="Splits/mini_trainval.txt"),
        )
    )
)

val_dataloader = dict(
    dataset=dict(
        ann_file="Splits/mini_test.txt",
    )
)
test_dataloader = val_dataloader

train_cfg = dict(max_epochs=1, val_interval=1)
