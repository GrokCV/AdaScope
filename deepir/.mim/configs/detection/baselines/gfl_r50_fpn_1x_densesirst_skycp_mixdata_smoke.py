_base_ = ["./gfl_r50_fpn_1x_densesirst_skycp_mixdata.py"]

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
