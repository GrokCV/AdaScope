_base_ = ["./gfl_r50_fpn_1x_densesirst_refdensev3.py"]

train_pipeline = [
    dict(type="deepir.SkyCopyPaste", selected=True, paste_by_box=True),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline,
    )
)
