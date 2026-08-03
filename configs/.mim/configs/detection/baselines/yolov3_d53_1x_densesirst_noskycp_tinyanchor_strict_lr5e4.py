_base_ = ["./yolov3_d53_1x_densesirst_noskycp_tinyanchor_strict.py"]

optim_wrapper = dict(
    optimizer=dict(lr=0.0005, momentum=0.9, type="SGD", weight_decay=0.0005),
)
