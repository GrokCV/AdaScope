_base_ = [
    './fcos_changer_seg_r50-caffe_fpn_gn-head_1x_densesirst.py'
]

model = dict(
    neck=dict(num_outs=5),
    bbox_head=dict(
        type='deepir.FCOSChangerSegHead',
        strides=[8, 16, 32, 64, 128],
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, 100000000.0)),
    ),
)

work_dir = '/root/data-tmp/BAFE-Net/work_dirs/fcos_changer_seg_r50-caffe_fpn_gn-head_1x_densesirst_5lvl'
