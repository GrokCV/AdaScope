"""
Train Script:

CUDA_VISIBLE_DEVICES=0 nohup python tools/train_det.py configs/detection/fcos_changer_seg/fcos_r50-caffe_fpn_gn-head_1x_densesirst.py > train.log 2>&1 &

CUDA_VISIBLE_DEVICES=0  python tools/train_det.py configs/detection/fcos_changer_seg/fcos_r50-caffe_fpn_gn-head_1x_densesirst.py 


"""



_base_ = [
    '../_base_/datasets/sirst_det_seg_voc.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]



device_cpn="cuda:0"

INF = 1e8
# model settings
model = dict(
    type='FCOS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32,
        pad_seg=True),  
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        num_outs=3,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=1,  # 你的小目标类别数，例如 'Target' 就是1类
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        ##把 regress_ranges 下限调低，迎合小目标

        regress_ranges = ((-1, 64), (1, 128), (1, 256),),
        strides=[ 8, 16, 32], # FCOS Head 需要知道它处理的特征图的 stride
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        ## 中心度对小目标不利，会降低小目标影响
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5)
    ),

    # testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# learning rate
param_scheduler = [
    dict(type='ConstantLR', factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[8, 16],
        gamma=0.1)
]

base_lr = 1.0
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='DAdaptAdam', lr=base_lr, weight_decay=0.05,
        decouple=True),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
