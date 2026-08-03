
"""
Train Script:

CUDA_VISIBLE_DEVICES=0 nohup python tools/train_det.py configs/detection/grid/my_c5_validation.py > train.log 2>&1 &

CUDA_VISIBLE_DEVICES=0  python tools/train_det.py configs/detection/grid/my_c5_validation.py 

"""



# --- 1. 基础配置 ---
# 继承 schedules 和 default_runtime, 但数据集我们自己定义
_base_ = [
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py',
    '../_base_/datasets/sirst_cls.py'
]



# --- 3. 定义模型 (与上一轮相同) ---
model = dict(
    type='SingleStageDetector', 
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,), # 关键: 只输出 C5
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    
    neck=None,

   

    bbox_head=dict(
        type='C5ParallelHead', # <-- 你的自定义头
        in_channels=2048, # 如果不用fpn
       
        feat_channels=256,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.75,
            loss_weight=50.0)
    ),
    
    train_cfg=None,
    test_cfg=None
)


# 训练循环配置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1) # 1x schedule
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 学习率调度
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=1.0/3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12, # 对应 1x schedule
        by_epoch=True,
        milestones=[8, 11], # 对应 1x schedule
        gamma=0.1)
]

# --- 7. (可选) 优化器和学习率 ---
# 你可以沿用你 FCOS 的配置，或者使用 _base_ 默认的
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001), # <-- 你的学习率
    clip_grad=None)


val_evaluator = dict(
    type='MyGridMetric',      # <-- 使用你的新评估器
    cls_threshold=0.5         # 你可以调整这个分类阈值
) 
test_evaluator = val_evaluator