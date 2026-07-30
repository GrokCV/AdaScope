# --- 2. 数据集路径和类别 (使用你的路径) ---
data_root = '/opt/data/private/cjt/data/DenseSIRST/SIRSTdevkit'
dataset_type = 'deepir.SIRSTVOCDetClusterDataset'

backend_args = None

core_scale=512

# --- 4. 定义数据流水线 (Pipeline) ---
# 关键: 强制所有图像都变成 1024x1024
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    
    # 强制所有图像都变成 1024x1024
    dict(type='Resize', scale=(core_scale, core_scale), keep_ratio=False),
    # dict(type='Pad', size=(1024, 1024)), # 强制补齐到 1024x1024
    
    # 在 1024x1024 的图像上生成 32x32 的标签
    dict(type='GenerateC5Targets', stride=32), # <-- 你的自定义 Transform
    
    # 打包时, 告诉它要包含我们的新标签
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 
                                         'scale_factor', 'gt_cls_map')) 
]

# 验证集也使用这个 pipeline

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True), # <-- 必须在 Resize 和 Pad 之前
    dict(type='Resize', scale=(core_scale, core_scale), keep_ratio=False),
    # dict(type='Pad', size=(1024, 1024)),
    dict(type='GenerateC5Targets', stride=32),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 
                                         'scale_factor', 'gt_cls_map'))
]

test_pipeline=val_pipeline

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),   
        
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="Splits/trainval_v2.txt",
        data_prefix=dict(sub_data_root=''),
        filter_cfg=dict(filter_empty_gt=False, min_size=0, bbox_min_size=0),
        pipeline=train_pipeline,
        backend_args=backend_args,
        ),
        
        
    
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="Splits/test_v2.txt",
        data_prefix=dict(sub_data_root=''),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader




