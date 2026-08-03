_base_ = ['./adazoom_fcos_r50_fpn_densesirst_paper_collab.py']

custom_hooks = [
    dict(
        type='AdaZoomStageHook',
        detector_iters=0,
        policy_iters=2,
        collaborative_policy_iters=1,
        collaborative_detector_iters=1,
        detector_lr=2e-5,
        policy_lr=2e-5,
        collaborative_detector_lr=1e-4,
        collaborative_policy_lr=2e-6,
    ),
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=4,
    val_interval=4,
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=4, save_best='auto', by_epoch=False),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'),
)

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
)

val_dataloader = dict(
    num_workers=0,
    persistent_workers=False,
)

work_dir = '/root/data-tmp/BAFE-Net/work_dirs/_tmp_adazoom_fcos_collab_smoke'
