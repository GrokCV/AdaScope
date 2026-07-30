_base_ = ['./adazoom_fcos_r50_fpn_densesirst_paper_rl_5k.py']

custom_imports = dict(
    imports=[
        'deepir.datasets',
        'deepir.engine',
        'deepir.models.detectors.adazoom_fcos',
    ],
    allow_failed_imports=False,
)

model = dict(
    use_collaborative_reward=True,
    collaborative_score_power=1.0,
)

custom_hooks = [
    dict(
        type='AdaZoomStageHook',
        detector_iters=0,
        policy_iters=5000,
        collaborative_policy_iters=500,
        collaborative_detector_iters=1000,
        detector_lr=2e-5,
        policy_lr=2e-5,
        collaborative_detector_lr=1e-4,
        collaborative_policy_lr=2e-6,
    ),
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=9500,
    val_interval=500,
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=500, save_best='auto', by_epoch=False),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'),
)

log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)

work_dir = '/root/data-tmp/BAFE-Net/work_dirs/adazoom_fcos_r50_fpn_densesirst_paper_collab'
