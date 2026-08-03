_base_ = ['./adazoom_fcos_r50_fpn_densesirst.py']

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
    detector_patch_topk_actions=1,
)

custom_hooks = [
    dict(
        type='AdaZoomStageHook',
        detector_iters=0,
        policy_iters=13824,
        collaborative_policy_iters=500,
        collaborative_detector_iters=1000,
        detector_lr=1.0,
        policy_lr=1.0,
        collaborative_detector_lr=1e-1,
        collaborative_policy_lr=1e-1,
    ),
]

load_from = '/root/data-tmp/BAFE-Net/work_dirs/adazoom_fcos_r50_fpn_densesirst/epoch_12.pth'
resume = True

train_dataloader = dict(
    batch_size=1,
)

work_dir = '/root/data-tmp/BAFE-Net/work_dirs/adazoom_fcos_r50_fpn_densesirst_epoch_collab_resume_epoch12'
