_base_ = ['./adazoom_fcos_r50_fpn_densesirst_paper_collab.py']

load_from = '/root/data-tmp/BAFE-Net/work_dirs/adazoom_fcos_r50_fpn_densesirst_paper_rl_5k/iter_5000.pth'
resume = False

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
    dict(
        type='AdaZoomIterBootstrapHook',
        start_iter=5000,
    ),
]

work_dir = '/root/data-tmp/BAFE-Net/work_dirs/adazoom_fcos_r50_fpn_densesirst_paper_collab_resume_iter5000'
