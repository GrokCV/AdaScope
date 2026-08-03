_base_ = ['./adazoom_fcos_r50_fpn_densesirst_paper_rl_5k.py']

load_from = '/root/data-tmp/BAFE-Net/work_dirs/adazoom_fcos_r50_fpn_densesirst/epoch_1.pth'
resume = False

custom_hooks = [
    dict(type='AdaZoomIterBootstrapHook', start_iter=1152)
]

work_dir = '/root/data-tmp/BAFE-Net/work_dirs/adazoom_fcos_r50_fpn_densesirst_paper_rl_5k_resume_epoch1'
