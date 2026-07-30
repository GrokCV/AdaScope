_base_ = ['./c5_clean_grpo24_flatdet_threeaction_template3stage_densesirst_disconnected.py']

custom_imports = dict(
    imports=[
        'deepir.datasets.transforms.cluster_gt_targets',
        'deepir.engine.hooks.syn_warmup_sup_grpo_stage_hook',
        'deepir.evaluation.metrics.selective_voc_metric',
        'deepir.models.detectors.plain_flat_sync_ppo_detector',
        'deepir.models.refine.plain_template_three_action_ppo_refiner',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='PlainFlatSyncPPODetector',
    refiner=dict(type='PlainTemplateThreeActionPPORefiner'),
    rl_group_size=4,
    rl_update_steps=4,
    ppo_clip_eps=0.1,
    rl_entropy_weight=0.0005,
    ppo_value_loss_weight=0.5,
    rl_use_reference_policy=False,
)

work_dir = '/root/data-tmp/BAFE-Net/work_dirs/c5_clean_plainppo24_flatdet_threeaction_template3stage_densesirst_disconnected'
