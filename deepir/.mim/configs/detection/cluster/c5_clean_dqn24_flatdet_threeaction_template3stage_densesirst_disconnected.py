_base_ = ['./c5_clean_grpo24_flatdet_threeaction_template3stage_densesirst_disconnected.py']

custom_imports = dict(
    imports=[
        'deepir.datasets.transforms.cluster_gt_targets',
        'deepir.engine.hooks.syn_warmup_sup_grpo_stage_hook',
        'deepir.models.detectors.flat_sync_clean_grpo_detector',
        'deepir.models.refine.fpn_template_three_action_grpo_refiner',
        'deepir.evaluation.metrics.selective_voc_metric',
        'deepir.models.detectors.flat_sync_clean_rl_detector',
        'deepir.models.refine.fpn_template_three_action_rl_refiner',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='FlatSyncCleanDQNDetector',
    refiner=dict(
        type='FPNTemplateThreeActionDQNRefiner',
        dqn_epsilon=0.1,
    ),
    grpo_use_reference_policy=False,
    grpo_ref_kl_weight=0.0,
    dqn_loss_weight=1.0,
)

work_dir = '/root/data-tmp/BAFE-Net/work_dirs/c5_clean_dqn24_flatdet_threeaction_template3stage_densesirst_disconnected'
