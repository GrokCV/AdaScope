_base_ = ['./c5_clean_grpo24_flatdet_threeaction_template3stage_densesirst_disconnected.py']

custom_imports = dict(
    imports=[
        'deepir.datasets.transforms.cluster_gt_targets',
        'deepir.engine.hooks.syn_warmup_sup_grpo_stage_hook',
        'deepir.evaluation.metrics.selective_voc_metric',
        'deepir.models.detectors.plain_flat_sync_dqn_detector',
        'deepir.models.refine.plain_template_three_action_dqn_refiner',
    ],
    allow_failed_imports=False,
)

model = dict(
    type='PlainFlatSyncDQNDetector',
    refiner=dict(
        type='PlainTemplateThreeActionDQNRefiner',
        dqn_epsilon=0.1,
    ),
    rl_group_size=4,
    rl_update_steps=4,
    dqn_loss_weight=1.0,
    rl_use_reference_policy=False,
)

work_dir = '/root/data-tmp/BAFE-Net/work_dirs/c5_clean_plaindqn24_flatdet_threeaction_template3stage_densesirst_disconnected'
