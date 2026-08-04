"""AdaScope with SAC (Soft Actor-Critic) policy optimizer.

Same coarse-to-fine pipeline as :mod:`AdaScope` (GEDM + ARFR + local FCOS),
with the refiner trained by SAC using a replay buffer. The policy stage is
named ``policy``.

Usage:
    python tools/train_det.py configs/detection/cluster/AdaScope_sac.py
"""
_base_ = ['./AdaScope.py']

custom_imports = dict(
    imports=[
        'deepir.datasets.transforms.adascope_cluster_targets',
        'deepir.engine.hooks.adascope_stage_hook',
        'deepir.evaluation.metrics.selective_voc_metric',
        'deepir.models.detectors.adascope_rl_detector',
        'deepir.models.detectors.adascope_sac_detector',
        'deepir.models.refine.adascope_sac_refiner',
        'deepir.models.cluster_heads.adascope_cluster_head',
    ],
    allow_failed_imports=False,
)

# ── Model: swap the GRPO detector/refiner for the SAC variant ──
model = dict(
    type='AdaScopeSACDetector',
    refiner=dict(type='AdaScopeSACRefiner'),
    rl_algorithm='sac',
    policy_stage_name='policy',
    rl_group_size=4,
    rl_update_steps=4,
    rl_entropy_weight=0.0005,
    rl_advantage_eps=1e-6,
    rl_use_reference_policy=False,
    sac_actor_loss_weight=1.0,
    sac_critic_loss_weight=1.0,
    sac_entropy_alpha=0.2,
)

# ── Hooks: SAC uses the policy-style stage hook ─────────────────
custom_hooks = [
    dict(type='AdaScopePolicyStageHook', warmup_epochs=12,
         refiner_supervised_epochs=4, reference_metric_key='merged_voc/mAP'),
]

work_dir = 'work_dirs/adascope_sac_densesirst'
