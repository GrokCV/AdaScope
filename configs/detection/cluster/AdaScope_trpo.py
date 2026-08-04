"""AdaScope with TRPO (Trust Region Policy Optimization) policy optimizer.

Same coarse-to-fine pipeline as :mod:`AdaScope` (GEDM + ARFR + local FCOS),
with the refiner trained by TRPO via conjugate gradient + line search on a
KL-constrained surrogate. The policy stage is named ``policy``.

Usage:
    python tools/train_det.py configs/detection/cluster/AdaScope_trpo.py
"""
_base_ = ['./AdaScope.py']

custom_imports = dict(
    imports=[
        'deepir.datasets.transforms.adascope_cluster_targets',
        'deepir.engine.hooks.adascope_stage_hook',
        'deepir.evaluation.metrics.selective_voc_metric',
        'deepir.models.detectors.adascope_rl_detector',
        'deepir.models.detectors.adascope_trpo_detector',
        'deepir.models.refine.adascope_trpo_refiner',
        'deepir.models.cluster_heads.adascope_cluster_head',
    ],
    allow_failed_imports=False,
)

# ── Model: swap the GRPO detector/refiner for the TRPO variant ──
model = dict(
    type='AdaScopeTRPODetector',
    refiner=dict(type='AdaScopeTRPORefiner'),
    rl_algorithm='trpo',
    policy_stage_name='policy',
    rl_group_size=4,
    rl_update_steps=1,
    rl_entropy_weight=0.0005,
    rl_advantage_eps=1e-6,
    rl_use_reference_policy=False,
    trpo_max_kl=0.01,
    trpo_cg_damping=0.01,
    trpo_cg_iters=10,
    trpo_backtrack_iters=10,
    trpo_backtrack_ratio=0.5,
    trpo_accept_ratio=0.1,
    trpo_value_loss_weight=0.5,
)

# ── Hooks: TRPO uses the policy-style stage hook ────────────────
custom_hooks = [
    dict(type='SynWarmupSupPolicyStageHook', warmup_epochs=12,
         refiner_supervised_epochs=4, reference_metric_key='merged_voc/mAP'),
]

work_dir = 'work_dirs/adascope_trpo_densesirst'
