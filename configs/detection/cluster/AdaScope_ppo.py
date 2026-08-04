"""AdaScope with PPO policy optimizer.

Same coarse-to-fine pipeline as :mod:`AdaScope` (GEDM + ARFR + local FCOS),
with the refiner trained by PPO (group-relative clipped policy optimization).
The policy stage is named ``grpo``.

Usage:
    python tools/train_det.py configs/detection/cluster/AdaScope_ppo.py
"""
_base_ = ['./AdaScope.py']

custom_imports = dict(
    imports=[
        'deepir.datasets.transforms.adascope_cluster_targets',
        'deepir.engine.hooks.adascope_stage_hook',
        'deepir.evaluation.metrics.selective_voc_metric',
        'deepir.models.detectors.adascope_rl_detector',
        'deepir.models.detectors.adascope_ppo_detector',
        'deepir.models.refine.adascope_ppo_refiner',
        'deepir.models.cluster_heads.adascope_cluster_head',
    ],
    allow_failed_imports=False,
)

# ── Model: swap the GRPO detector/refiner for the PPO variant ──
model = dict(
    type='AdaScopePPODetector',
    refiner=dict(type='AdaScopePPORefiner'),
    rl_algorithm='ppo',
    policy_stage_name='grpo',
    rl_group_size=4,
    rl_update_steps=4,
    ppo_clip_eps=0.1,
    rl_entropy_weight=0.0005,
    ppo_value_loss_weight=0.5,
    rl_use_reference_policy=False,
)

work_dir = 'work_dirs/adascope_ppo_densesirst'
