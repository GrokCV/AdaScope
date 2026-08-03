"""PPO variant of the AdaScope detector.

Reuses :class:`AdaScopeRLDetector` with the PPO policy optimizer
(GRPO-style group-relative policy optimization with clipped ratios).
"""
from mmdet.registry import MODELS
from .adascope_rl_detector import AdaScopeRLDetector


@MODELS.register_module()
class AdaScopePPODetector(AdaScopeRLDetector):
    """AdaScope detector trained with PPO.

    Same coarse-to-fine pipeline as the base detector; the policy stage is
    named ``grpo`` and losses are computed with PPO clipping.
    """

    def __init__(self, *args, rl_algorithm: str = 'ppo',
                 policy_stage_name: str = 'grpo', **kwargs):
        super().__init__(
            *args, rl_algorithm=rl_algorithm,
            policy_stage_name=policy_stage_name, **kwargs)
