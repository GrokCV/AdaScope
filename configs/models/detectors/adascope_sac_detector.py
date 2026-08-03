"""SAC (Soft Actor-Critic) variant of the AdaScope detector.

Reuses :class:`AdaScopeRLDetector` with the SAC policy optimizer.
The SAC refiner maintains a replay buffer; transitions are pushed by
:meth:`_maybe_push_replay` and losses are computed with replay sampling.
"""
from typing import Dict

from torch import Tensor

from mmdet.registry import MODELS
from .adascope_rl_detector import AdaScopeRLDetector


@MODELS.register_module()
class AdaScopeSACDetector(AdaScopeRLDetector):
    """AdaScope detector trained with SAC.

    The policy stage is named ``policy``. The refiner must implement the SAC
    interface: ``evaluate_sac_outputs``, ``push_replay``, ``sample_replay``,
    ``compute_replay_sac_losses`` and ``update_targets``.
    """

    def __init__(self, *args, rl_algorithm: str = 'sac',
                 policy_stage_name: str = 'policy', **kwargs):
        super().__init__(
            *args, rl_algorithm=rl_algorithm,
            policy_stage_name=policy_stage_name, **kwargs)

    def maybe_update_policy_reference(self, score: float, epoch: int) -> bool:
        """Cache the best refiner as the reference policy for SAC."""
        if not self.rl_use_reference_policy or float(score) <= self._rl_reference_score:
            return False
        self._rl_reference_refiner = self._clone_refiner(self.refiner)
        self._rl_reference_score = float(score)
        self._rl_reference_epoch = int(epoch)
        return True

    def _maybe_push_replay(self, old_outputs, reward_outputs, matched_mask) -> None:
        if hasattr(self.refiner, 'push_replay'):
            self.refiner.push_replay(
                embed=old_outputs['embed'],
                action_indices=old_outputs['action_indices'],
                rewards=reward_outputs['rewards'],
                valid_mask=matched_mask,
            )

    def _compute_sac_losses(self, rollout: Dict[str, Tensor]):
        sac_outputs = self.refiner.evaluate_sac_outputs(
            rollout['det_feats'],
            rollout['proposal_instances'],
            rollout['batch_data_samples'],
        )
        zero = self._policy_zero_loss(sac_outputs['scale_context_logits'])
        valid_mask = rollout['policy_valid_mask']
        if valid_mask.numel() == 0 or not bool(valid_mask.any()):
            return self._zero_algorithm_logs(zero)
        replay_batch = self.refiner.sample_replay(
            device=sac_outputs['scale_context_logits'].device)
        if replay_batch['embed'] is None:
            replay_embed = rollout['old_embed'][valid_mask]
            replay_actions = rollout['old_action_indices'][valid_mask]
            replay_rewards = rollout['rewards'][valid_mask]
        else:
            replay_embed = replay_batch['embed']
            replay_actions = replay_batch['action_indices']
            replay_rewards = replay_batch['rewards']
        losses = self.refiner.compute_replay_sac_losses(
            embed=replay_embed,
            action_indices=replay_actions,
            rewards=replay_rewards,
            actor_loss_weight=self.sac_actor_loss_weight,
            critic_loss_weight=self.sac_critic_loss_weight,
        )
        self.refiner.update_targets()
        clipfrac = zero.detach().new_zeros(())
        loss_kl, approx_kl_mean, approx_kl_raw_mean = self._compute_reference_kl(
            sac_outputs['scale_context_logits'], rollout, zero)
        losses.update(dict(
            sac_loss_kl=loss_kl,
            sac_approx_kl=approx_kl_mean,
            sac_approx_kl_raw=approx_kl_raw_mean,
        ))
        valid_group_mask = valid_mask[:, None].expand_as(rollout['rewards'])
        losses.update(self._common_rl_stats(
            rollout, valid_group_mask,
            losses['sac_loss_actor'].detach().new_ones(()), clipfrac))
        return losses
