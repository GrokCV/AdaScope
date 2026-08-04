"""TRPO (Trust Region Policy Optimization) variant of the AdaScope detector.

Reuses :class:`AdaScopeRLDetector` with the TRPO policy optimizer.
TRPO optimizes the policy with conjugate-gradient + line search on the KL
constrained surrogate objective; the value head is updated by MSE regression.
"""
from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn.functional as F
from mmengine.optim import OptimWrapper
from torch import Tensor

from mmdet.registry import MODELS
from .adascope_rl_detector import AdaScopeRLDetector


@MODELS.register_module()
class AdaScopeTRPODetector(AdaScopeRLDetector):
    """AdaScope detector trained with TRPO.

    The policy stage is named ``policy``. Requires a refiner implementing
    ``sample_scale_context_groups``, ``evaluate_policy_outputs`` and
    ``evaluate_state_values`` (the base three-action template refiner API).
    """

    def __init__(
        self,
        *args,
        rl_algorithm: str = 'trpo',
        policy_stage_name: str = 'policy',
        trpo_max_kl: float = 0.01,
        trpo_cg_damping: float = 0.01,
        trpo_cg_iters: int = 10,
        trpo_backtrack_iters: int = 10,
        trpo_backtrack_ratio: float = 0.5,
        trpo_accept_ratio: float = 0.1,
        trpo_value_loss_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            *args, rl_algorithm=rl_algorithm,
            policy_stage_name=policy_stage_name, **kwargs)
        self.trpo_max_kl = float(trpo_max_kl)
        self.trpo_cg_damping = float(trpo_cg_damping)
        self.trpo_cg_iters = int(trpo_cg_iters)
        self.trpo_backtrack_iters = int(trpo_backtrack_iters)
        self.trpo_backtrack_ratio = float(trpo_backtrack_ratio)
        self.trpo_accept_ratio = float(trpo_accept_ratio)
        self.trpo_value_loss_weight = float(trpo_value_loss_weight)

    # ── TRPO-specific rollout ────────────────────────────────────
    def _collect_plain_rollout(self, batch_inputs: Tensor, batch_data_samples):
        old_refiner = self._clone_refiner(self.refiner)
        with torch.no_grad():
            backbone_feats, det_feats = self._extract_backbone_and_det_feats(batch_inputs)
            det_feats = tuple(feat.detach() for feat in det_feats)
            c5_feats = (backbone_feats[-1].detach(),)
            cluster_logits = self.cluster_head(c5_feats)
            proposal_instances = self._limit_policy_proposals(self._filter_train_proposals(
                self._get_merged_cluster_instances_by_feat(cluster_logits.detach(), batch_data_samples),
                batch_data_samples))
            center_refiner = self._get_center_anchor_refiner()
            center_proposals = center_refiner.predict_center_instances(
                det_feats, proposal_instances, batch_data_samples, use_expected=True)
            old_outputs = old_refiner.sample_scale_context_groups(
                det_feats, center_proposals, batch_data_samples,
                group_size=self.rl_group_size, deterministic=False)
            if old_outputs['boxes'].numel() == 0:
                empty_valid = torch.zeros((0,), dtype=torch.bool, device=batch_inputs.device)
                return dict(det_feats=det_feats, proposal_instances=center_proposals,
                            batch_data_samples=batch_data_samples,
                            action_raw=old_outputs['action_raw'].detach(),
                            old_log_prob=old_outputs['log_prob'].detach(),
                            rewards=old_outputs['log_prob'].detach(),
                            policy_valid_mask=empty_valid)
            assigned_gt, matched_mask = self._match_flat_proposals(
                old_outputs['boxes'], old_outputs['batch_inds'], batch_data_samples)
            assigned_cluster_boxes = self._gather_assigned_cluster_boxes(
                old_outputs['boxes'], old_outputs['batch_inds'],
                batch_data_samples, assigned_gt, matched_mask)
            center_patch_preds = self._predict_external_local_per_roi_stitched(
                batch_inputs, old_outputs['boxes'], old_outputs['batch_inds'],
                batch_data_samples)
            rollout_boxes = old_outputs['roi_boxes']
            refined_patch_preds = self._predict_external_local_per_roi_stitched(
                batch_inputs, rollout_boxes.reshape(-1, 4),
                self._repeat_batch_inds(old_outputs['batch_inds'], self.rl_group_size),
                batch_data_samples)
            reward_outputs = self._compute_patch_rewards(
                old_outputs['boxes'], rollout_boxes, assigned_cluster_boxes,
                center_patch_preds, refined_patch_preds, matched_mask)
            old_policy_outputs = old_refiner.evaluate_policy_outputs(
                det_feats, center_proposals, batch_data_samples)
            rollout = dict(
                det_feats=det_feats,
                proposal_instances=center_proposals,
                batch_data_samples=batch_data_samples,
                states=old_outputs['embed'].detach(),
                old_action_indices=old_outputs['action_indices'].detach(),
                old_logits=old_policy_outputs['scale_context_logits'].detach(),
                action_raw=old_outputs['action_raw'].detach(),
                old_log_prob=old_outputs['log_prob'].detach(),
                rewards=reward_outputs['rewards'].detach(),
                policy_valid_mask=matched_mask.detach(),
                reward_det_gain=reward_outputs['det_gain'].detach(),
                reward_cover_gain=reward_outputs['cover_gain'].detach(),
                reward_cover_penalty=reward_outputs['cover_penalty'].detach(),
                reward_area_penalty=reward_outputs['area_penalty'].detach(),
                reward_count_gain=reward_outputs['count_gain'].detach(),
                reward_purity_gain=reward_outputs['purity_gain'].detach(),
                reward_leak_gain=reward_outputs['leak_gain'].detach(),
                old_values=old_policy_outputs['values'].detach(),
                value_targets=reward_outputs['rewards'].mean(dim=1).detach(),
            )
        return rollout

    # ── TRPO loss dispatch ───────────────────────────────────────
    def _compute_rl_losses(self, rollout: Dict[str, Tensor]):
        if self.rl_algorithm == 'trpo':
            return self._compute_trpo_losses(rollout)
        return super()._compute_rl_losses(rollout)

    def train_step(self, data: Dict, optim_wrapper: OptimWrapper):
        data = self.data_preprocessor(data, True)
        if self.training_stage == 'warmup':
            return self._standard_train_step(data, optim_wrapper)
        if self.training_stage == 'refiner_supervised':
            return self._supervised_refiner_train_step(data, optim_wrapper)
        if self.rl_algorithm == 'trpo':
            return self._trpo_train_step(data, optim_wrapper)
        return super().train_step(data, optim_wrapper)

    # ── TRPO core ────────────────────────────────────────────────
    def _trpo_train_step(self, data: Dict, optim_wrapper: OptimWrapper):
        rollout = self._collect_plain_rollout(data['inputs'], data['data_samples'])
        if rollout['policy_valid_mask'].numel() == 0 or not bool(rollout['policy_valid_mask'].any()):
            zero = self._policy_zero_loss()
            return self._zero_algorithm_logs(zero)
        with optim_wrapper.optim_context(self):
            losses = self._compute_trpo_losses(rollout)
            value_outputs = self.refiner.evaluate_state_values(
                rollout['det_feats'],
                rollout['proposal_instances'],
                rollout['batch_data_samples'],
            )
            valid_mask = rollout['policy_valid_mask']
            value_loss = self.trpo_value_loss_weight * F.mse_loss(
                value_outputs['values'][valid_mask],
                rollout['value_targets'][valid_mask],
                reduction='mean',
            )
        optim_wrapper.zero_grad()
        optim_wrapper.backward(value_loss)
        optim_wrapper.step()
        _, value_log_vars = self.parse_losses({'trpo_loss_value': value_loss})
        log_vars = OrderedDict(value_log_vars)
        for key, value in losses.items():
            if key == 'trpo_loss_value':
                continue
            if isinstance(value, torch.Tensor):
                log_vars[key] = value.detach()
            else:
                log_vars[key] = value
        return log_vars

    def _compute_trpo_losses(self, rollout: Dict[str, Tensor]):
        policy_outputs, zero, valid_group_mask, advantages, surrogate, mean_kl = self._surrogate_and_kl(rollout)
        if surrogate is None:
            return self._zero_algorithm_logs(zero)
        policy_params = self._policy_params()
        loss_policy = -surrogate
        grads = torch.autograd.grad(loss_policy, policy_params, retain_graph=True)
        loss_grad = self._flatten_tensors(grads).detach()
        step_dir = self._conjugate_gradient(
            lambda v: self._fisher_vector_product(rollout, v), -loss_grad)
        fvp_step = self._fisher_vector_product(rollout, step_dir)
        shs = 0.5 * torch.dot(step_dir, fvp_step)
        if shs <= 0:
            return self._zero_algorithm_logs(zero)
        scale = torch.sqrt(torch.tensor(2.0 * self.trpo_max_kl,
                                        device=step_dir.device) / (shs + 1e-8))
        full_step = step_dir * scale
        old_params = self._get_flat_params(policy_params)
        old_surrogate = surrogate.detach()
        accepted = False
        final_kl = mean_kl.detach()
        final_surrogate = surrogate.detach()
        for backtrack_idx in range(self.trpo_backtrack_iters):
            step_frac = self.trpo_backtrack_ratio ** backtrack_idx
            self._set_flat_params(policy_params, old_params + step_frac * full_step)
            _, _, _, _, new_surrogate, new_kl = self._surrogate_and_kl(rollout)
            if (torch.isfinite(new_surrogate) and torch.isfinite(new_kl)
                    and new_kl <= self.trpo_max_kl
                    and (new_surrogate.detach() - old_surrogate) > 0):
                accepted = True
                final_kl = new_kl.detach()
                final_surrogate = new_surrogate.detach()
                break
        if not accepted:
            self._set_flat_params(policy_params, old_params)
        valid_mask = rollout['policy_valid_mask']
        entropy = -(policy_outputs['scale_context_probs'][valid_mask] * F.log_softmax(
            policy_outputs['scale_context_logits'][valid_mask], dim=-1)).sum(dim=-1).mean()
        losses = dict(
            trpo_loss_policy=-final_surrogate,
            trpo_loss_entropy=-self.rl_entropy_weight * entropy,
            trpo_kl_mean=final_kl,
            trpo_step_norm=full_step.norm().detach(),
            trpo_advantage_mean=advantages[valid_group_mask].mean().detach(),
            trpo_line_search_success=entropy.detach().new_tensor(1.0 if accepted else 0.0),
        )
        losses.update(self._common_rl_stats(
            rollout, valid_group_mask,
            losses['trpo_line_search_success'].detach(),
            zero.detach().new_zeros(())))
        return losses

    def _surrogate_and_kl(self, rollout: Dict[str, Tensor]):
        policy_outputs = self.refiner.evaluate_policy_outputs(
            rollout['det_feats'],
            rollout['proposal_instances'],
            rollout['batch_data_samples'],
        )
        valid_mask = rollout['policy_valid_mask']
        zero = self._policy_zero_loss(policy_outputs['scale_context_logits'])
        if valid_mask.numel() == 0 or not bool(valid_mask.any()):
            return policy_outputs, zero, None, None, None, None
        valid_group_mask = valid_mask[:, None].expand_as(rollout['rewards'])
        advantages = self._normalize_advantages(
            rollout['rewards'], rollout['old_values'], valid_mask)
        new_log_probs_all = F.log_softmax(policy_outputs['scale_context_logits'], dim=-1)
        new_log_prob = new_log_probs_all.gather(1, rollout['old_action_indices'])[valid_group_mask]
        old_log_prob = rollout['old_log_prob'][valid_group_mask]
        ratio = torch.exp(new_log_prob - old_log_prob)
        surrogate = (ratio * advantages[valid_group_mask]).mean()
        old_log_probs_full = F.log_softmax(rollout['old_logits'][valid_mask], dim=-1)
        old_probs_full = old_log_probs_full.exp()
        new_log_probs_full = F.log_softmax(
            policy_outputs['scale_context_logits'][valid_mask], dim=-1)
        mean_kl = (old_probs_full * (old_log_probs_full - new_log_probs_full)).sum(dim=-1).mean()
        return policy_outputs, zero, valid_group_mask, advantages, surrogate, mean_kl

    def _conjugate_gradient(self, fvp_fn, b: Tensor):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)
        for _ in range(self.trpo_cg_iters):
            z = fvp_fn(p)
            denom = torch.dot(p, z)
            if denom <= 0:
                break
            alpha = rdotr / (denom + 1e-8)
            x = x + alpha * p
            r = r - alpha * z
            new_rdotr = torch.dot(r, r)
            if new_rdotr.sqrt() < 1e-10:
                break
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def _fisher_vector_product(self, rollout: Dict[str, Tensor], vector: Tensor):
        _, _, _, _, _, mean_kl = self._surrogate_and_kl(rollout)
        policy_params = self._policy_params()
        grads = torch.autograd.grad(mean_kl, policy_params, create_graph=True, retain_graph=True)
        flat_grads = self._flatten_tensors(grads)
        grad_vector = (flat_grads * vector).sum()
        hvp = torch.autograd.grad(grad_vector, policy_params, retain_graph=True)
        return self._flatten_tensors(hvp).detach() + self.trpo_cg_damping * vector

    @staticmethod
    def _flatten_tensors(tensors):
        return (torch.cat([tensor.reshape(-1) for tensor in tensors])
                if len(tensors) > 0 else torch.zeros((0,)))

    @staticmethod
    def _get_flat_params(params):
        return torch.cat([param.data.reshape(-1) for param in params])

    @staticmethod
    def _set_flat_params(params, flat_params):
        offset = 0
        for param in params:
            numel = param.numel()
            param.data.copy_(flat_params[offset:offset + numel].view_as(param))
            offset += numel

    def _policy_params(self):
        return list(self.refiner.policy_head.parameters())
