import math
from collections import deque
from copy import deepcopy
from typing import Deque, Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData

from mmdet.registry import MODELS

from .adascope_rl_refiner import PlainTemplateThreeActionRLRefiner


@MODELS.register_module()
class AdaScopeSACRefiner(nn.Module):
    """Clean SAC refiner built directly on nn.Module."""

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        feat_channels: int = 256,
        num_ins: int = 5,
        fusion_level: int = 2,
        fusion_type: str = 'concat',
        resize_mode: str = 'bilinear',
        state_roi_size: Tuple[int, int] = (7, 7),
        state_sampling_ratio: int = 2,
        hidden_dim: int = 256,
        min_box_size: float = 2.0,
        template_shift_values: Sequence[float] = (-0.5, -0.25, 0.0, 0.25, 0.5),
        template_shape_width_values: Sequence[float] = (0.75, 1.0, 1.25, 1.5),
        template_shape_height_values: Sequence[float] = (0.75, 1.0, 1.25, 1.5),
        template_rf_expand_values: Sequence[float] = (1.0, 1.25, 1.5, 2.0, 2.5),
        replay_buffer_size: int = 4096,
        replay_batch_size: int = 512,
        target_tau: float = 0.005,
        target_entropy_scale: float = 0.5,
        init_alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.impl = PlainTemplateThreeActionRLRefiner(
            in_channels=in_channels,
            feat_channels=feat_channels,
            num_ins=num_ins,
            fusion_level=fusion_level,
            fusion_type=fusion_type,
            resize_mode=resize_mode,
            state_roi_size=state_roi_size,
            state_sampling_ratio=state_sampling_ratio,
            hidden_dim=hidden_dim,
            min_box_size=min_box_size,
            algorithm='ppo',
            template_shift_values=template_shift_values,
            template_shape_width_values=template_shape_width_values,
            template_shape_height_values=template_shape_height_values,
            template_rf_expand_values=template_rf_expand_values,
        )
        self.hidden_dim = int(hidden_dim)
        self.min_box_size = float(min_box_size)
        self.replay_batch_size = int(replay_batch_size)
        self.target_tau = float(target_tau)
        self.policy_head = self.impl.policy_head

        self.num_templates = int(self.impl.num_templates)
        self.num_shift_groups = int(self.impl.num_shift_groups)
        self.num_scale_context_groups = int(self.impl.num_scale_context_groups)
        self.dqn_epsilon = 0.0

        self.q1_head = nn.Linear(self.hidden_dim, self.num_templates)
        self.q2_head = nn.Linear(self.hidden_dim, self.num_templates)
        nn.init.zeros_(self.q1_head.weight)
        nn.init.zeros_(self.q1_head.bias)
        nn.init.zeros_(self.q2_head.weight)
        nn.init.zeros_(self.q2_head.bias)
        self.q1_target = deepcopy(self.q1_head)
        self.q2_target = deepcopy(self.q2_head)
        for param in self.q1_target.parameters():
            param.requires_grad_(False)
        for param in self.q2_target.parameters():
            param.requires_grad_(False)

        self.log_alpha = nn.Parameter(torch.tensor(math.log(max(init_alpha, 1e-6)), dtype=torch.float32))
        self.target_entropy = float(target_entropy_scale * math.log(max(self.num_templates, 2)))

        self._replay_buffer: Deque[Dict[str, torch.Tensor]] = deque(maxlen=max(int(replay_buffer_size), 1))

    @property
    def template_actions(self):
        return self.impl.template_actions

    @property
    def shift_actions(self):
        return self.impl.shift_actions

    @property
    def shift_group_ids(self):
        return self.impl.shift_group_ids

    @property
    def scale_context_group_ids(self):
        return self.impl.scale_context_group_ids

    def encode(
        self,
        feats: Tuple[torch.Tensor, ...],
        proposal_instances: List[InstanceData],
        batch_data_samples: List[InstanceData],
    ) -> Dict[str, torch.Tensor]:
        return self.impl.encode(feats, proposal_instances, batch_data_samples)

    def enumerate_center_groups(
        self,
        feats: Tuple[torch.Tensor, ...],
        proposal_instances: List[InstanceData],
        batch_data_samples: List[InstanceData],
    ) -> Dict[str, torch.Tensor]:
        return self.impl.enumerate_center_groups(feats, proposal_instances, batch_data_samples)

    def apply_shift_groups(self, *args, **kwargs):
        return self.impl.apply_shift_groups(*args, **kwargs)

    def apply_margin_groups(self, *args, **kwargs):
        return self.impl.apply_margin_groups(*args, **kwargs)

    def apply_context_groups(self, *args, **kwargs):
        return self.impl.apply_context_groups(*args, **kwargs)

    def boxes_to_instance_list(self, *args, **kwargs):
        return self.impl.boxes_to_instance_list(*args, **kwargs)

    def predict_center_instances(self, *args, **kwargs):
        return self.impl.predict_center_instances(*args, **kwargs)

    def _empty_outputs(self, enc: Dict[str, torch.Tensor], group_size: int = 1) -> Dict[str, torch.Tensor]:
        zero_boxes = enc['boxes'].new_zeros((0, group_size, 4))
        zero_scores = enc['boxes'].new_zeros((0, group_size))
        zero_actions = enc['boxes'].new_zeros((0, group_size, 7))
        enc.update(
            scale_context_logits=enc['boxes'].new_zeros((0, self.num_templates)),
            scale_context_probs=enc['boxes'].new_zeros((0, self.num_templates)),
            q1_values=enc['boxes'].new_zeros((0, self.num_templates)),
            q2_values=enc['boxes'].new_zeros((0, self.num_templates)),
            action_indices=enc['batch_inds'].new_zeros((0, group_size)),
            action_raw=zero_actions,
            log_prob=zero_scores,
            entropy=zero_scores,
            refined_boxes=zero_boxes,
            roi_boxes=zero_boxes,
            best_scale_context_inds=enc['batch_inds'].new_zeros((0,)),
            best_scale_context_boxes=enc['boxes'].new_zeros((0, 4)),
            best_scale_context_roi_boxes=enc['boxes'].new_zeros((0, 4)),
            expected_scale_context_boxes=enc['boxes'].new_zeros((0, 4)),
            expected_scale_context_roi_boxes=enc['boxes'].new_zeros((0, 4)),
        )
        return enc

    def enumerate_scale_context_groups(
        self,
        feats: Tuple[torch.Tensor, ...],
        proposal_instances: List[InstanceData],
        batch_data_samples: List[InstanceData],
    ) -> Dict[str, torch.Tensor]:
        enc = self.encode(feats, proposal_instances, batch_data_samples)
        if enc['boxes'].numel() == 0:
            return self._empty_outputs(enc)
        logits = self.policy_head(enc['embed'])
        probs = F.softmax(logits, dim=-1)
        action_bank = self.template_actions.to(device=enc['boxes'].device, dtype=enc['boxes'].dtype)
        num_props = enc['boxes'].shape[0]
        all_action_raw = action_bank.unsqueeze(0).expand(num_props, -1, -1)
        shifted_boxes = self.apply_shift_groups(
            enc['boxes'],
            all_action_raw[..., :2],
            enc['batch_inds'],
            batch_data_samples,
        )
        refined_boxes = self.apply_margin_groups(
            shifted_boxes,
            all_action_raw[..., 2:6],
            enc['batch_inds'],
            batch_data_samples,
        )
        roi_boxes = self.apply_context_groups(
            refined_boxes,
            all_action_raw[..., 6],
            enc['batch_inds'],
            batch_data_samples,
        )
        best_inds = torch.argmax(logits, dim=-1)
        gather = torch.arange(num_props, device=best_inds.device)
        enc.update(
            scale_context_logits=logits,
            scale_context_probs=probs,
            q1_values=self.q1_head(enc['embed'].detach()),
            q2_values=self.q2_head(enc['embed'].detach()),
            all_scale_context_action_raw=all_action_raw,
            all_scale_context_refined_boxes=refined_boxes,
            all_scale_context_roi_boxes=roi_boxes,
            best_scale_context_inds=best_inds,
            best_scale_context_boxes=refined_boxes[gather, best_inds],
            best_scale_context_roi_boxes=roi_boxes[gather, best_inds],
            expected_scale_context_boxes=(refined_boxes * probs.unsqueeze(-1)).sum(dim=1),
            expected_scale_context_roi_boxes=(roi_boxes * probs.unsqueeze(-1)).sum(dim=1),
        )
        return enc

    def _sample_indices(
        self,
        logits: torch.Tensor,
        group_size: int,
        deterministic: bool = False,
        best_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        if logits.numel() == 0:
            return logits.new_zeros((0, group_size), dtype=torch.long)
        if deterministic:
            best = torch.argmax(logits, dim=-1) if best_indices is None else best_indices
            return best[:, None].expand(-1, group_size)
        probs = F.softmax(logits.detach(), dim=-1).to(dtype=torch.float32)
        flat = torch.multinomial(probs, num_samples=group_size, replacement=True)
        return flat.to(device=logits.device, dtype=torch.long)

    def sample_scale_context_groups(
        self,
        feats: Tuple[torch.Tensor, ...],
        proposal_instances: List[InstanceData],
        batch_data_samples: List[InstanceData],
        group_size: int,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        enc = self.enumerate_scale_context_groups(feats, proposal_instances, batch_data_samples)
        group_size = int(group_size)
        if enc['boxes'].numel() == 0:
            return self._empty_outputs(enc, group_size=group_size)
        log_probs = F.log_softmax(enc['scale_context_logits'], dim=-1)
        entropy = -(enc['scale_context_probs'] * log_probs).sum(dim=-1, keepdim=True)
        action_indices = self._sample_indices(
            enc['scale_context_logits'],
            group_size=group_size,
            deterministic=deterministic,
            best_indices=enc['best_scale_context_inds'],
        )
        prop_inds = torch.arange(enc['boxes'].shape[0], device=enc['boxes'].device)[:, None]
        expanded_props = prop_inds.expand(-1, group_size).reshape(-1)
        flat_actions = action_indices.reshape(-1)
        enc.update(
            action_indices=action_indices,
            action_raw=enc['all_scale_context_action_raw'][expanded_props, flat_actions].view(enc['boxes'].shape[0], group_size, 7),
            log_prob=log_probs.gather(1, action_indices),
            entropy=entropy.expand_as(action_indices),
            refined_boxes=enc['all_scale_context_refined_boxes'][expanded_props, flat_actions].view(enc['boxes'].shape[0], group_size, 4),
            roi_boxes=enc['all_scale_context_roi_boxes'][expanded_props, flat_actions].view(enc['boxes'].shape[0], group_size, 4),
        )
        return enc

    def evaluate_scale_context_groups(
        self,
        feats: Tuple[torch.Tensor, ...],
        proposal_instances: List[InstanceData],
        batch_data_samples: List[InstanceData],
        action_raw: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        enc = self.enumerate_scale_context_groups(feats, proposal_instances, batch_data_samples)
        if enc['boxes'].numel() == 0:
            return self._empty_outputs(enc, group_size=int(action_raw.shape[1]))
        bank = self.template_actions.to(device=action_raw.device, dtype=action_raw.dtype)
        flat_actions = action_raw.reshape(-1, action_raw.shape[-1])
        distances = (flat_actions[:, None, :] - bank[None, :, :]).abs().sum(dim=-1)
        action_indices = torch.argmin(distances, dim=1).view(*action_raw.shape[:2])
        log_probs = F.log_softmax(enc['scale_context_logits'], dim=-1)
        entropy = -(enc['scale_context_probs'] * log_probs).sum(dim=-1, keepdim=True)
        group_size = int(action_raw.shape[1])
        prop_inds = torch.arange(enc['boxes'].shape[0], device=enc['boxes'].device)[:, None]
        expanded_props = prop_inds.expand(-1, group_size).reshape(-1)
        flat_indices = action_indices.reshape(-1)
        enc.update(
            action_indices=action_indices,
            action_raw=action_raw,
            log_prob=log_probs.gather(1, action_indices),
            entropy=entropy.expand_as(action_indices),
            refined_boxes=enc['all_scale_context_refined_boxes'][expanded_props, flat_indices].view(enc['boxes'].shape[0], group_size, 4),
            roi_boxes=enc['all_scale_context_roi_boxes'][expanded_props, flat_indices].view(enc['boxes'].shape[0], group_size, 4),
        )
        return enc

    def evaluate_sac_outputs(
        self,
        feats: Tuple[torch.Tensor, ...],
        proposal_instances: List[InstanceData],
        batch_data_samples: List[InstanceData],
    ) -> Dict[str, torch.Tensor]:
        return self.enumerate_scale_context_groups(feats, proposal_instances, batch_data_samples)

    def predict_scale_context_instances(
        self,
        feats: Tuple[torch.Tensor, ...],
        proposal_instances: List[InstanceData],
        batch_data_samples: List[InstanceData],
        use_expected: bool = False,
    ) -> List[InstanceData]:
        if use_expected:
            outputs = self.enumerate_scale_context_groups(feats, proposal_instances, batch_data_samples)
            return self.boxes_to_instance_list(
                boxes=outputs['expected_scale_context_boxes'],
                scores=outputs['scores'],
                labels=outputs['labels'],
                batch_inds=outputs['batch_inds'],
                batch_data_samples=batch_data_samples,
                roi_boxes=outputs['expected_scale_context_roi_boxes'],
            )
        sampled = self.sample_scale_context_groups(
            feats,
            proposal_instances,
            batch_data_samples,
            group_size=1,
            deterministic=True,
        )
        return self.boxes_to_instance_list(
            boxes=sampled['refined_boxes'].squeeze(1),
            scores=sampled['scores'],
            labels=sampled['labels'],
            batch_inds=sampled['batch_inds'],
            batch_data_samples=batch_data_samples,
            roi_boxes=sampled['roi_boxes'].squeeze(1),
        )

    def sac_alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def push_replay(
        self,
        embed: torch.Tensor,
        action_indices: torch.Tensor,
        rewards: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if embed.numel() == 0 or valid_mask.numel() == 0 or not bool(valid_mask.any()):
            return
        valid_embeds = embed[valid_mask].detach().to(device='cpu', dtype=torch.float32)
        valid_actions = action_indices[valid_mask].detach().to(device='cpu', dtype=torch.long)
        valid_rewards = rewards[valid_mask].detach().to(device='cpu', dtype=torch.float32)
        for idx in range(valid_embeds.shape[0]):
            self._replay_buffer.append(
                dict(
                    embed=valid_embeds[idx],
                    action_indices=valid_actions[idx],
                    rewards=valid_rewards[idx],
                ))

    def sample_replay(self, device: torch.device) -> Dict[str, torch.Tensor]:
        if len(self._replay_buffer) == 0:
            return dict(embed=None, action_indices=None, rewards=None)
        batch_size = min(len(self._replay_buffer), self.replay_batch_size)
        if len(self._replay_buffer) == batch_size:
            picked = list(self._replay_buffer)
        else:
            inds = torch.randperm(len(self._replay_buffer))[:batch_size].tolist()
            picked = [self._replay_buffer[ind] for ind in inds]
        return dict(
            embed=torch.stack([item['embed'] for item in picked], dim=0).to(device=device),
            action_indices=torch.stack([item['action_indices'] for item in picked], dim=0).to(device=device),
            rewards=torch.stack([item['rewards'] for item in picked], dim=0).to(device=device),
        )

    def compute_replay_sac_losses(
        self,
        embed: torch.Tensor,
        action_indices: torch.Tensor,
        rewards: torch.Tensor,
        actor_loss_weight: float,
        critic_loss_weight: float,
    ) -> Dict[str, torch.Tensor]:
        logits = self.policy_head(embed)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        q1_values = self.q1_head(embed.detach())
        q2_values = self.q2_head(embed.detach())
        q1_selected = q1_values.gather(1, action_indices)
        q2_selected = q2_values.gather(1, action_indices)
        q_targets = rewards
        loss_critic = 0.5 * (
            F.mse_loss(q1_selected, q_targets, reduction='mean') +
            F.mse_loss(q2_selected, q_targets, reduction='mean')
        )
        min_q = torch.minimum(self.q1_head(embed), self.q2_head(embed))
        alpha = self.sac_alpha()
        entropy = -(probs * log_probs).sum(dim=-1)
        loss_actor = (probs * (alpha.detach() * log_probs - min_q)).sum(dim=-1).mean()
        loss_alpha = -(self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
        return dict(
            sac_loss_actor=float(actor_loss_weight) * loss_actor,
            sac_loss_critic=float(critic_loss_weight) * loss_critic,
            sac_loss_alpha=loss_alpha,
            sac_entropy_mean=entropy.mean().detach(),
            sac_q_mean=min_q.mean().detach(),
            sac_q_selected_mean=0.5 * (q1_selected.mean().detach() + q2_selected.mean().detach()),
            sac_log_prob_mean=log_probs.gather(1, action_indices).mean().detach(),
            sac_alpha=alpha.detach(),
        )

    @torch.no_grad()
    def update_targets(self) -> None:
        tau = self.target_tau
        for target_param, source_param in zip(self.q1_target.parameters(), self.q1_head.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)
        for target_param, source_param in zip(self.q2_target.parameters(), self.q2_head.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)
