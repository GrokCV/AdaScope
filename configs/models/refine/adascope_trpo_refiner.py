from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData

from mmdet.registry import MODELS

from .adascope_rl_refiner import PlainTemplateThreeActionRLRefiner


@MODELS.register_module()
class AdaScopeTRPORefiner(nn.Module):
    """Clean TRPO refiner built directly on nn.Module."""

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
        self.policy_head = self.impl.policy_head
        self.value_head = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        self.num_templates = int(self.impl.num_templates)
        self.num_shift_groups = int(self.impl.num_shift_groups)
        self.num_scale_context_groups = int(self.impl.num_scale_context_groups)
        self.dqn_epsilon = 0.0

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
            values=enc['boxes'].new_zeros((0,)),
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
        shifted_boxes = self.apply_shift_groups(enc['boxes'], all_action_raw[..., :2], enc['batch_inds'], batch_data_samples)
        refined_boxes = self.apply_margin_groups(shifted_boxes, all_action_raw[..., 2:6], enc['batch_inds'], batch_data_samples)
        roi_boxes = self.apply_context_groups(refined_boxes, all_action_raw[..., 6], enc['batch_inds'], batch_data_samples)
        best_inds = torch.argmax(logits, dim=-1)
        gather = torch.arange(num_props, device=best_inds.device)
        enc.update(
            scale_context_logits=logits,
            scale_context_probs=probs,
            all_scale_context_action_raw=all_action_raw,
            all_scale_context_refined_boxes=refined_boxes,
            all_scale_context_roi_boxes=roi_boxes,
            best_scale_context_inds=best_inds,
            best_scale_context_boxes=refined_boxes[gather, best_inds],
            best_scale_context_roi_boxes=roi_boxes[gather, best_inds],
            expected_scale_context_boxes=(refined_boxes * probs.unsqueeze(-1)).sum(dim=1),
            expected_scale_context_roi_boxes=(roi_boxes * probs.unsqueeze(-1)).sum(dim=1),
            values=self.value_head(enc['embed']).squeeze(-1),
        )
        return enc

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
        if deterministic:
            action_indices = enc['best_scale_context_inds'][:, None].expand(-1, group_size)
        else:
            action_indices = torch.multinomial(enc['scale_context_probs'], num_samples=group_size, replacement=True)
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

    def evaluate_policy_outputs(
        self,
        feats: Tuple[torch.Tensor, ...],
        proposal_instances: List[InstanceData],
        batch_data_samples: List[InstanceData],
    ) -> Dict[str, torch.Tensor]:
        return self.enumerate_scale_context_groups(feats, proposal_instances, batch_data_samples)

    def evaluate_state_values(
        self,
        feats: Tuple[torch.Tensor, ...],
        proposal_instances: List[InstanceData],
        batch_data_samples: List[InstanceData],
    ) -> Dict[str, torch.Tensor]:
        enc = self.encode(feats, proposal_instances, batch_data_samples)
        enc['values'] = self.value_head(enc['embed']).squeeze(-1)
        return enc

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
        sampled = self.sample_scale_context_groups(feats, proposal_instances, batch_data_samples, group_size=1, deterministic=True)
        return self.boxes_to_instance_list(
            boxes=sampled['refined_boxes'].squeeze(1),
            scores=sampled['scores'],
            labels=sampled['labels'],
            batch_inds=sampled['batch_inds'],
            batch_data_samples=batch_data_samples,
            roi_boxes=sampled['roi_boxes'].squeeze(1),
        )
