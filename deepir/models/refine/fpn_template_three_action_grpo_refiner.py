from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torchvision.ops import roi_align

from mmdet.registry import MODELS


@MODELS.register_module()
class FPNTemplateThreeActionGRPORefiner(nn.Module):
    """Three-action discrete refiner for DenseSIRST cluster proposals.

    Direct inheritance from nn.Module (not through FPNTemplateGRPORefiner).
    """

    policy_mode = 'template_categorical_three_action'

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

        self.template_shape_width_values = tuple(float(v) for v in template_shape_width_values)
        self.template_shape_height_values = tuple(float(v) for v in template_shape_height_values)
        self.template_rf_expand_values = tuple(float(v) for v in template_rf_expand_values)

        if len(self.template_shape_width_values) == 0:
            raise ValueError('template_shape_width_values must not be empty.')
        if len(self.template_shape_height_values) == 0:
            raise ValueError('template_shape_height_values must not be empty.')
        if len(self.template_rf_expand_values) == 0:
            raise ValueError('template_rf_expand_values must not be empty.')
        if min(self.template_shape_width_values) <= 0.0:
            raise ValueError('template_shape_width_values must be positive.')
        if min(self.template_shape_height_values) <= 0.0:
            raise ValueError('template_shape_height_values must be positive.')
        if min(self.template_rf_expand_values) < 1.0:
            raise ValueError('template_rf_expand_values must be >= 1.0.')

        self.fusion_level = int(fusion_level)
        self.fusion_type = str(fusion_type).lower()
        self.resize_mode = str(resize_mode)
        self.state_roi_size = (int(state_roi_size[0]), int(state_roi_size[1]))
        self.state_sampling_ratio = int(state_sampling_ratio)
        self.hidden_dim = int(hidden_dim)
        self.min_box_size = float(min_box_size)

        if isinstance(in_channels, int):
            input_channels = tuple(int(in_channels) for _ in range(int(num_ins)))
        else:
            input_channels = tuple(int(c) for c in in_channels)
            num_ins = len(input_channels)
        self.num_ins = int(num_ins)
        self.input_channels = input_channels

        self.template_shift_values = tuple(float(v) for v in template_shift_values)

        if self.num_ins <= 0:
            raise ValueError('num_ins must be positive.')
        if not (0 <= self.fusion_level < self.num_ins):
            raise ValueError(f'fusion_level must be in [0, {self.num_ins - 1}], got {self.fusion_level}.')
        if self.fusion_type not in ('sum', 'concat'):
            raise ValueError("fusion_type must be 'sum' or 'concat'.")

        self.max_center_shift = max(abs(v) for v in self.template_shift_values)
        self.max_margin_scale = 0.5 * max(
            max(abs(v - 1.0) for v in self.template_shape_width_values),
            max(abs(v - 1.0) for v in self.template_shape_height_values),
        )

        self.input_proj = nn.ModuleList([
            nn.Conv2d(in_ch, feat_channels, 1) for in_ch in self.input_channels
        ])
        if self.fusion_type == 'concat':
            self.fuse = nn.Sequential(
                nn.Conv2d(feat_channels * self.num_ins, feat_channels, 1),
                nn.ReLU(inplace=True),
            )
        else:
            self.fuse = None

        self.shared = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        meta_dim = 8
        self.state_mlp = nn.Sequential(
            nn.Linear(feat_channels + meta_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self._build_template_bank()

        self.policy_head = nn.Linear(hidden_dim, self.num_templates)
        nn.init.zeros_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

    def _build_template_bank(self) -> torch.Tensor:
        actions = []
        shift_ids = []
        context_ids = []
        shape_ids = []
        scale_context_ids = []
        scale_context_meta = []
        scale_context_action_raw = []

        num_shift = len(self.template_shift_values)
        num_shape_w = len(self.template_shape_width_values)
        num_shape_h = len(self.template_shape_height_values)
        num_shape = num_shape_w * num_shape_h

        for rf_idx, rf_expand in enumerate(self.template_rf_expand_values):
            context_factor = float(rf_expand - 1.0)
            for h_idx, shape_h in enumerate(self.template_shape_height_values):
                for w_idx, shape_w in enumerate(self.template_shape_width_values):
                    shape_id = h_idx * num_shape_w + w_idx
                    shape_rf_id = rf_idx * num_shape + shape_id
                    margin_x = 0.5 * (float(shape_w) - 1.0)
                    margin_y = 0.5 * (float(shape_h) - 1.0)

                    scale_context_meta.append([float(shape_w), float(shape_h), context_factor])
                    scale_context_action_raw.append([0.0, 0.0, margin_x, margin_y, margin_x, margin_y, context_factor])

                    for shift_y_idx, shift_y in enumerate(self.template_shift_values):
                        for shift_x_idx, shift_x in enumerate(self.template_shift_values):
                            shift_id = shift_y_idx * num_shift + shift_x_idx
                            actions.append([float(shift_x), float(shift_y), margin_x, margin_y, margin_x, margin_y, context_factor])
                            shift_ids.append(shift_id)
                            context_ids.append(rf_idx)
                            shape_ids.append(shape_id)
                            scale_context_ids.append(shape_rf_id)

        template_actions = torch.tensor(actions, dtype=torch.float32)
        self.register_buffer('template_actions', template_actions, persistent=True)
        self.num_templates = int(template_actions.shape[0])

        shift_actions = torch.tensor(
            [[float(shift_x), float(shift_y)] for shift_y in self.template_shift_values for shift_x in self.template_shift_values],
            dtype=torch.float32,
        )
        self.register_buffer('shift_actions', shift_actions, persistent=True)
        self.num_shift_groups = int(shift_actions.shape[0])
        self.register_buffer('shift_group_ids', torch.tensor(shift_ids, dtype=torch.long), persistent=True)

        context_actions = torch.tensor([float(v - 1.0) for v in self.template_rf_expand_values], dtype=torch.float32)
        self.register_buffer('context_actions', context_actions, persistent=True)
        self.num_context_groups = int(context_actions.shape[0])
        self.register_buffer('context_group_ids', torch.tensor(context_ids, dtype=torch.long), persistent=True)

        shape_actions = torch.tensor(
            [[float(shape_w), float(shape_h)] for shape_h in self.template_shape_height_values for shape_w in self.template_shape_width_values],
            dtype=torch.float32,
        )
        self.register_buffer('shape_actions', shape_actions, persistent=True)
        self.num_shape_groups = int(shape_actions.shape[0])
        self.register_buffer('shape_group_ids', torch.tensor(shape_ids, dtype=torch.long), persistent=True)

        scale_context_actions = torch.tensor(scale_context_meta, dtype=torch.float32)
        self.register_buffer('scale_context_actions', scale_context_actions, persistent=True)
        self.num_scale_context_groups = int(scale_context_actions.shape[0])
        self.register_buffer('scale_context_group_ids', torch.tensor(scale_context_ids, dtype=torch.long), persistent=True)
        self.register_buffer('scale_context_action_raw_bank', torch.tensor(scale_context_action_raw, dtype=torch.float32), persistent=True)
        return template_actions

    @staticmethod
    def _expand_boxes_for_groups(boxes: torch.Tensor, num_groups: int) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.new_zeros((0, num_groups, 4))
        return boxes[:, None, :].expand(-1, num_groups, -1)

    def _aggregate_logits_by_group(self, logits: torch.Tensor, group_ids: torch.Tensor, num_groups: int) -> torch.Tensor:
        if logits.numel() == 0:
            return logits.new_zeros((0, num_groups))
        group_ids = group_ids.to(device=logits.device)
        parts = []
        for group_idx in range(num_groups):
            mask = group_ids == group_idx
            if bool(mask.any()):
                parts.append(torch.logsumexp(logits[:, mask], dim=1))
            else:
                parts.append(logits.new_full((logits.shape[0],), float('-inf')))
        return torch.stack(parts, dim=1)

    def _resize_feat(self, feat: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if feat.shape[-2:] == size:
            return feat
        if self.resize_mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
            return F.interpolate(feat, size=size, mode=self.resize_mode, align_corners=False)
        return F.interpolate(feat, size=size, mode=self.resize_mode)

    def _fuse_feats(self, feats: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(feats) < self.num_ins:
            raise ValueError(f'Expected at least {self.num_ins} FPN levels, got {len(feats)}.')
        feats = feats[:self.num_ins]
        target_size = feats[self.fusion_level].shape[-2:]
        resized = [self._resize_feat(proj(feat), target_size) for feat, proj in zip(feats, self.input_proj)]
        if self.fusion_type == 'concat':
            fused = self.fuse(torch.cat(resized, dim=1))
        else:
            fused = torch.stack(resized, dim=0).sum(dim=0)
        return self.shared(fused)

    @staticmethod
    def _empty_instance(device: torch.device) -> InstanceData:
        inst = InstanceData()
        inst.bboxes = torch.zeros((0, 4), device=device, dtype=torch.float32)
        inst.scores = torch.zeros((0,), device=device, dtype=torch.float32)
        inst.labels = torch.zeros((0,), device=device, dtype=torch.long)
        inst.roi_bboxes = torch.zeros((0, 4), device=device, dtype=torch.float32)
        return inst

    def _build_policy_inputs(
        self,
        proposal_instances: List[InstanceData],
        batch_data_samples: List[InstanceData],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        boxes, scores, labels, batch_inds = [], [], [], []
        for img_idx, (proposal, _) in enumerate(zip(proposal_instances, batch_data_samples)):
            if proposal.bboxes.numel() == 0:
                continue
            boxes.append(proposal.bboxes.to(device=device, dtype=dtype))
            scores.append(proposal.scores.to(device=device, dtype=dtype))
            labels.append(proposal.labels.to(device=device, dtype=torch.long))
            batch_inds.append(torch.full((proposal.bboxes.shape[0],), img_idx, dtype=torch.long, device=device))
        if len(boxes) == 0:
            return dict(
                boxes=torch.zeros((0, 4), device=device, dtype=dtype),
                scores=torch.zeros((0,), device=device, dtype=dtype),
                labels=torch.zeros((0,), device=device, dtype=torch.long),
                batch_inds=torch.zeros((0,), device=device, dtype=torch.long),
            )
        return dict(
            boxes=torch.cat(boxes, dim=0),
            scores=torch.cat(scores, dim=0),
            labels=torch.cat(labels, dim=0),
            batch_inds=torch.cat(batch_inds, dim=0),
        )

    def _build_box_meta(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        batch_inds: torch.Tensor,
        batch_data_samples: List[InstanceData],
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.new_zeros((0, 8))
        meta = boxes.new_zeros((boxes.shape[0], 8))
        for img_idx, data_sample in enumerate(batch_data_samples):
            mask = batch_inds == img_idx
            if not bool(mask.any()):
                continue
            img_h, img_w = data_sample.img_shape[:2]
            img_boxes = boxes[mask]
            img_scores = scores[mask]
            cx = 0.5 * (img_boxes[:, 0] + img_boxes[:, 2])
            cy = 0.5 * (img_boxes[:, 1] + img_boxes[:, 3])
            bw = (img_boxes[:, 2] - img_boxes[:, 0]).clamp(min=1.0)
            bh = (img_boxes[:, 3] - img_boxes[:, 1]).clamp(min=1.0)
            area = (bw * bh) / max(float(img_w * img_h), 1.0)
            aspect = bw / bh.clamp(min=1.0)
            meta[mask, 0] = cx / max(float(img_w), 1.0)
            meta[mask, 1] = cy / max(float(img_h), 1.0)
            meta[mask, 2] = bw / max(float(img_w), 1.0)
            meta[mask, 3] = bh / max(float(img_h), 1.0)
            meta[mask, 4] = area
            meta[mask, 5] = torch.log(aspect.clamp(min=1e-6))
            meta[mask, 6] = (img_boxes[:, 0] / max(float(img_w), 1.0)).clamp(min=0.0, max=1.0)
            meta[mask, 7] = img_scores.clamp(min=0.0, max=1.0)
        return meta

    @staticmethod
    def _boxes_to_rois(boxes: torch.Tensor, batch_inds: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.new_zeros((0, 5))
        return torch.cat([batch_inds[:, None].to(dtype=boxes.dtype), boxes], dim=1)

    def _roi_state_embed(
        self,
        fused_feat: torch.Tensor,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        batch_inds: torch.Tensor,
        batch_data_samples: List[InstanceData],
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return fused_feat.new_zeros((0, self.hidden_dim))
        rois = self._boxes_to_rois(boxes, batch_inds)
        meta = self._build_box_meta(boxes, scores, batch_inds, batch_data_samples)
        img_h, img_w = batch_data_samples[0].img_shape[:2]
        spatial_scale = fused_feat.shape[-1] / max(float(img_w), 1.0)
        roi_feat = roi_align(
            fused_feat, rois, output_size=self.state_roi_size,
            spatial_scale=spatial_scale, sampling_ratio=self.state_sampling_ratio, aligned=True
        )
        pooled = F.adaptive_avg_pool2d(roi_feat, 1).flatten(1)
        return self.state_mlp(torch.cat([pooled, meta], dim=1))

    def encode(
        self,
        feats: Tuple[torch.Tensor, ...],
        proposal_instances: List[InstanceData],
        batch_data_samples: List[InstanceData],
    ) -> Dict[str, torch.Tensor]:
        fused_feat = self._fuse_feats(feats)
        policy_inputs = self._build_policy_inputs(
            proposal_instances=proposal_instances,
            batch_data_samples=batch_data_samples,
            dtype=fused_feat.dtype,
            device=fused_feat.device
        )
        embed = self._roi_state_embed(
            fused_feat=fused_feat,
            boxes=policy_inputs['boxes'],
            scores=policy_inputs['scores'],
            batch_inds=policy_inputs['batch_inds'],
            batch_data_samples=batch_data_samples
        )
        policy_inputs['embed'] = embed
        policy_inputs['fused_feat'] = fused_feat
        return policy_inputs

    def apply_shift_groups(
        self,
        boxes: torch.Tensor,
        offsets: torch.Tensor,
        batch_inds: torch.Tensor,
        batch_data_samples: List[InstanceData],
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.new_zeros(offsets.shape[:2] + (4,))
        x1 = boxes[:, 0][:, None]
        y1 = boxes[:, 1][:, None]
        x2 = boxes[:, 2][:, None]
        y2 = boxes[:, 3][:, None]
        bw = (x2 - x1).clamp(min=self.min_box_size)
        bh = (y2 - y1).clamp(min=self.min_box_size)
        cx = 0.5 * (x1 + x2) + offsets[..., 0] * bw
        cy = 0.5 * (y1 + y2) + offsets[..., 1] * bh
        shift_x1 = cx - 0.5 * bw
        shift_y1 = cy - 0.5 * bh
        shift_x2 = cx + 0.5 * bw
        shift_y2 = cy + 0.5 * bh
        if batch_inds.numel() > 0:
            img_heights = shift_x1.new_tensor([float(d.img_shape[0]) for d in batch_data_samples])
            img_widths = shift_x1.new_tensor([float(d.img_shape[1]) for d in batch_data_samples])
            max_h = img_heights.index_select(0, batch_inds).unsqueeze(1)
            max_w = img_widths.index_select(0, batch_inds).unsqueeze(1)
            shift_x1 = torch.minimum(shift_x1.clamp(min=0.0), max_w)
            shift_y1 = torch.minimum(shift_y1.clamp(min=0.0), max_h)
            shift_x2 = torch.minimum(shift_x2.clamp(min=0.0), max_w)
            shift_y2 = torch.minimum(shift_y2.clamp(min=0.0), max_h)
        shift_x2 = torch.maximum(shift_x2, shift_x1 + self.min_box_size)
        shift_y2 = torch.maximum(shift_y2, shift_y1 + self.min_box_size)
        return torch.stack([shift_x1, shift_y1, shift_x2, shift_y2], dim=-1)

    def apply_margin_groups(
        self,
        shifted_boxes: torch.Tensor,
        margins: torch.Tensor,
        batch_inds: torch.Tensor,
        batch_data_samples: List[InstanceData],
    ) -> torch.Tensor:
        if shifted_boxes.numel() == 0:
            return shifted_boxes.new_zeros(margins.shape[:2] + (4,))
        x1 = shifted_boxes[..., 0]
        y1 = shifted_boxes[..., 1]
        x2 = shifted_boxes[..., 2]
        y2 = shifted_boxes[..., 3]
        bw = (x2 - x1).clamp(min=self.min_box_size)
        bh = (y2 - y1).clamp(min=self.min_box_size)
        ref_x1 = x1 - margins[..., 0] * bw
        ref_y1 = y1 - margins[..., 1] * bh
        ref_x2 = x2 + margins[..., 2] * bw
        ref_y2 = y2 + margins[..., 3] * bh
        if batch_inds.numel() > 0:
            img_heights = ref_x1.new_tensor([float(d.img_shape[0]) for d in batch_data_samples])
            img_widths = ref_x1.new_tensor([float(d.img_shape[1]) for d in batch_data_samples])
            max_h = img_heights.index_select(0, batch_inds).unsqueeze(1)
            max_w = img_widths.index_select(0, batch_inds).unsqueeze(1)
            ref_x1 = torch.minimum(ref_x1.clamp(min=0.0), max_w)
            ref_y1 = torch.minimum(ref_y1.clamp(min=0.0), max_h)
            ref_x2 = torch.minimum(ref_x2.clamp(min=0.0), max_w)
            ref_y2 = torch.minimum(ref_y2.clamp(min=0.0), max_h)
        ref_x2 = torch.maximum(ref_x2, ref_x1 + self.min_box_size)
        ref_y2 = torch.maximum(ref_y2, ref_y1 + self.min_box_size)
        return torch.stack([ref_x1, ref_y1, ref_x2, ref_y2], dim=-1)

    def apply_context_groups(
        self,
        refined_boxes: torch.Tensor,
        context_factors: torch.Tensor,
        batch_inds: torch.Tensor,
        batch_data_samples: List[InstanceData],
    ) -> torch.Tensor:
        if refined_boxes.numel() == 0:
            return refined_boxes.new_zeros(context_factors.shape + (4,))
        ratio = 1.0 + context_factors.clamp(min=0.0)
        x1 = refined_boxes[..., 0]
        y1 = refined_boxes[..., 1]
        x2 = refined_boxes[..., 2]
        y2 = refined_boxes[..., 3]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = (x2 - x1).clamp(min=self.min_box_size) * ratio
        bh = (y2 - y1).clamp(min=self.min_box_size) * ratio
        roi_x1 = cx - 0.5 * bw
        roi_y1 = cy - 0.5 * bh
        roi_x2 = cx + 0.5 * bw
        roi_y2 = cy + 0.5 * bh
        if batch_inds.numel() > 0:
            img_heights = roi_x1.new_tensor([float(d.img_shape[0]) for d in batch_data_samples])
            img_widths = roi_x1.new_tensor([float(d.img_shape[1]) for d in batch_data_samples])
            max_h = img_heights.index_select(0, batch_inds).unsqueeze(1)
            max_w = img_widths.index_select(0, batch_inds).unsqueeze(1)
            roi_x1 = torch.minimum(roi_x1.clamp(min=0.0), max_w)
            roi_y1 = torch.minimum(roi_y1.clamp(min=0.0), max_h)
            roi_x2 = torch.minimum(roi_x2.clamp(min=0.0), max_w)
            roi_y2 = torch.minimum(roi_y2.clamp(min=0.0), max_h)
        roi_x2 = torch.maximum(roi_x2, roi_x1 + self.min_box_size)
        roi_y2 = torch.maximum(roi_y2, roi_y1 + self.min_box_size)
        return torch.stack([roi_x1, roi_y1, roi_x2, roi_y2], dim=-1)

    def enumerate_center_groups(
        self,
        feats: Tuple[torch.Tensor, ...],
        proposal_instances: List[InstanceData],
        batch_data_samples: List[InstanceData],
    ) -> Dict[str, torch.Tensor]:
        enc = self.encode(feats, proposal_instances, batch_data_samples)
        if enc['boxes'].numel() == 0:
            zero_boxes = enc['boxes'].new_zeros((0, self.num_shift_groups, 4))
            enc.update(
                center_logits=enc['boxes'].new_zeros((0, self.num_shift_groups)),
                center_probs=enc['boxes'].new_zeros((0, self.num_shift_groups)),
                all_center_offsets=enc['boxes'].new_zeros((0, self.num_shift_groups, 2)),
                all_center_boxes=zero_boxes,
                expected_center_boxes=enc['boxes'].new_zeros((0, 4)),
                best_center_inds=enc['batch_inds'].new_zeros((0,)),
                best_center_boxes=enc['boxes'].new_zeros((0, 4)),
            )
            return enc
        full_logits = self.policy_head(enc['embed'])
        center_logits = self._aggregate_logits_by_group(full_logits, self.shift_group_ids, self.num_shift_groups)
        center_probs = F.softmax(center_logits, dim=-1)
        center_offsets = self.shift_actions.to(device=enc['boxes'].device, dtype=enc['boxes'].dtype).unsqueeze(0).expand(enc['boxes'].shape[0], -1, -1)
        all_center_boxes = self.apply_shift_groups(enc['boxes'], center_offsets, enc['batch_inds'], batch_data_samples)
        expected_center_boxes = torch.sum(all_center_boxes * center_probs.unsqueeze(-1), dim=1)
        best_center_inds = torch.argmax(center_logits, dim=-1)
        gather_index = torch.arange(enc['boxes'].shape[0], device=best_center_inds.device)
        best_center_boxes = all_center_boxes[gather_index, best_center_inds]
        enc.update(
            center_logits=center_logits,
            center_probs=center_probs,
            all_center_offsets=center_offsets,
            all_center_boxes=all_center_boxes,
            expected_center_boxes=expected_center_boxes,
            best_center_inds=best_center_inds,
            best_center_boxes=best_center_boxes,
        )
        return enc

    def enumerate_scale_context_groups(
        self,
        feats: Tuple[torch.Tensor, ...],
        proposal_instances,
        batch_data_samples,
    ) -> Dict[str, torch.Tensor]:
        enc = self.encode(feats, proposal_instances, batch_data_samples)
        if enc['boxes'].numel() == 0:
            zero_boxes = enc['boxes'].new_zeros((0, self.num_scale_context_groups, 4))
            zero_actions = enc['boxes'].new_zeros((0, self.num_scale_context_groups, 7))
            enc.update(
                scale_context_logits=enc['boxes'].new_zeros((0, self.num_scale_context_groups)),
                scale_context_probs=enc['boxes'].new_zeros((0, self.num_scale_context_groups)),
                all_scale_context_action_raw=zero_actions,
                all_scale_context_offsets=zero_actions[..., :2],
                all_scale_context_margins=zero_actions[..., 2:6],
                all_scale_context_factors=enc['boxes'].new_zeros((0, self.num_scale_context_groups)),
                all_scale_context_shifted_boxes=zero_boxes,
                all_scale_context_refined_boxes=zero_boxes,
                all_scale_context_roi_boxes=zero_boxes,
                expected_scale_context_boxes=enc['boxes'].new_zeros((0, 4)),
                expected_scale_context_roi_boxes=enc['boxes'].new_zeros((0, 4)),
                best_scale_context_inds=enc['batch_inds'].new_zeros((0,)),
                best_scale_context_boxes=enc['boxes'].new_zeros((0, 4)),
                best_scale_context_roi_boxes=enc['boxes'].new_zeros((0, 4)),
            )
            return enc
        full_logits = self.policy_head(enc['embed'])
        scale_context_logits = self._aggregate_logits_by_group(
            full_logits, self.scale_context_group_ids, self.num_scale_context_groups
        )
        scale_context_probs = F.softmax(scale_context_logits, dim=-1)
        num_props = enc['boxes'].shape[0]
        action_bank = self.scale_context_action_raw_bank.to(device=enc['boxes'].device, dtype=enc['boxes'].dtype)
        all_scale_context_action_raw = action_bank.unsqueeze(0).expand(num_props, -1, -1)
        all_scale_context_offsets = all_scale_context_action_raw[..., :2]
        all_scale_context_margins = all_scale_context_action_raw[..., 2:6]
        all_scale_context_factors = all_scale_context_action_raw[..., 6]
        all_scale_context_shifted_boxes = self._expand_boxes_for_groups(enc['boxes'], self.num_scale_context_groups)
        all_scale_context_refined_boxes = self.apply_margin_groups(
            all_scale_context_shifted_boxes, all_scale_context_margins, enc['batch_inds'], batch_data_samples
        )
        all_scale_context_roi_boxes = self.apply_context_groups(
            all_scale_context_refined_boxes, all_scale_context_factors, enc['batch_inds'], batch_data_samples
        )
        expected_scale_context_boxes = torch.sum(all_scale_context_refined_boxes * scale_context_probs.unsqueeze(-1), dim=1)
        expected_scale_context_roi_boxes = torch.sum(all_scale_context_roi_boxes * scale_context_probs.unsqueeze(-1), dim=1)
        best_scale_context_inds = torch.argmax(scale_context_logits, dim=-1)
        gather_index = torch.arange(num_props, device=best_scale_context_inds.device)
        best_scale_context_boxes = all_scale_context_refined_boxes[gather_index, best_scale_context_inds]
        best_scale_context_roi_boxes = all_scale_context_roi_boxes[gather_index, best_scale_context_inds]
        enc.update(
            scale_context_logits=scale_context_logits,
            scale_context_probs=scale_context_probs,
            all_scale_context_action_raw=all_scale_context_action_raw,
            all_scale_context_offsets=all_scale_context_offsets,
            all_scale_context_margins=all_scale_context_margins,
            all_scale_context_factors=all_scale_context_factors,
            all_scale_context_shifted_boxes=all_scale_context_shifted_boxes,
            all_scale_context_refined_boxes=all_scale_context_refined_boxes,
            all_scale_context_roi_boxes=all_scale_context_roi_boxes,
            expected_scale_context_boxes=expected_scale_context_boxes,
            expected_scale_context_roi_boxes=expected_scale_context_roi_boxes,
            best_scale_context_inds=best_scale_context_inds,
            best_scale_context_boxes=best_scale_context_boxes,
            best_scale_context_roi_boxes=best_scale_context_roi_boxes,
        )
        return enc

    def sample_scale_context_groups(
        self,
        feats: Tuple[torch.Tensor, ...],
        proposal_instances,
        batch_data_samples,
        group_size: int,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        enc = self.enumerate_scale_context_groups(feats, proposal_instances, batch_data_samples)
        group_size = int(group_size)
        if group_size <= 0:
            raise ValueError('group_size must be positive.')
        if enc['boxes'].numel() == 0:
            zero_boxes = enc['boxes'].new_zeros((0, group_size, 4))
            zero_scores = enc['boxes'].new_zeros((0, group_size))
            zero_actions = enc['boxes'].new_zeros((0, group_size, 7))
            enc.update(
                action_indices=enc['batch_inds'].new_zeros((0, group_size)),
                action_raw=zero_actions,
                offsets=zero_actions[..., :2],
                margins=zero_actions[..., 2:6],
                context_factors=zero_scores,
                shifted_boxes=zero_boxes,
                log_prob=zero_scores,
                entropy=zero_scores,
                refined_boxes=zero_boxes,
                roi_boxes=zero_boxes,
            )
            return enc
        log_probs_all = F.log_softmax(enc['scale_context_logits'], dim=-1)
        entropy = -(enc['scale_context_probs'] * log_probs_all).sum(dim=-1, keepdim=True)
        if deterministic:
            action_indices = enc['best_scale_context_inds'][:, None].expand(-1, group_size)
        else:
            dist = torch.distributions.Categorical(logits=enc['scale_context_logits'])
            action_indices = dist.sample((group_size,)).permute(1, 0)
        flat_indices = action_indices.reshape(-1)
        action_bank = self.scale_context_action_raw_bank.to(device=enc['boxes'].device, dtype=enc['boxes'].dtype)
        action_raw = action_bank.index_select(0, flat_indices).view(enc['boxes'].shape[0], group_size, -1)
        margins = action_raw[..., 2:6]
        context_factors = action_raw[..., 6]
        shifted_boxes = self._expand_boxes_for_groups(enc['boxes'], group_size)
        refined_boxes = self.apply_margin_groups(shifted_boxes, margins, enc['batch_inds'], batch_data_samples)
        roi_boxes = self.apply_context_groups(refined_boxes, context_factors, enc['batch_inds'], batch_data_samples)
        enc.update(
            action_indices=action_indices,
            action_raw=action_raw,
            offsets=action_raw[..., :2],
            margins=margins,
            context_factors=context_factors,
            shifted_boxes=shifted_boxes,
            log_prob=log_probs_all.gather(1, action_indices),
            entropy=entropy.expand_as(action_indices),
            refined_boxes=refined_boxes,
            roi_boxes=roi_boxes,
        )
        return enc

    def evaluate_scale_context_groups(
        self,
        feats: Tuple[torch.Tensor, ...],
        proposal_instances,
        batch_data_samples,
        action_raw: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        enc = self.enumerate_scale_context_groups(feats, proposal_instances, batch_data_samples)
        if enc['boxes'].numel() == 0:
            zero_scores = action_raw.new_zeros(action_raw.shape[:2])
            zero_boxes = action_raw.new_zeros(action_raw.shape[:2] + (4,))
            enc.update(
                action_indices=action_raw.new_zeros(action_raw.shape[:2], dtype=torch.long),
                action_raw=action_raw,
                offsets=action_raw[..., :2],
                margins=action_raw[..., 2:6],
                context_factors=zero_scores,
                shifted_boxes=zero_boxes,
                log_prob=zero_scores,
                entropy=zero_scores,
                refined_boxes=zero_boxes,
                roi_boxes=zero_boxes,
            )
            return enc
        bank = self.scale_context_action_raw_bank.to(device=action_raw.device, dtype=action_raw.dtype)
        flat_actions = action_raw.reshape(-1, action_raw.shape[-1])
        distances = (flat_actions[:, None, 2:7] - bank[None, :, 2:7]).abs().sum(dim=-1)
        action_indices = torch.argmin(distances, dim=1).view(*action_raw.shape[:2])
        log_probs_all = F.log_softmax(enc['scale_context_logits'], dim=-1)
        entropy = -(enc['scale_context_probs'] * log_probs_all).sum(dim=-1, keepdim=True)
        margins = action_raw[..., 2:6]
        context_factors = action_raw[..., 6]
        shifted_boxes = self._expand_boxes_for_groups(enc['boxes'], action_raw.shape[1])
        refined_boxes = self.apply_margin_groups(shifted_boxes, margins, enc['batch_inds'], batch_data_samples)
        roi_boxes = self.apply_context_groups(refined_boxes, context_factors, enc['batch_inds'], batch_data_samples)
        enc.update(
            action_indices=action_indices,
            action_raw=action_raw,
            offsets=action_raw[..., :2],
            margins=margins,
            context_factors=context_factors,
            shifted_boxes=shifted_boxes,
            log_prob=log_probs_all.gather(1, action_indices),
            entropy=entropy.expand_as(action_indices),
            refined_boxes=refined_boxes,
            roi_boxes=roi_boxes,
        )
        return enc

    def boxes_to_instance_list(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        batch_inds: torch.Tensor,
        batch_data_samples: List[InstanceData],
        roi_boxes: torch.Tensor = None,
    ) -> List[InstanceData]:
        outputs: List[InstanceData] = []
        device = boxes.device if boxes.numel() > 0 else scores.device
        for img_idx in range(len(batch_data_samples)):
            mask = batch_inds == img_idx
            if not bool(mask.any()):
                outputs.append(self._empty_instance(device))
                continue
            inst = InstanceData()
            inst.bboxes = boxes[mask]
            inst.scores = scores[mask]
            inst.labels = labels[mask]
            if roi_boxes is not None and roi_boxes.numel() > 0:
                inst.roi_bboxes = roi_boxes[mask]
            outputs.append(inst)
        return outputs

    def predict_center_instances(
        self,
        feats: Tuple[torch.Tensor, ...],
        proposal_instances: List[InstanceData],
        batch_data_samples: List[InstanceData],
        use_expected: bool = False,
    ) -> List[InstanceData]:
        center_outputs = self.enumerate_center_groups(feats, proposal_instances, batch_data_samples)
        boxes = (
            center_outputs['expected_center_boxes']
            if use_expected else center_outputs['best_center_boxes']
        )
        return self.boxes_to_instance_list(
            boxes=boxes,
            scores=center_outputs['scores'],
            labels=center_outputs['labels'],
            batch_inds=center_outputs['batch_inds'],
            batch_data_samples=batch_data_samples,
            roi_boxes=boxes,
        )

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
        action_outputs = self.sample_scale_context_groups(
            feats=feats,
            proposal_instances=proposal_instances,
            batch_data_samples=batch_data_samples,
            group_size=1,
            deterministic=True,
        )
        return self.boxes_to_instance_list(
            boxes=action_outputs['refined_boxes'].squeeze(1),
            scores=action_outputs['scores'],
            labels=action_outputs['labels'],
            batch_inds=action_outputs['batch_inds'],
            batch_data_samples=batch_data_samples,
            roi_boxes=action_outputs['roi_boxes'].squeeze(1),
        )
