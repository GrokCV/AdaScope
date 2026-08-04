import logging
import math
from collections import OrderedDict, deque
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.logging import print_log
from mmengine.optim import OptimWrapper
from mmengine.structures import InstanceData
from mmengine.utils import import_modules_from_strings
from mmdet.apis import init_detector
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox_overlaps
from mmdet.utils import ConfigType, OptConfigType
from torch import Tensor
from torchvision.ops import nms as torchvision_nms


@MODELS.register_module()
class AdaScopeDetector(SingleStageDetector):
    """Flattened SyncCleanGRPODetector for the current external-local path."""

    def __init__(
        self,
        backbone: ConfigType,
        cluster_head: OptConfigType,
        global_bbox_head: OptConfigType = None,
        bbox_head: OptConfigType = None,
        neck: OptConfigType = None,
        refiner: OptConfigType = None,
        use_local_head: bool = False,
        train_local_head: bool = False,
        share_local_with_global: bool = True,
        local_loss_weight: float = 1.0,
        local_train_use_gt_rois: bool = True,
        refiner_handoff_alpha: float = 1.0,
        cluster_feature_gate_strength: float = 0.0,
        cluster_feature_gate_detach: bool = True,
        cluster_score_fusion_weight: float = 0.0,
        local_roi_source: str = 'blend',
        global_local_suppress_iou_thr: float = 0.0,
        use_refiner: bool = True,
        train_global_head: bool = True,
        roi_topk: int = 20,
        roi_score_thr: float = 0.05,
        merge_nms_iou_thr: float = 0.5,
        proposal_score_reduction: str = 'max',
        cluster_connectivity=4,
        proposal_train_topk: int = 128,
        proposal_train_score_thr: float = 0.05,
        proposal_match_iou_thr: float = 0.1,
        proposal_bbox_loss_weight: float = 1.0,
        proposal_iou_loss_weight: float = 1.0,
        proposal_smooth_l1_beta: float = 1.0,
        proposal_min_box_size: float = 2.0,
        proposal_use_detached_logits: bool = True,
        proposal_offset_clamp: float = 2.0,
        proposal_scale_log_clamp: float = 1.5,
        enable_external_local: bool = True,
        external_local_frozen: bool = True,
        external_local_device: str = 'cuda:0',
        external_local_cfg: str = '',
        external_local_ckpt: str = '',
        auto_sync_external_from_global: bool = True,
        roi_source: str = 'refined',
        raw_roi_expand_ratio: float = 1.0,
        ext_local_score_thr: float = 0.05,
        ext_local_score_scale: float = 1.0,
        ext_local_batch_size: int = 2,
        ext_local_max_per_patch: int = 150,
        stitched_cell_size: Tuple[int, int] = (384, 384),
        stitched_cell_gap: int = 8,
        stitched_max_cols: int = 4,
        policy_reward_det_weight: float = 1.0,
        policy_reward_geo_weight: float = 0.2,
        policy_reward_area_weight: float = 0.1,
        policy_reward_fp_weight: float = 0.15,
        policy_reward_peak_weight: float = 1.0,
        policy_reward_mass_weight: float = 0.25,
        policy_reward_count_weight: float = 0.15,
        policy_reward_purity_weight: float = 0.1,
        policy_reward_outside_mass_weight: float = 0.05,
        policy_cover_penalty_weight: float = 0.05,
        policy_use_gtcluster_det_baseline: bool = False,
        policy_fp_iou_thr: float = 0.1,
        policy_margin_reg_weight: float = 0.01,
        policy_reward_clip: float = 2.0,
        policy_train_topk: int = 8,
        grpo_group_size: int = 4,
        grpo_update_steps: int = 2,
        grpo_clip_eps: float = 0.2,
        grpo_policy_clip_enabled: bool = False,
        grpo_entropy_weight: float = 0.001,
        grpo_center_reward_weight: float = 0.1,
        grpo_shift_reg_weight: Optional[float] = None,
        grpo_margin_reg_weight: float = 0.25,
        grpo_cover_reg_weight: float = 0.5,
        grpo_area_reg_weight: float = 0.1,
        grpo_shift_mag_reg_weight: float = 0.02,
        grpo_area_budget: float = 2.0,
        grpo_cover_keep_ratio: float = 0.9,
        grpo_dense_min_gt: int = 2,
        grpo_advantage_eps: float = 1e-6,
        grpo_ref_kl_weight: float = 0.0,
        grpo_ref_kl_loss_scale: float = 20.0,
        grpo_ref_kl_safe_max: float = 200.0,
        grpo_use_reference_policy: bool = False,
        grpo_reference_metric_key: str = 'merged_voc/mAP',
        grpo_seed_reference_on_stage_enable: bool = False,
        grpo_seed_reference_score: float = float('-inf'),
        grpo_seed_reference_epoch: int = -1,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptConfigType = None,
        **kwargs,
    ) -> None:
        if global_bbox_head is None:
            global_bbox_head = bbox_head
        if global_bbox_head is None:
            raise ValueError('global_bbox_head (or bbox_head) must be provided.')

        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=global_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )

        self.cluster_head = MODELS.build(cluster_head) if cluster_head is not None else None
        self.use_cluster_head = self.cluster_head is not None
        if use_refiner and not self.use_cluster_head:
            raise ValueError('use_refiner=True requires cluster_head.')

        self.refiner = MODELS.build(refiner) if refiner is not None else None
        self.use_refiner = bool(use_refiner and self.refiner is not None and self.use_cluster_head)
        self.global_bbox_head = self.bbox_head

        self.use_local_head = bool(use_local_head)
        self.train_local_head = bool(train_local_head)
        self.share_local_with_global = bool(share_local_with_global)
        self.local_loss_weight = float(local_loss_weight)
        self.local_train_use_gt_rois = bool(local_train_use_gt_rois)
        self.refiner_handoff_alpha = float(refiner_handoff_alpha)
        self.cluster_feature_gate_strength = float(cluster_feature_gate_strength)
        self.cluster_feature_gate_detach = bool(cluster_feature_gate_detach)
        self.cluster_score_fusion_weight = float(cluster_score_fusion_weight)
        self.local_roi_source = str(local_roi_source).lower()
        self.global_local_suppress_iou_thr = float(global_local_suppress_iou_thr)
        self.train_global_head = bool(train_global_head)

        if self.cluster_feature_gate_strength != 0.0 or self.cluster_score_fusion_weight != 0.0:
            raise ValueError('Cluster feature/score gating must stay disabled.')
        if self.global_local_suppress_iou_thr != 0.0:
            raise ValueError('Global/local suppression must stay disabled.')

        self.roi_topk = int(roi_topk)
        self.roi_score_thr = float(roi_score_thr)
        self.merge_nms_iou_thr = float(merge_nms_iou_thr)

        self.proposal_score_reduction = str(proposal_score_reduction).lower()
        self.cluster_connectivity = self._normalize_cluster_connectivity(
            cluster_connectivity)
        self.proposal_train_topk = int(proposal_train_topk)
        self.proposal_train_score_thr = float(proposal_train_score_thr)
        self.proposal_match_iou_thr = float(proposal_match_iou_thr)
        self.proposal_bbox_loss_weight = float(proposal_bbox_loss_weight)
        self.proposal_iou_loss_weight = float(proposal_iou_loss_weight)
        self.proposal_smooth_l1_beta = float(proposal_smooth_l1_beta)
        self.proposal_min_box_size = float(proposal_min_box_size)
        self.proposal_use_detached_logits = bool(proposal_use_detached_logits)
        self.proposal_offset_clamp = float(proposal_offset_clamp)
        self.proposal_scale_log_clamp = float(proposal_scale_log_clamp)

        self.enable_external_local = bool(enable_external_local)
        self.external_local_frozen = bool(external_local_frozen)
        self.external_local_device = str(external_local_device)
        self.external_local_cfg = str(external_local_cfg)
        self.external_local_ckpt = str(external_local_ckpt)
        self.auto_sync_external_from_global = bool(auto_sync_external_from_global)
        self.roi_source = str(roi_source).lower()
        self.raw_roi_expand_ratio = float(raw_roi_expand_ratio)
        self.ext_local_score_thr = float(ext_local_score_thr)
        self.ext_local_score_scale = float(ext_local_score_scale)
        self.ext_local_batch_size = max(int(ext_local_batch_size), 1)
        self.ext_local_max_per_patch = int(ext_local_max_per_patch)
        self.stitched_cell_size = (
            max(int(stitched_cell_size[0]), 8),
            max(int(stitched_cell_size[1]), 8),
        )
        self.stitched_cell_gap = max(int(stitched_cell_gap), 0)
        self.stitched_max_cols = max(int(stitched_max_cols), 1)
        self._external_local_holder = dict(
            model=None,
            needs_sync=True,
            last_sync_summary=None,
            sync_log_printed=False,
        )

        self.policy_reward_det_weight = float(policy_reward_det_weight)
        self.policy_reward_geo_weight = float(policy_reward_geo_weight)
        self.policy_reward_area_weight = float(policy_reward_area_weight)
        self.policy_reward_fp_weight = float(policy_reward_fp_weight)
        self.policy_reward_peak_weight = float(policy_reward_peak_weight)
        self.policy_reward_mass_weight = float(policy_reward_mass_weight)
        self.policy_reward_count_weight = float(policy_reward_count_weight)
        self.policy_reward_purity_weight = float(policy_reward_purity_weight)
        self.policy_reward_outside_mass_weight = float(policy_reward_outside_mass_weight)
        self.policy_cover_penalty_weight = float(policy_cover_penalty_weight)
        self.policy_use_gtcluster_det_baseline = bool(policy_use_gtcluster_det_baseline)
        self.policy_fp_iou_thr = float(policy_fp_iou_thr)
        self.policy_margin_reg_weight = float(policy_margin_reg_weight)
        self.policy_reward_clip = float(policy_reward_clip)
        self.policy_train_topk = int(policy_train_topk)
        self.policy_stage_enabled = False
        self._policy_reward_baseline_state = None
        self._policy_reward_baseline_score = float('-inf')
        self._policy_reward_baseline_epoch = -1
        self._policy_reward_baseline_restored = False
        self.training_stage = str(kwargs.pop('training_stage', 'warmup')).lower()
        self.refiner_supervised_stage_enabled = False
        self.refiner_sup_center_weight = float(kwargs.pop('refiner_sup_center_weight', 1.0))
        self.refiner_sup_use_policy_topk = bool(kwargs.pop('refiner_sup_use_policy_topk', False))
        self._grpo_center_anchor_refiner = None
        self._refiner_sup_baseline_state = None
        self._refiner_sup_baseline_score = float('-inf')
        self._refiner_sup_baseline_epoch = -1
        self._refiner_sup_baseline_restored = False

        self.grpo_group_size = int(grpo_group_size)
        self.grpo_update_steps = int(grpo_update_steps)
        self.grpo_clip_eps = float(grpo_clip_eps)
        self.grpo_policy_clip_enabled = bool(grpo_policy_clip_enabled)
        self.grpo_entropy_weight = float(grpo_entropy_weight)
        self.grpo_shift_reg_weight = float(
            grpo_center_reward_weight if grpo_shift_reg_weight is None else grpo_shift_reg_weight)
        self.grpo_margin_reg_weight = float(grpo_margin_reg_weight)
        self.grpo_cover_reg_weight = float(grpo_cover_reg_weight)
        self.grpo_area_reg_weight = float(grpo_area_reg_weight)
        self.grpo_shift_mag_reg_weight = float(grpo_shift_mag_reg_weight)
        self.grpo_area_budget = float(grpo_area_budget)
        self.grpo_cover_keep_ratio = float(grpo_cover_keep_ratio)
        self.grpo_dense_min_gt = int(grpo_dense_min_gt)
        self.grpo_advantage_eps = float(grpo_advantage_eps)
        self.grpo_ref_kl_weight = float(grpo_ref_kl_weight)
        self.grpo_ref_kl_loss_scale = float(grpo_ref_kl_loss_scale)
        self.grpo_ref_kl_safe_max = float(grpo_ref_kl_safe_max)
        self.grpo_use_reference_policy = bool(grpo_use_reference_policy)
        self.grpo_reference_metric_key = str(grpo_reference_metric_key)
        self.grpo_seed_reference_on_stage_enable = bool(grpo_seed_reference_on_stage_enable)
        self.grpo_seed_reference_score = float(grpo_seed_reference_score)
        self.grpo_seed_reference_epoch = int(grpo_seed_reference_epoch)
        self._grpo_reference_refiner = None
        self._grpo_reference_score = float('-inf')
        self._grpo_reference_epoch = -1
        self.extra_init_kwargs = dict(kwargs)

    def train(self, mode: bool = True):
        out = super().train(mode)
        if mode:
            self._external_local_holder['needs_sync'] = True
        return out

    @staticmethod
    def _add_prefix(losses: Dict[str, Tensor], prefix: str) -> Dict[str, Tensor]:
        return {f'{prefix}_{k}': v for k, v in losses.items()}

    def _extract_backbone_and_det_feats(self, batch_inputs: Tensor):
        backbone_feats = self.backbone(batch_inputs)
        if isinstance(backbone_feats, Tensor):
            backbone_feats = (backbone_feats,)
        det_feats = self.neck(backbone_feats) if self.with_neck else backbone_feats
        if isinstance(det_feats, Tensor):
            det_feats = (det_feats,)
        return tuple(backbone_feats), tuple(det_feats)

    @staticmethod
    def _get_gt_cluster_boxes(data_sample, device: torch.device) -> Tensor:
        cluster_boxes = getattr(data_sample, 'gt_cluster_bboxes', None)
        if cluster_boxes is None and hasattr(data_sample, 'metainfo'):
            cluster_boxes = data_sample.metainfo.get('gt_cluster_bboxes', None)
        if cluster_boxes is None:
            return torch.zeros((0, 4), device=device, dtype=torch.float32)
        boxes = torch.as_tensor(cluster_boxes, device=device, dtype=torch.float32)
        if boxes.numel() == 0:
            return torch.zeros((0, 4), device=device, dtype=torch.float32)
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)
        return boxes[:, :4]

    @staticmethod
    def _empty_instance_list(batch_size: int, device: torch.device) -> List[InstanceData]:
        outputs = []
        for _ in range(batch_size):
            inst = InstanceData()
            inst.bboxes = torch.zeros((0, 4), device=device)
            inst.scores = torch.zeros((0,), device=device)
            inst.labels = torch.zeros((0,), dtype=torch.long, device=device)
            outputs.append(inst)
        return outputs

    @staticmethod
    def _normalize_cluster_connectivity(connectivity):
        if isinstance(connectivity, str):
            value = connectivity.strip().lower()
            mapping = {
                '4': '4',
                'four': '4',
                'cc4': '4',
                '4-connected': '4',
                '8': '8',
                'eight': '8',
                'cc8': '8',
                '8-connected': '8',
                '0': 'disconnected',
                'none': 'disconnected',
                'single': 'disconnected',
                'isolated': 'disconnected',
                'disconnected': 'disconnected',
            }
            if value in mapping:
                return mapping[value]
        elif connectivity in (0, 4, 8):
            return 'disconnected' if int(connectivity) == 0 else str(int(connectivity))

        raise ValueError(
            'cluster_connectivity must be one of 4, 8, or disconnected.')

    def _component_neighbors(self, y: int, x: int):
        if self.cluster_connectivity == '4':
            return ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
        if self.cluster_connectivity == '8':
            return (
                (y - 1, x - 1), (y - 1, x), (y - 1, x + 1),
                (y, x - 1), (y, x + 1),
                (y + 1, x - 1), (y + 1, x), (y + 1, x + 1),
            )
        return ()

    def _merge_mask_to_boxes(self, prob_map: Tensor, mask: Tensor, img_shape):
        img_h, img_w = img_shape[:2]
        feat_h, feat_w = prob_map.shape
        stride_h = float(img_h) / max(float(feat_h), 1.0)
        stride_w = float(img_w) / max(float(feat_w), 1.0)
        active = (mask > 0).detach().cpu()
        if not bool(active.any()):
            return prob_map.new_zeros((0, 4)), prob_map.new_zeros((0,))

        prob_cpu = prob_map.detach().cpu()
        visited = torch.zeros_like(active, dtype=torch.bool)
        boxes = []
        scores = []
        for y in range(feat_h):
            for x in range(feat_w):
                if not bool(active[y, x]) or bool(visited[y, x]):
                    continue
                queue = deque([(y, x)])
                visited[y, x] = True
                ys = []
                xs = []
                comp_scores = []
                while queue:
                    cy, cx = queue.popleft()
                    ys.append(cy)
                    xs.append(cx)
                    comp_scores.append(float(prob_cpu[cy, cx].item()))
                    for ny, nx in self._component_neighbors(cy, cx):
                        if ny < 0 or nx < 0 or ny >= feat_h or nx >= feat_w:
                            continue
                        if not bool(active[ny, nx]) or bool(visited[ny, nx]):
                            continue
                        visited[ny, nx] = True
                        queue.append((ny, nx))

                boxes.append([
                    min(xs) * stride_w,
                    min(ys) * stride_h,
                    (max(xs) + 1) * stride_w,
                    (max(ys) + 1) * stride_h,
                ])
                scores.append(
                    sum(comp_scores) / max(len(comp_scores), 1)
                    if self.proposal_score_reduction == 'mean' else max(comp_scores))
        return (
            prob_map.new_tensor(boxes, dtype=torch.float32),
            prob_map.new_tensor(scores, dtype=torch.float32),
        )

    def _get_merged_cluster_instances_by_feat(self, cls_score: Tensor, batch_data_samples: SampleList):
        cls_prob = cls_score.sigmoid()
        proposal_instances = []
        for i, data_sample in enumerate(batch_data_samples):
            prob_map = cls_prob[i, 0]
            mask = self.cluster_head._build_mask(prob_map, data_sample, use_gt_mask=False)
            boxes, scores = self._merge_mask_to_boxes(prob_map, mask, data_sample.img_shape)
            data_sample.pred_cls_heatmap = cls_prob[i]
            inst = InstanceData()
            inst.bboxes = boxes
            inst.scores = scores
            inst.labels = torch.zeros((boxes.shape[0],), device=boxes.device, dtype=torch.long)
            proposal_instances.append(inst)
        return proposal_instances

    def _filter_train_proposals(self, proposal_instances, batch_data_samples):
        filtered = []
        for proposal, data_sample in zip(proposal_instances, batch_data_samples):
            boxes = proposal.bboxes
            scores = proposal.scores
            device = boxes.device
            if boxes.numel() == 0:
                filtered.append(self._empty_instance_list(1, device)[0])
                continue

            img_h, img_w = data_sample.img_shape[:2]
            boxes = boxes.clone()
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, float(img_w))
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, float(img_h))
            keep = (
                (boxes[:, 2] - boxes[:, 0] >= self.proposal_min_box_size) &
                (boxes[:, 3] - boxes[:, 1] >= self.proposal_min_box_size) &
                (scores >= self.proposal_train_score_thr)
            )
            boxes = boxes[keep]
            scores = scores[keep]
            if boxes.numel() == 0:
                filtered.append(self._empty_instance_list(1, device)[0])
                continue
            if self.proposal_train_topk > 0 and boxes.shape[0] > self.proposal_train_topk:
                order = torch.argsort(scores, descending=True)[:self.proposal_train_topk]
                boxes = boxes[order]
                scores = scores[order]
            inst = InstanceData()
            inst.bboxes = boxes
            inst.scores = scores
            inst.labels = torch.zeros((boxes.shape[0],), device=device, dtype=torch.long)
            filtered.append(inst)
        return filtered

    def _match_one_to_one(self, proposals: Tensor, gt_boxes: Tensor) -> Tensor:
        assigned_gt = proposals.new_full((proposals.shape[0],), -1, dtype=torch.long)
        if proposals.shape[0] == 0 or gt_boxes.numel() == 0:
            return assigned_gt
        ious = bbox_overlaps(proposals, gt_boxes)
        work = ious.clone()
        num_gts = gt_boxes.shape[0]
        while True:
            max_val, flat_idx = work.view(-1).max(dim=0)
            if float(max_val.item()) < self.proposal_match_iou_thr:
                break
            prop_idx = int(flat_idx.item() // num_gts)
            gt_idx = int(flat_idx.item() % num_gts)
            assigned_gt[prop_idx] = gt_idx
            work[prop_idx, :] = -1
            work[:, gt_idx] = -1
        return assigned_gt

    def _proposal_refiner_loss(self, proposal_instances, refined_instances, batch_data_samples, device):
        total_bbox_loss = torch.zeros((), device=device)
        total_iou_loss = torch.zeros((), device=device)
        total_pos = torch.zeros((), device=device)
        for proposal, refined, data_sample in zip(proposal_instances, refined_instances, batch_data_samples):
            raw_boxes = proposal.bboxes
            refined_boxes = refined.bboxes
            if raw_boxes.numel() == 0 or refined_boxes.numel() == 0:
                continue
            gt_boxes = self._get_gt_cluster_boxes(data_sample, device=device)
            if gt_boxes.numel() == 0:
                continue
            assigned_gt = self._match_one_to_one(raw_boxes, gt_boxes)
            pos_mask = assigned_gt >= 0
            if not bool(pos_mask.any()):
                continue
            matched_gt = gt_boxes[assigned_gt[pos_mask]]
            matched_refined = refined_boxes[pos_mask]
            img_h, img_w = data_sample.img_shape[:2]
            norm = matched_refined.new_tensor([float(img_w), float(img_h), float(img_w), float(img_h)]).clamp(min=1.0)
            diff = (matched_refined / norm - matched_gt / norm).abs()
            beta = self.proposal_smooth_l1_beta
            bbox_loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta).mean(dim=1)
            aligned_iou = bbox_overlaps(matched_refined, matched_gt, is_aligned=True).clamp(min=0.0, max=1.0)
            total_bbox_loss = total_bbox_loss + bbox_loss.sum()
            total_iou_loss = total_iou_loss + (1.0 - aligned_iou).sum()
            total_pos = total_pos + float(matched_refined.shape[0])
        normalizer = total_pos.clamp(min=1.0)
        return dict(
            loss_bbox=self.proposal_bbox_loss_weight * total_bbox_loss / normalizer,
            loss_iou=self.proposal_iou_loss_weight * total_iou_loss / normalizer,
        )

    def _base_detector_loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        backbone_feats, det_feats = self._extract_backbone_and_det_feats(batch_inputs)
        c5_feats = (backbone_feats[-1],)
        losses = {}
        if self.use_cluster_head:
            cluster_logits = self.cluster_head(c5_feats)
            losses.update(self._add_prefix(
                self.cluster_head.loss_by_feat(cluster_logits, batch_data_samples), 'cluster'))
        if self.use_refiner:
            proposal_logits = cluster_logits.detach() if self.proposal_use_detached_logits else cluster_logits
            proposal_instances = self._filter_train_proposals(
                self._get_merged_cluster_instances_by_feat(proposal_logits, batch_data_samples),
                batch_data_samples)
            losses.update(self._add_prefix(self.refiner.loss(det_feats, batch_data_samples), 'refiner'))
            pred_maps = self._stabilize_pred_maps(self.refiner.predict_maps(det_feats))
            refined_instances = self.refiner.refine_instances(
                proposal_instances, pred_maps, batch_data_samples)
            losses.update(self._add_prefix(
                self._proposal_refiner_loss(
                    proposal_instances, refined_instances, batch_data_samples, device=det_feats[0].device),
                'refiner'))
        if self.train_global_head:
            losses.update(self._add_prefix(
                self.global_bbox_head.loss(det_feats, batch_data_samples), 'global'))
        return losses

    def _stabilize_pred_maps(self, pred_maps: Dict[str, Tensor]) -> Dict[str, Tensor]:
        stable_maps = {}
        for key, value in pred_maps.items():
            if key == 'offset':
                stable_maps[key] = value.clamp(
                    min=-self.proposal_offset_clamp, max=self.proposal_offset_clamp)
            elif key == 'scale':
                stable_maps[key] = value.clamp(
                    min=-self.proposal_scale_log_clamp, max=self.proposal_scale_log_clamp)
            else:
                stable_maps[key] = value
        return stable_maps

    def _resolve_roi_source(self) -> str:
        if self.roi_source == 'auto':
            return 'refined' if self.use_refiner else 'raw'
        if self.roi_source == 'refined' and not self.use_refiner:
            return 'raw'
        return self.roi_source

    @staticmethod
    def _expand_boxes_centered(boxes: Tensor, img_h: int, img_w: int, ratio: float) -> Tensor:
        if boxes.numel() == 0 or ratio <= 1.0:
            return boxes
        cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
        cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
        bw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0) * ratio
        bh = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0) * ratio
        expanded = boxes.clone()
        expanded[:, 0] = (cx - 0.5 * bw).clamp(min=0, max=float(img_w))
        expanded[:, 1] = (cy - 0.5 * bh).clamp(min=0, max=float(img_h))
        expanded[:, 2] = (cx + 0.5 * bw).clamp(min=0, max=float(img_w))
        expanded[:, 3] = (cy + 0.5 * bh).clamp(min=0, max=float(img_h))
        return expanded

    def _build_rois_from_instances(self, pred_instances_list, batch_data_samples, expand_raw_ratio: float = 1.0):
        rois = []
        for img_idx, (pred, data_sample) in enumerate(zip(pred_instances_list, batch_data_samples)):
            if pred.bboxes.numel() == 0:
                continue
            scores = pred.scores if hasattr(pred, 'scores') else torch.ones(
                (pred.bboxes.size(0),), device=pred.bboxes.device)
            keep = scores >= self.roi_score_thr
            if not keep.any():
                continue
            roi_boxes = getattr(pred, 'roi_bboxes', None)
            if roi_boxes is not None and roi_boxes.shape == pred.bboxes.shape:
                boxes = roi_boxes[keep]
            else:
                boxes = pred.bboxes[keep]
            scores = scores[keep]
            order = torch.argsort(scores, descending=True)
            if self.roi_topk > 0:
                order = order[:self.roi_topk]
            boxes = boxes[order]
            img_h, img_w = data_sample.img_shape[:2]
            if expand_raw_ratio > 1.0:
                boxes = self._expand_boxes_centered(boxes, img_h, img_w, expand_raw_ratio)
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_h)
            valid = ((boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2))
            boxes = boxes[valid]
            if boxes.numel() == 0:
                continue
            batch_inds = torch.full((boxes.size(0), 1), float(img_idx), dtype=boxes.dtype, device=boxes.device)
            rois.append(torch.cat([batch_inds, boxes], dim=1))
        if len(rois) == 0:
            device = pred_instances_list[0].bboxes.device if len(pred_instances_list) > 0 else torch.device('cpu')
            return torch.zeros((0, 5), dtype=torch.float32, device=device)
        return torch.cat(rois, dim=0)

    def _build_external_local_model(self):
        if self.auto_sync_external_from_global:
            cfg = Config.fromfile(self.external_local_cfg)
            if hasattr(cfg, 'custom_imports'):
                import_modules_from_strings(**cfg.custom_imports)
            model_cfg = deepcopy(cfg.model)
            backbone_cfg = model_cfg.get('backbone', None)
            if backbone_cfg is not None and isinstance(backbone_cfg, dict):
                backbone_cfg = deepcopy(backbone_cfg)
                backbone_cfg['init_cfg'] = None
                model_cfg['backbone'] = backbone_cfg
            model = MODELS.build(model_cfg)
            model.to(torch.device(self.external_local_device))
            model.eval()
        else:
            model = init_detector(
                self.external_local_cfg,
                self.external_local_ckpt,
                device=self.external_local_device)
            model.eval()
        if self.external_local_frozen:
            for param in model.parameters():
                param.requires_grad_(False)
        return model

    def _collect_external_local_state(self):
        state = {}
        for key, value in self.data_preprocessor.state_dict().items():
            state[f'data_preprocessor.{key}'] = value.detach().cpu()
        for key, value in self.backbone.state_dict().items():
            state[f'backbone.{key}'] = value.detach().cpu()
        if self.with_neck and self.neck is not None:
            for key, value in self.neck.state_dict().items():
                state[f'neck.{key}'] = value.detach().cpu()
        for key, value in self.global_bbox_head.state_dict().items():
            state[f'bbox_head.{key}'] = value.detach().cpu()
        return state

    def _sync_external_local_from_global(self, model) -> None:
        load_msg = model.load_state_dict(self._collect_external_local_state(), strict=False)
        self._external_local_holder['last_sync_summary'] = dict(
            missing_keys=list(load_msg.missing_keys),
            unexpected_keys=list(load_msg.unexpected_keys),
        )
        self._external_local_holder['needs_sync'] = False
        if not self._external_local_holder.get('sync_log_printed', False):
            summary = self._external_local_holder['last_sync_summary']
            if summary['missing_keys'] or summary['unexpected_keys']:
                print_log(
                    '[AutoSyncExternalLocal] '
                    f'missing={len(summary["missing_keys"])} '
                    f'unexpected={len(summary["unexpected_keys"])}',
                    logger='current',
                    level=logging.WARNING,
                )
            else:
                print_log(
                    '[AutoSyncExternalLocal] external local FCOS synchronized from current global branch.',
                    logger='current',
                    level=logging.INFO,
                )
            self._external_local_holder['sync_log_printed'] = True

    def _ensure_external_local_model(self):
        if not self.enable_external_local:
            return None
        model = self._external_local_holder.get('model', None)
        if model is None:
            model = self._build_external_local_model()
            self._external_local_holder['model'] = model
            self._external_local_holder['needs_sync'] = True
        if self.auto_sync_external_from_global and self._external_local_holder.get('needs_sync', True):
            self._sync_external_local_from_global(model)
        model.eval()
        return model

    def _run_external_local_patches(self, patches: List[Tensor]):
        model = self._ensure_external_local_model()
        if model is None:
            device = patches[0].device if len(patches) > 0 else torch.device('cpu')
            return self._empty_instance_list(len(patches), device)
        if len(patches) == 0:
            return []
        device = torch.device(self.external_local_device)
        preds = []
        with torch.no_grad():
            for start in range(0, len(patches), self.ext_local_batch_size):
                chunk = patches[start:start + self.ext_local_batch_size]
                max_h = max(int(p.shape[-2]) for p in chunk)
                max_w = max(int(p.shape[-1]) for p in chunk)
                batch = torch.zeros((len(chunk), int(chunk[0].shape[1]), max_h, max_w), dtype=chunk[0].dtype, device=device)
                patch_samples = []
                for patch_idx, patch in enumerate(chunk):
                    ph = int(patch.shape[-2])
                    pw = int(patch.shape[-1])
                    batch[patch_idx, :, :ph, :pw] = patch[0].to(device=device, non_blocking=True)
                    sample = DetDataSample()
                    sample.set_metainfo(dict(
                        img_shape=(ph, pw),
                        ori_shape=(ph, pw),
                        scale_factor=(1.0, 1.0),
                        batch_input_shape=(max_h, max_w),
                        pad_shape=(max_h, max_w),
                    ))
                    patch_samples.append(sample)
                feats = model.extract_feat(batch)
                preds.extend(model.bbox_head.predict(feats, patch_samples, rescale=False))
        return preds

    def _build_stitched_canvas(self, patches: List[Tensor], patch_infos: List[Dict[str, float]]):
        cell_h, cell_w = self.stitched_cell_size
        cols = max(1, min(self.stitched_max_cols, int(math.ceil(math.sqrt(len(patches))))))
        rows = int(math.ceil(len(patches) / cols))
        gap = self.stitched_cell_gap
        canvas = torch.zeros(
            (1, int(patches[0].shape[1]), rows * cell_h + max(rows - 1, 0) * gap,
             cols * cell_w + max(cols - 1, 0) * gap),
            dtype=patches[0].dtype,
            device=patches[0].device)
        layouts = []
        for idx, (patch, patch_info) in enumerate(zip(patches, patch_infos)):
            row = idx // cols
            col = idx % cols
            cell_x = col * (cell_w + gap)
            cell_y = row * (cell_h + gap)
            orig_h = int(patch.shape[-2])
            orig_w = int(patch.shape[-1])
            scale = min(cell_w / max(float(orig_w), 1.0), cell_h / max(float(orig_h), 1.0))
            resized_w = max(1, min(cell_w, int(round(orig_w * scale))))
            resized_h = max(1, min(cell_h, int(round(orig_h * scale))))
            resized = F.interpolate(patch, size=(resized_h, resized_w), mode='bilinear', align_corners=False)
            canvas[:, :, cell_y:cell_y + resized_h, cell_x:cell_x + resized_w] = resized
            layout = dict(patch_info)
            layout.update(cell_x=float(cell_x), cell_y=float(cell_y), resized_w=float(resized_w), resized_h=float(resized_h), scale=float(scale))
            layouts.append(layout)
        return canvas, layouts

    @staticmethod
    def _find_layout_by_center(box: Tensor, layouts: List[Dict[str, float]]) -> int:
        cx = 0.5 * float(box[0].item() + box[2].item())
        cy = 0.5 * float(box[1].item() + box[3].item())
        for idx, layout in enumerate(layouts):
            if cx >= layout['cell_x'] and cx < layout['cell_x'] + layout['resized_w'] and cy >= layout['cell_y'] and cy < layout['cell_y'] + layout['resized_h']:
                return idx
        return -1

    def _predict_external_local_instances(self, batch_inputs: Tensor, rois: Tensor, batch_data_samples: SampleList):
        if not self.enable_external_local or rois.numel() == 0:
            return self._empty_instance_list(len(batch_data_samples), batch_inputs.device)
        _ = self._ensure_external_local_model()
        main_device = batch_inputs.device
        _, _, img_h_in, img_w_in = batch_inputs.shape
        rois_by_img = [[] for _ in range(len(batch_data_samples))]
        for roi in rois:
            rois_by_img[int(roi[0].item())].append(roi)

        canvases = []
        canvas_infos = []
        for img_idx, img_rois in enumerate(rois_by_img):
            patches = []
            patch_infos = []
            for roi in img_rois:
                x1, y1, x2, y2 = [float(v.item()) for v in roi[1:5]]
                ix1 = max(min(int(math.floor(x1)), img_w_in - 1), 0)
                iy1 = max(min(int(math.floor(y1)), img_h_in - 1), 0)
                ix2 = max(min(int(math.ceil(x2)), img_w_in), 0)
                iy2 = max(min(int(math.ceil(y2)), img_h_in), 0)
                if ix2 - ix1 < 2 or iy2 - iy1 < 2:
                    continue
                patch = batch_inputs[img_idx:img_idx + 1, :, iy1:iy2, ix1:ix2]
                if patch.numel() == 0:
                    continue
                patches.append(patch)
                patch_infos.append(dict(orig_x=float(ix1), orig_y=float(iy1), orig_w=float(ix2 - ix1), orig_h=float(iy2 - iy1)))
            if len(patches) == 0:
                continue
            canvases.append(self._build_stitched_canvas(patches, patch_infos)[0])
            canvas_infos.append((img_idx, self._build_stitched_canvas(patches, patch_infos)[1]))

        accum_boxes = [[] for _ in range(len(batch_data_samples))]
        accum_scores = [[] for _ in range(len(batch_data_samples))]
        accum_labels = [[] for _ in range(len(batch_data_samples))]
        for (img_idx, layouts), pred in zip(canvas_infos, self._run_external_local_patches(canvases)):
            if pred.bboxes.numel() == 0:
                continue
            keep = pred.scores >= self.ext_local_score_thr
            if not bool(keep.any()):
                continue
            boxes = pred.bboxes[keep]
            scores = pred.scores[keep]
            labels = pred.labels[keep]
            if self.ext_local_score_scale != 1.0:
                scores = (scores * self.ext_local_score_scale).clamp_(min=0.0, max=1.0)
            order = torch.argsort(scores, descending=True)
            if self.ext_local_max_per_patch > 0:
                order = order[:self.ext_local_max_per_patch]
            for box, score, label in zip(boxes[order], scores[order], labels[order]):
                layout_idx = self._find_layout_by_center(box, layouts)
                if layout_idx < 0:
                    continue
                layout = layouts[layout_idx]
                local = box.clone()
                local[0] -= layout['cell_x']; local[2] -= layout['cell_x']
                local[1] -= layout['cell_y']; local[3] -= layout['cell_y']
                local[0::2] = local[0::2].clamp(0, layout['resized_w'])
                local[1::2] = local[1::2].clamp(0, layout['resized_h'])
                scale = max(layout['scale'], 1e-6)
                mapped = local.clone()
                mapped[0::2] = mapped[0::2] / scale + layout['orig_x']
                mapped[1::2] = mapped[1::2] / scale + layout['orig_y']
                img_h, img_w = batch_data_samples[img_idx].img_shape[:2]
                mapped[0::2] = mapped[0::2].clamp(0, img_w)
                mapped[1::2] = mapped[1::2].clamp(0, img_h)
                if float(mapped[2].item()) <= float(mapped[0].item()) or float(mapped[3].item()) <= float(mapped[1].item()):
                    continue
                accum_boxes[img_idx].append(mapped.to(main_device))
                accum_scores[img_idx].append(score.to(main_device))
                accum_labels[img_idx].append(label.to(main_device))
        outputs = []
        for img_idx in range(len(batch_data_samples)):
            inst = InstanceData()
            if len(accum_boxes[img_idx]) == 0:
                inst.bboxes = torch.zeros((0, 4), device=main_device)
                inst.scores = torch.zeros((0,), device=main_device)
                inst.labels = torch.zeros((0,), dtype=torch.long, device=main_device)
            else:
                inst.bboxes = torch.stack(accum_boxes[img_idx], dim=0)
                inst.scores = torch.stack(accum_scores[img_idx], dim=0)
                inst.labels = torch.stack(accum_labels[img_idx], dim=0)
            outputs.append(inst)
        return outputs

    def _predict_external_local_per_roi_stitched(self, batch_inputs: Tensor, boxes: Tensor, batch_inds: Tensor, batch_data_samples: SampleList):
        outputs = self._empty_instance_list(int(boxes.shape[0]), batch_inputs.device)
        if not self.enable_external_local or boxes.numel() == 0:
            return outputs
        _ = self._ensure_external_local_model()
        main_device = batch_inputs.device
        _, _, img_h_in, img_w_in = batch_inputs.shape
        canvases = []
        canvas_infos = []
        for img_idx in range(len(batch_data_samples)):
            roi_indices = torch.nonzero(batch_inds == img_idx, as_tuple=False).squeeze(1)
            if roi_indices.numel() == 0:
                continue
            patches = []
            patch_infos = []
            for roi_idx in roi_indices.tolist():
                x1, y1, x2, y2 = [float(v.item()) for v in boxes[roi_idx]]
                ix1 = max(min(int(math.floor(x1)), img_w_in - 1), 0)
                iy1 = max(min(int(math.floor(y1)), img_h_in - 1), 0)
                ix2 = max(min(int(math.ceil(x2)), img_w_in), 0)
                iy2 = max(min(int(math.ceil(y2)), img_h_in), 0)
                if ix2 - ix1 < 2 or iy2 - iy1 < 2:
                    continue
                patch = batch_inputs[img_idx:img_idx + 1, :, iy1:iy2, ix1:ix2]
                if patch.numel() == 0:
                    continue
                patches.append(patch)
                patch_infos.append(dict(orig_x=float(ix1), orig_y=float(iy1), orig_w=float(ix2 - ix1), orig_h=float(iy2 - iy1), roi_idx=float(roi_idx)))
            if len(patches) == 0:
                continue
            canvas, layouts = self._build_stitched_canvas(patches, patch_infos)
            canvases.append(canvas)
            canvas_infos.append((img_idx, layouts))
        if len(canvases) == 0:
            return outputs
        accum_boxes = [[] for _ in range(int(boxes.shape[0]))]
        accum_scores = [[] for _ in range(int(boxes.shape[0]))]
        accum_labels = [[] for _ in range(int(boxes.shape[0]))]
        for (img_idx, layouts), pred in zip(canvas_infos, self._run_external_local_patches(canvases)):
            if pred.bboxes.numel() == 0:
                continue
            keep = pred.scores >= self.ext_local_score_thr
            if not bool(keep.any()):
                continue
            boxes_kept = pred.bboxes[keep]
            scores_kept = pred.scores[keep]
            labels_kept = pred.labels[keep]
            if self.ext_local_score_scale != 1.0:
                scores_kept = (scores_kept * self.ext_local_score_scale).clamp_(min=0.0, max=1.0)
            order = torch.argsort(scores_kept, descending=True)
            if self.ext_local_max_per_patch > 0:
                order = order[:self.ext_local_max_per_patch]
            for box, score, label in zip(boxes_kept[order], scores_kept[order], labels_kept[order]):
                layout_idx = self._find_layout_by_center(box, layouts)
                if layout_idx < 0:
                    continue
                layout = layouts[layout_idx]
                roi_idx = int(layout['roi_idx'])
                local = box.clone()
                local[0] -= layout['cell_x']; local[2] -= layout['cell_x']
                local[1] -= layout['cell_y']; local[3] -= layout['cell_y']
                local[0::2] = local[0::2].clamp(0, layout['resized_w'])
                local[1::2] = local[1::2].clamp(0, layout['resized_h'])
                if float(local[2].item()) <= float(local[0].item()):
                    continue
                if float(local[3].item()) <= float(local[1].item()):
                    continue
                scale = max(layout['scale'], 1e-6)
                mapped = local.clone()
                mapped[0::2] = mapped[0::2] / scale + layout['orig_x']
                mapped[1::2] = mapped[1::2] / scale + layout['orig_y']
                img_h, img_w = batch_data_samples[img_idx].img_shape[:2]
                mapped[0::2] = mapped[0::2].clamp(0, img_w)
                mapped[1::2] = mapped[1::2].clamp(0, img_h)
                if float(mapped[2].item()) <= float(mapped[0].item()):
                    continue
                if float(mapped[3].item()) <= float(mapped[1].item()):
                    continue
                accum_boxes[roi_idx].append(mapped.to(main_device))
                accum_scores[roi_idx].append(score.to(main_device))
                accum_labels[roi_idx].append(label.to(main_device))
        for roi_idx in range(len(outputs)):
            if len(accum_boxes[roi_idx]) == 0:
                continue
            outputs[roi_idx].bboxes = torch.stack(accum_boxes[roi_idx], dim=0)
            outputs[roi_idx].scores = torch.stack(accum_scores[roi_idx], dim=0)
            outputs[roi_idx].labels = torch.stack(accum_labels[roi_idx], dim=0)
        return outputs

    @staticmethod
    def _cpu_state_dict(module) -> Dict[str, Tensor]:
        if module is None:
            return {}
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    def maybe_update_policy_reward_baseline(self, score: float, epoch: int) -> bool:
        if score <= self._policy_reward_baseline_score:
            return False
        self._policy_reward_baseline_state = dict(
            backbone=self._cpu_state_dict(self.backbone),
            neck=self._cpu_state_dict(self.neck),
            cluster_head=self._cpu_state_dict(self.cluster_head),
            global_bbox_head=self._cpu_state_dict(self.global_bbox_head),
        )
        self._policy_reward_baseline_score = float(score)
        self._policy_reward_baseline_epoch = int(epoch)
        self._policy_reward_baseline_restored = False
        return True

    def restore_policy_reward_baseline(self) -> bool:
        if self._policy_reward_baseline_state is None:
            return False
        self.backbone.load_state_dict(self._policy_reward_baseline_state['backbone'])
        if self.neck is not None:
            self.neck.load_state_dict(self._policy_reward_baseline_state['neck'])
        self.cluster_head.load_state_dict(self._policy_reward_baseline_state['cluster_head'])
        self.global_bbox_head.load_state_dict(self._policy_reward_baseline_state['global_bbox_head'])
        self._policy_reward_baseline_restored = True
        self._external_local_holder['needs_sync'] = True
        self._external_local_holder['sync_log_printed'] = False
        return True

    def maybe_update_refiner_supervised_baseline(self, score: float, epoch: int) -> bool:
        if self.refiner is None or score <= self._refiner_sup_baseline_score:
            return False
        self._refiner_sup_baseline_state = dict(
            refiner=self._cpu_state_dict(self.refiner),
        )
        self._refiner_sup_baseline_score = float(score)
        self._refiner_sup_baseline_epoch = int(epoch)
        self._refiner_sup_baseline_restored = False
        return True

    def restore_refiner_supervised_baseline(self) -> bool:
        if self.refiner is None or self._refiner_sup_baseline_state is None:
            return False
        self.refiner.load_state_dict(self._refiner_sup_baseline_state['refiner'])
        self._refiner_sup_baseline_restored = True
        return True

    @staticmethod
    def _set_requires_grad(module, enabled: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad_(enabled)

    def _policy_zero_loss(self, ref_tensor: Tensor = None) -> Tensor:
        if ref_tensor is not None and ref_tensor.requires_grad:
            return ref_tensor.sum() * 0.0
        param = next(self.refiner.parameters(), None)
        if param is not None:
            return param.sum() * 0.0
        return torch.zeros((), device=ref_tensor.device) if ref_tensor is not None else torch.zeros(())

    def _zero_refiner_supervised_losses(self, ref_tensor: Optional[Tensor] = None):
        zero = self._policy_zero_loss(ref_tensor)
        return dict(
            refiner_sup_loss_box=zero,
            refiner_sup_iou=zero.detach(),
            refiner_sup_valid_ratio=zero.detach(),
            refiner_sup_num_valid=zero.detach(),
            refiner_sup_template_iou=zero.detach(),
        )

    def _is_template_policy_refiner(self, refiner=None) -> bool:
        module = self.refiner if refiner is None else refiner
        policy_mode = str(getattr(module, 'policy_mode', '')).lower()
        return policy_mode.startswith('template_categorical')

    def _clone_refiner(self, refiner=None):
        src = self.refiner if refiner is None else refiner
        cloned = deepcopy(src)
        device = next(self.refiner.parameters()).device
        cloned.to(device)
        cloned.eval()
        for param in cloned.parameters():
            param.requires_grad_(False)
        return cloned

    def _get_center_anchor_refiner(self):
        if self.training_stage == 'grpo' and self._grpo_center_anchor_refiner is not None:
            return self._grpo_center_anchor_refiner
        return self.refiner

    def maybe_update_grpo_reference(self, score: float, epoch: int) -> bool:
        # Fixed GRPO: reference policy disabled, always return False
        return False

    def set_training_stage(self, stage: str, mode: str = 'train') -> None:
        stage = str(stage).lower()
        if stage not in ('warmup', 'refiner_supervised', 'grpo'):
            raise ValueError(f'Unsupported training stage: {stage}')

        prev_stage = getattr(self, 'training_stage', 'warmup')
        if mode == 'train' and stage in ('refiner_supervised', 'grpo'):
            if not self._policy_reward_baseline_restored:
                self.restore_policy_reward_baseline()
            if stage == 'grpo' and self._refiner_sup_baseline_state is not None:
                if not self._refiner_sup_baseline_restored:
                    self.restore_refiner_supervised_baseline()

        self.training_stage = stage
        self.policy_stage_enabled = stage == 'grpo'
        self.refiner_supervised_stage_enabled = stage == 'refiner_supervised'
        self.use_refiner = bool(stage != 'warmup' and self.refiner is not None)

        if mode == 'train':
            train_refiner = stage in ('refiner_supervised', 'grpo')
            train_detector = stage == 'warmup'
            self._set_requires_grad(self.refiner, train_refiner)
            self._set_requires_grad(self.backbone, train_detector)
            self._set_requires_grad(self.neck, train_detector)
            self._set_requires_grad(self.cluster_head, train_detector)
            self._set_requires_grad(self.global_bbox_head, train_detector)
            if train_detector or train_refiner:
                self._external_local_holder['needs_sync'] = True

        if (
            stage == 'grpo' and
            mode == 'train' and
            prev_stage != 'grpo' and
            self.refiner is not None
        ):
            self._grpo_center_anchor_refiner = self._clone_refiner(self.refiner)
            # Fixed GRPO: reference policy disabled, no initialization needed
            pass

        if (
            stage == 'grpo' and
            mode == 'val' and
            self.refiner is not None and
            self._grpo_center_anchor_refiner is None
        ):
            self._grpo_center_anchor_refiner = self._clone_refiner(self.refiner)

    def set_policy_refiner_stage(self, enabled: bool, mode: str = 'train') -> None:
        self.set_training_stage('grpo' if bool(enabled) else 'warmup', mode=mode)

    def _limit_policy_proposals(self, proposal_instances):
        if self.policy_train_topk <= 0:
            return proposal_instances
        limited = []
        for proposal in proposal_instances:
            if proposal.bboxes.numel() == 0 or proposal.bboxes.shape[0] <= self.policy_train_topk:
                limited.append(proposal)
                continue
            order = torch.argsort(proposal.scores, descending=True)[:self.policy_train_topk]
            inst = InstanceData()
            inst.bboxes = proposal.bboxes[order]
            inst.scores = proposal.scores[order]
            inst.labels = proposal.labels[order]
            limited.append(inst)
        return limited

    def _match_flat_proposals(self, boxes: Tensor, batch_inds: Tensor, batch_data_samples: SampleList):
        assigned = batch_inds.new_full((boxes.shape[0],), -1)
        valid = torch.zeros((boxes.shape[0],), dtype=torch.bool, device=boxes.device)
        for img_idx, data_sample in enumerate(batch_data_samples):
            mask = batch_inds == img_idx
            if not bool(mask.any()):
                continue
            gt_cluster_boxes = self._get_gt_cluster_boxes(data_sample, device=boxes.device)
            img_assigned = self._match_one_to_one(boxes[mask], gt_cluster_boxes)
            assigned[mask] = img_assigned
            valid[mask] = img_assigned >= 0
        return assigned, valid

    def _repeat_batch_inds(self, batch_inds: Tensor, repeat: int) -> Tensor:
        return batch_inds[:, None].expand(-1, repeat).reshape(-1)

    @staticmethod
    def _box_cover_ratio(boxes: Tensor, targets: Tensor) -> Tensor:
        if boxes.numel() == 0:
            shape = boxes.shape[:-1]
            return boxes.new_zeros(shape)
        lt = torch.maximum(boxes[..., :2], targets[..., :2])
        rb = torch.minimum(boxes[..., 2:], targets[..., 2:])
        wh = (rb - lt).clamp(min=0.0)
        inter = wh[..., 0] * wh[..., 1]
        target_wh = (targets[..., 2:] - targets[..., :2]).clamp(min=1.0)
        target_area = target_wh[..., 0] * target_wh[..., 1]
        return inter / target_area.clamp(min=1.0)

    @staticmethod
    def _empty_cluster_response_stats(device: torch.device) -> Dict[str, Tensor]:
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return dict(
            inside_peak=zero,
            inside_mass=zero,
            inside_count=zero,
            purity=zero,
            outside_peak=zero,
            outside_mass=zero,
        )

    def _cluster_response_stats(
        self,
        pred_instances: InstanceData,
        gt_cluster_box: Tensor,
    ) -> Dict[str, Tensor]:
        device = gt_cluster_box.device
        if pred_instances is None or pred_instances.bboxes.numel() == 0:
            return self._empty_cluster_response_stats(device)

        pred_boxes = pred_instances.bboxes.to(device=device, dtype=torch.float32)
        pred_scores = pred_instances.scores.to(device=device, dtype=torch.float32)
        if pred_scores.numel() == 0:
            return self._empty_cluster_response_stats(device)

        centers_x = 0.5 * (pred_boxes[:, 0] + pred_boxes[:, 2])
        centers_y = 0.5 * (pred_boxes[:, 1] + pred_boxes[:, 3])
        inside_mask = (
            (centers_x >= gt_cluster_box[0]) & (centers_x <= gt_cluster_box[2]) &
            (centers_y >= gt_cluster_box[1]) & (centers_y <= gt_cluster_box[3]))

        inside_scores = pred_scores[inside_mask]
        outside_scores = pred_scores[~inside_mask]
        zero = pred_scores.new_zeros(())

        if inside_scores.numel() > 0:
            inside_topk = inside_scores.topk(min(3, int(inside_scores.numel()))).values.mean()
            inside_mass = torch.log1p(inside_scores.sum())
            inside_count = torch.log1p(inside_scores.new_tensor(float(inside_scores.numel())))
        else:
            inside_topk = zero
            inside_mass = zero
            inside_count = zero

        if outside_scores.numel() > 0:
            outside_topk = outside_scores.topk(min(3, int(outside_scores.numel()))).values.mean()
            outside_mass = torch.log1p(outside_scores.sum())
        else:
            outside_topk = zero
            outside_mass = zero

        total_mass = inside_scores.sum() + outside_scores.sum()
        purity = inside_scores.sum() / total_mass.clamp(min=1e-6)
        return dict(
            inside_peak=inside_topk,
            inside_mass=inside_mass,
            inside_count=inside_count,
            purity=purity,
            outside_peak=outside_topk,
            outside_mass=outside_mass,
        )

    def _cluster_response_quality(self, stats: Dict[str, Tensor]) -> Tensor:
        return (
            self.policy_reward_peak_weight * stats['inside_peak'] +
            self.policy_reward_mass_weight * stats['inside_mass'] +
            self.policy_reward_count_weight * stats['inside_count'] +
            self.policy_reward_purity_weight * stats['purity'] -
            self.policy_reward_fp_weight * stats['outside_peak'] -
            self.policy_reward_outside_mass_weight * stats['outside_mass']
        )

    def _gather_assigned_cluster_boxes(
        self,
        ref_boxes: Tensor,
        batch_inds: Tensor,
        batch_data_samples: SampleList,
        assigned_gt: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        cluster_boxes = ref_boxes.clone()
        for prop_idx in torch.nonzero(valid_mask, as_tuple=False).squeeze(1):
            img_idx = int(batch_inds[prop_idx].item())
            gt_cluster_boxes = self._get_gt_cluster_boxes(
                batch_data_samples[img_idx], device=ref_boxes.device)
            gt_idx = int(assigned_gt[prop_idx].item())
            if 0 <= gt_idx < gt_cluster_boxes.shape[0]:
                cluster_boxes[prop_idx] = gt_cluster_boxes[gt_idx]
        return cluster_boxes

    def _compute_grpo_reward(
        self,
        centered_boxes: Tensor,
        roi_boxes: Tensor,
        gt_cluster_boxes: Tensor,
        centered_patch_preds,
        refined_patch_preds,
        valid_mask: Tensor,
    ):
        num_props, group_size = roi_boxes.shape[:2]
        rewards = roi_boxes.new_zeros((num_props, group_size))
        det_gains = roi_boxes.new_zeros((num_props, group_size))
        cover_gains = roi_boxes.new_zeros((num_props, group_size))
        cover_penalties = roi_boxes.new_zeros((num_props, group_size))
        area_penalties = roi_boxes.new_zeros((num_props, group_size))
        count_gains = roi_boxes.new_zeros((num_props, group_size))
        purity_gains = roi_boxes.new_zeros((num_props, group_size))
        leak_gains = roi_boxes.new_zeros((num_props, group_size))
        raw_cover = self._box_cover_ratio(centered_boxes, gt_cluster_boxes)
        centered_areas = (
            (centered_boxes[:, 2] - centered_boxes[:, 0]).clamp(min=1.0) *
            (centered_boxes[:, 3] - centered_boxes[:, 1]).clamp(min=1.0)
        )
        for prop_idx in range(num_props):
            if not bool(valid_mask[prop_idx]):
                continue
            gt_cluster = gt_cluster_boxes[prop_idx]
            raw_stats = self._cluster_response_stats(
                centered_patch_preds[prop_idx], gt_cluster)
            raw_q = self._cluster_response_quality(raw_stats)
            for group_idx in range(group_size):
                ref_pred = refined_patch_preds[prop_idx * group_size + group_idx]
                refined_stats = self._cluster_response_stats(ref_pred, gt_cluster)
                refined_q = self._cluster_response_quality(refined_stats)
                roi_cover = self._box_cover_ratio(
                    roi_boxes[prop_idx, group_idx], gt_cluster)
                roi_area = (
                    (roi_boxes[prop_idx, group_idx, 2] - roi_boxes[prop_idx, group_idx, 0]).clamp(min=1.0) *
                    (roi_boxes[prop_idx, group_idx, 3] - roi_boxes[prop_idx, group_idx, 1]).clamp(min=1.0)
                )
                positive_cover_gain = F.relu(roi_cover - raw_cover[prop_idx])
                cover_penalty = F.relu(
                    raw_cover[prop_idx] * self.grpo_cover_keep_ratio - roi_cover)
                area_penalty = F.relu(
                    roi_area / centered_areas[prop_idx].clamp(min=1.0) - self.grpo_area_budget)
                det_gain = refined_q - raw_q
                cover_gain = roi_cover - raw_cover[prop_idx]
                count_gain = refined_stats['inside_count'] - raw_stats['inside_count']
                purity_gain = refined_stats['purity'] - raw_stats['purity']
                leak_gain = (
                    raw_stats['outside_peak'] + raw_stats['outside_mass'] -
                    refined_stats['outside_peak'] - refined_stats['outside_mass']
                )
                det_gains[prop_idx, group_idx] = det_gain
                cover_gains[prop_idx, group_idx] = cover_gain
                cover_penalties[prop_idx, group_idx] = cover_penalty
                area_penalties[prop_idx, group_idx] = area_penalty
                count_gains[prop_idx, group_idx] = count_gain
                purity_gains[prop_idx, group_idx] = purity_gain
                leak_gains[prop_idx, group_idx] = leak_gain
                rewards[prop_idx, group_idx] = (
                    self.policy_reward_det_weight * det_gain +
                    self.policy_reward_geo_weight * positive_cover_gain -
                    self.policy_cover_penalty_weight * cover_penalty -
                    self.policy_reward_area_weight * area_penalty
                ).clamp(min=-self.policy_reward_clip, max=self.policy_reward_clip)
        return dict(
            rewards=rewards,
            det_gain=det_gains,
            cover_gain=cover_gains,
            cover_penalty=cover_penalties,
            area_penalty=area_penalties,
            count_gain=count_gains,
            purity_gain=purity_gains,
            leak_gain=leak_gains,
        )

    def _compute_shift_targets(self, raw_boxes: Tensor, gt_envelopes: Tensor) -> Tensor:
        bw = (raw_boxes[:, 2] - raw_boxes[:, 0]).clamp(min=1.0)
        bh = (raw_boxes[:, 3] - raw_boxes[:, 1]).clamp(min=1.0)
        raw_cx = 0.5 * (raw_boxes[:, 0] + raw_boxes[:, 2])
        raw_cy = 0.5 * (raw_boxes[:, 1] + raw_boxes[:, 3])
        env_cx = 0.5 * (gt_envelopes[:, 0] + gt_envelopes[:, 2])
        env_cy = 0.5 * (gt_envelopes[:, 1] + gt_envelopes[:, 3])
        targets = raw_boxes.new_zeros((raw_boxes.shape[0], 2))
        targets[:, 0] = ((env_cx - raw_cx) / bw).clamp(min=-self.refiner.max_center_shift, max=self.refiner.max_center_shift)
        targets[:, 1] = ((env_cy - raw_cy) / bh).clamp(min=-self.refiner.max_center_shift, max=self.refiner.max_center_shift)
        return targets

    def _compute_margin_targets(self, shifted_boxes: Tensor, gt_envelopes: Tensor) -> Tensor:
        env = gt_envelopes[:, None, :]
        bw = (shifted_boxes[..., 2] - shifted_boxes[..., 0]).clamp(min=1.0)
        bh = (shifted_boxes[..., 3] - shifted_boxes[..., 1]).clamp(min=1.0)
        return torch.stack([
            ((shifted_boxes[..., 0] - env[..., 0]) / bw).clamp(min=-self.refiner.max_margin_scale, max=self.refiner.max_margin_scale),
            ((shifted_boxes[..., 1] - env[..., 1]) / bh).clamp(min=-self.refiner.max_margin_scale, max=self.refiner.max_margin_scale),
            ((env[..., 2] - shifted_boxes[..., 2]) / bw).clamp(min=-self.refiner.max_margin_scale, max=self.refiner.max_margin_scale),
            ((env[..., 3] - shifted_boxes[..., 3]) / bh).clamp(min=-self.refiner.max_margin_scale, max=self.refiner.max_margin_scale),
        ], dim=-1)

    def _compute_regularization_losses(self, current_outputs: Dict[str, Tensor], rollout: Dict[str, Tensor]):
        valid_mask = rollout['reg_valid_mask']
        valid_group_mask = valid_mask[:, None].expand_as(current_outputs['log_prob'])
        if valid_group_mask.numel() == 0 or not bool(valid_group_mask.any()):
            zero = self._policy_zero_loss()
            return dict(
                grpo_loss_shift_reg=zero, grpo_loss_margin_reg=zero, grpo_loss_cover_reg=zero,
                grpo_loss_area_reg=zero, grpo_loss_shift_mag=zero, grpo_loss_margin_mag=zero)
        raw_boxes = current_outputs['boxes']
        shifted_boxes = current_outputs['shifted_boxes']
        refined_boxes = current_outputs['refined_boxes']
        offsets = current_outputs['offsets']
        margins = current_outputs['margins']
        gt_envelopes = rollout['gt_envelopes']
        shift_targets = self._compute_shift_targets(raw_boxes, gt_envelopes)[:, None, :].expand_as(offsets)
        margin_targets = self._compute_margin_targets(shifted_boxes, gt_envelopes)
        env = gt_envelopes[:, None, :]
        raw_bw = (raw_boxes[:, 2] - raw_boxes[:, 0]).clamp(min=1.0)[:, None]
        raw_bh = (raw_boxes[:, 3] - raw_boxes[:, 1]).clamp(min=1.0)[:, None]
        cover_penalty = (
            F.relu(refined_boxes[..., 0] - env[..., 0]) / raw_bw +
            F.relu(refined_boxes[..., 1] - env[..., 1]) / raw_bh +
            F.relu(env[..., 2] - refined_boxes[..., 2]) / raw_bw +
            F.relu(env[..., 3] - refined_boxes[..., 3]) / raw_bh)
        env_area = ((env[..., 2] - env[..., 0]).clamp(min=1.0) * (env[..., 3] - env[..., 1]).clamp(min=1.0))
        refined_area = ((refined_boxes[..., 2] - refined_boxes[..., 0]).clamp(min=1.0) * (refined_boxes[..., 3] - refined_boxes[..., 1]).clamp(min=1.0))
        area_penalty = torch.relu(refined_area / env_area.clamp(min=1.0) - self.grpo_area_budget)
        return dict(
            grpo_loss_shift_reg=self.grpo_shift_reg_weight * F.smooth_l1_loss(offsets[valid_group_mask], shift_targets[valid_group_mask], reduction='mean'),
            grpo_loss_margin_reg=self.grpo_margin_reg_weight * F.smooth_l1_loss(margins[valid_group_mask], margin_targets[valid_group_mask], reduction='mean'),
            grpo_loss_cover_reg=self.grpo_cover_reg_weight * cover_penalty[valid_group_mask].mean(),
            grpo_loss_area_reg=self.grpo_area_reg_weight * area_penalty[valid_group_mask].mean(),
            grpo_loss_shift_mag=self.grpo_shift_mag_reg_weight * offsets[valid_group_mask].abs().mean(),
            grpo_loss_margin_mag=self.policy_margin_reg_weight * margins[valid_group_mask].abs().mean(),
        )

    def _normalize_group_advantages(self, rewards: Tensor, valid_mask: Tensor) -> Tensor:
        advantages = rewards.new_zeros(rewards.shape)
        if rewards.numel() == 0 or valid_mask.numel() == 0 or not bool(valid_mask.any()):
            return advantages
        valid_rewards = rewards[valid_mask]
        group_mean = valid_rewards.mean(dim=1, keepdim=True)
        centered = valid_rewards - group_mean
        group_std = valid_rewards.std(dim=1, keepdim=True, unbiased=False)
        normalized = centered / (group_std + self.grpo_advantage_eps)
        advantages[valid_mask] = torch.where(group_std <= self.grpo_advantage_eps, centered, normalized)
        return advantages

    def _collect_grpo_rollout(self, batch_inputs: Tensor, batch_data_samples: SampleList):
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
                feats=det_feats,
                proposal_instances=proposal_instances,
                batch_data_samples=batch_data_samples,
                use_expected=True,
            ) if self._is_template_policy_refiner(center_refiner) else proposal_instances

            if self._is_template_policy_refiner(old_refiner):
                old_outputs = old_refiner.sample_scale_context_groups(
                    feats=det_feats,
                    proposal_instances=center_proposals,
                    batch_data_samples=batch_data_samples,
                    group_size=self.grpo_group_size,
                    deterministic=False,
                )
            else:
                old_outputs = old_refiner.sample_action_groups(
                    feats=det_feats,
                    proposal_instances=center_proposals,
                    batch_data_samples=batch_data_samples,
                    group_size=self.grpo_group_size,
                    deterministic=False)
            if old_outputs['boxes'].numel() == 0:
                empty_valid = torch.zeros((0,), dtype=torch.bool, device=batch_inputs.device)
                return dict(
                    det_feats=det_feats, proposal_instances=center_proposals, batch_data_samples=batch_data_samples,
                    action_raw=old_outputs['action_raw'].detach(), old_log_prob=old_outputs['log_prob'].detach(),
                    rewards=old_outputs['log_prob'].detach(), advantages=old_outputs['log_prob'].detach(),
                    policy_valid_mask=empty_valid, reg_valid_mask=empty_valid, gt_envelopes=old_outputs['boxes'].detach())

            assigned_gt, matched_mask = self._match_flat_proposals(
                boxes=old_outputs['boxes'], batch_inds=old_outputs['batch_inds'], batch_data_samples=batch_data_samples)
            assigned_cluster_boxes = self._gather_assigned_cluster_boxes(
                ref_boxes=old_outputs['boxes'],
                batch_inds=old_outputs['batch_inds'],
                batch_data_samples=batch_data_samples,
                assigned_gt=assigned_gt,
                valid_mask=matched_mask,
            )
            policy_valid_mask = matched_mask
            reg_valid_mask = matched_mask
            center_patch_preds = self._predict_external_local_per_roi_stitched(
                batch_inputs=batch_inputs, boxes=old_outputs['boxes'], batch_inds=old_outputs['batch_inds'], batch_data_samples=batch_data_samples)
            rollout_boxes = old_outputs.get('roi_boxes', old_outputs['refined_boxes'])
            refined_patch_preds = self._predict_external_local_per_roi_stitched(
                batch_inputs=batch_inputs, boxes=rollout_boxes.reshape(-1, 4),
                batch_inds=self._repeat_batch_inds(old_outputs['batch_inds'], self.grpo_group_size),
                batch_data_samples=batch_data_samples)
            reward_outputs = self._compute_grpo_reward(
                centered_boxes=old_outputs['boxes'],
                roi_boxes=rollout_boxes,
                gt_cluster_boxes=assigned_cluster_boxes,
                centered_patch_preds=center_patch_preds,
                refined_patch_preds=refined_patch_preds,
                valid_mask=policy_valid_mask)
            rewards = reward_outputs['rewards']
            advantages = self._normalize_group_advantages(rewards, policy_valid_mask)
        return dict(
            det_feats=det_feats, proposal_instances=center_proposals, batch_data_samples=batch_data_samples,
            action_raw=old_outputs['action_raw'].detach(), old_log_prob=old_outputs['log_prob'].detach(),
            rewards=rewards.detach(), advantages=advantages.detach(), policy_valid_mask=policy_valid_mask.detach(),
            reg_valid_mask=reg_valid_mask.detach(), gt_envelopes=assigned_cluster_boxes.detach(),
            reward_det_gain=reward_outputs['det_gain'].detach(),
            reward_cover_gain=reward_outputs['cover_gain'].detach(),
            reward_cover_penalty=reward_outputs['cover_penalty'].detach(),
            reward_area_penalty=reward_outputs['area_penalty'].detach(),
            reward_count_gain=reward_outputs['count_gain'].detach(),
            reward_purity_gain=reward_outputs['purity_gain'].detach(),
            reward_leak_gain=reward_outputs['leak_gain'].detach())

    @staticmethod
    def _diagonal_gaussian_kl(current_mu: Tensor, current_log_std: Tensor, ref_mu: Tensor, ref_log_std: Tensor) -> Tensor:
        current_log_std = torch.nan_to_num(current_log_std, nan=0.0, posinf=0.0, neginf=0.0)
        ref_log_std = torch.nan_to_num(ref_log_std, nan=0.0, posinf=0.0, neginf=0.0)
        current_mu = torch.nan_to_num(current_mu, nan=0.0, posinf=0.0, neginf=0.0)
        ref_mu = torch.nan_to_num(ref_mu, nan=0.0, posinf=0.0, neginf=0.0)
        log_var_gap = 2.0 * (current_log_std - ref_log_std)
        inv_ref_var = torch.exp(-2.0 * ref_log_std)
        mean_gap_sq = (current_mu - ref_mu).pow(2)
        kl = ref_log_std - current_log_std + 0.5 * (torch.exp(log_var_gap) + mean_gap_sq * inv_ref_var - 1.0)
        return torch.nan_to_num(kl, nan=0.0, posinf=1e6, neginf=0.0).sum(dim=-1)

    def _stabilize_reference_kl(self, raw_kl: Tensor) -> Tensor:
        safe_kl = torch.nan_to_num(raw_kl, nan=0.0, posinf=self.grpo_ref_kl_safe_max, neginf=0.0).clamp(min=0.0, max=self.grpo_ref_kl_safe_max)
        scale = safe_kl.new_tensor(self.grpo_ref_kl_loss_scale)
        return scale * torch.log1p(safe_kl / scale) if self.grpo_ref_kl_loss_scale > 0 else safe_kl

    @staticmethod
    def _categorical_policy_kl(current_logits: Tensor, ref_logits: Tensor) -> Tensor:
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        current_probs = current_log_probs.exp()
        kl = (current_probs * (current_log_probs - ref_log_probs)).sum(dim=-1)
        return torch.nan_to_num(kl, nan=0.0, posinf=1e6, neginf=0.0)

    def _compute_refiner_supervised_losses(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
    ):
        with torch.no_grad():
            backbone_feats, det_feats = self._extract_backbone_and_det_feats(batch_inputs)
            det_feats = tuple(feat.detach() for feat in det_feats)
            c5_feats = (backbone_feats[-1].detach(),)
            cluster_logits = self.cluster_head(c5_feats)
            proposal_instances = self._filter_train_proposals(
                self._get_merged_cluster_instances_by_feat(cluster_logits.detach(), batch_data_samples),
                batch_data_samples,
            )
            if self.refiner_sup_use_policy_topk:
                proposal_instances = self._limit_policy_proposals(proposal_instances)

        if self._is_template_policy_refiner():
            current_outputs = self.refiner.enumerate_center_groups(
                feats=det_feats,
                proposal_instances=proposal_instances,
                batch_data_samples=batch_data_samples,
            )
        else:
            return self._zero_refiner_supervised_losses()

        if current_outputs['boxes'].numel() == 0:
            return self._zero_refiner_supervised_losses()

        assigned_gt, matched_mask = self._match_flat_proposals(
            boxes=current_outputs['boxes'],
            batch_inds=current_outputs['batch_inds'],
            batch_data_samples=batch_data_samples,
        )
        gt_cluster_boxes = self._gather_assigned_cluster_boxes(
            ref_boxes=current_outputs['boxes'],
            batch_inds=current_outputs['batch_inds'],
            batch_data_samples=batch_data_samples,
            assigned_gt=assigned_gt,
            valid_mask=matched_mask,
        )
        valid_mask = matched_mask
        if valid_mask.numel() == 0 or not bool(valid_mask.any()):
            return self._zero_refiner_supervised_losses(
                current_outputs.get('center_logits', current_outputs['embed']))

        raw_valid = current_outputs['boxes'][valid_mask]
        target_valid = gt_cluster_boxes[valid_mask]
        expected_valid = current_outputs['expected_center_boxes'][valid_mask]
        best_valid = current_outputs['best_center_boxes'][valid_mask]
        raw_bw = (raw_valid[:, 2] - raw_valid[:, 0]).clamp(min=1.0)
        raw_bh = (raw_valid[:, 3] - raw_valid[:, 1]).clamp(min=1.0)
        norm = torch.stack([raw_bw, raw_bh], dim=1)
        expected_center = torch.stack([
            0.5 * (expected_valid[:, 0] + expected_valid[:, 2]),
            0.5 * (expected_valid[:, 1] + expected_valid[:, 3]),
        ], dim=1)
        target_center = torch.stack([
            0.5 * (target_valid[:, 0] + target_valid[:, 2]),
            0.5 * (target_valid[:, 1] + target_valid[:, 3]),
        ], dim=1)
        expected_iou = bbox_overlaps(expected_valid, target_valid, is_aligned=True)
        best_iou = bbox_overlaps(best_valid, target_valid, is_aligned=True)
        return dict(
            refiner_sup_loss_box=self.refiner_sup_center_weight * F.smooth_l1_loss(
                expected_center / norm,
                target_center / norm,
                reduction='mean',
            ),
            refiner_sup_iou=expected_iou.mean().detach(),
            refiner_sup_valid_ratio=valid_mask.float().mean().detach(),
            refiner_sup_num_valid=valid_mask.float().sum().detach(),
            refiner_sup_template_iou=best_iou.mean().detach(),
        )

    def _compute_grpo_losses(self, rollout: Dict[str, Tensor]):
        if self._is_template_policy_refiner():
            current_outputs = self.refiner.evaluate_scale_context_groups(
                feats=rollout['det_feats'],
                proposal_instances=rollout['proposal_instances'],
                batch_data_samples=rollout['batch_data_samples'],
                action_raw=rollout['action_raw'])
        else:
            current_outputs = self.refiner.evaluate_action_groups(
                feats=rollout['det_feats'],
                proposal_instances=rollout['proposal_instances'],
                batch_data_samples=rollout['batch_data_samples'],
                action_raw=rollout['action_raw'])
        valid_mask = rollout['policy_valid_mask']
        if valid_mask.numel() == 0 or not bool(valid_mask.any()):
            zero = self._policy_zero_loss()
            return dict(
                grpo_loss_policy=zero, grpo_loss_entropy=zero, grpo_loss_shift_reg=zero,
                grpo_loss_margin_reg=zero, grpo_loss_cover_reg=zero, grpo_loss_area_reg=zero,
                grpo_loss_shift_mag=zero, grpo_loss_margin_mag=zero, grpo_loss_kl=zero,
                grpo_reward_mean=zero, grpo_ratio_mean=zero, grpo_clipfrac=zero,
                grpo_approx_kl=zero, grpo_approx_kl_raw=zero,
                grpo_det_gain_mean=zero, grpo_cover_gain_mean=zero,
                grpo_cover_penalty_mean=zero, grpo_area_penalty_mean=zero, grpo_count_gain_mean=zero,
                grpo_purity_gain_mean=zero, grpo_leak_gain_mean=zero)
        valid_group_mask = valid_mask[:, None].expand_as(rollout['advantages'])
        advantages = rollout['advantages'][valid_group_mask]
        old_log_prob = rollout['old_log_prob'][valid_group_mask]
        current_log_prob = current_outputs['log_prob'][valid_group_mask]
        ratio = torch.exp(current_log_prob - old_log_prob)
        if self.grpo_policy_clip_enabled:
            clipped_ratio = ratio.clamp(1.0 - self.grpo_clip_eps, 1.0 + self.grpo_clip_eps)
            loss_policy = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
            clipfrac = (ratio.detach() != clipped_ratio.detach()).to(ratio.dtype).mean()
        else:
            loss_policy = -(ratio * advantages).mean()
            clipfrac = ratio.detach().new_zeros(())
        if self._is_template_policy_refiner():
            zero_reg = self._policy_zero_loss(
                current_outputs.get('scale_context_logits', current_outputs.get('policy_logits')))
            reg_losses = dict(
                grpo_loss_shift_reg=zero_reg,
                grpo_loss_margin_reg=zero_reg,
                grpo_loss_cover_reg=zero_reg,
                grpo_loss_area_reg=zero_reg,
                grpo_loss_shift_mag=zero_reg,
                grpo_loss_margin_mag=zero_reg,
            )
        else:
            reg_losses = self._compute_regularization_losses(current_outputs, rollout)
        loss_entropy = -self.grpo_entropy_weight * current_outputs['entropy'][valid_group_mask].mean()
        # Fixed GRPO: reference policy disabled, KL always zero
        approx_kl_mean = loss_policy.detach().new_zeros(())
        approx_kl_raw_mean = approx_kl_mean
        loss_kl = approx_kl_mean
        losses = dict(
            grpo_loss_policy=loss_policy, grpo_loss_entropy=loss_entropy, grpo_loss_kl=loss_kl,
            grpo_reward_mean=rollout['rewards'][valid_group_mask].mean(),
            grpo_ratio_mean=ratio.mean().detach(), grpo_clipfrac=clipfrac,
            grpo_approx_kl=approx_kl_mean.detach(), grpo_approx_kl_raw=approx_kl_raw_mean.detach(),
            grpo_det_gain_mean=rollout['reward_det_gain'][valid_group_mask].mean().detach(),
            grpo_cover_gain_mean=rollout['reward_cover_gain'][valid_group_mask].mean().detach(),
            grpo_cover_penalty_mean=rollout['reward_cover_penalty'][valid_group_mask].mean().detach(),
            grpo_area_penalty_mean=rollout['reward_area_penalty'][valid_group_mask].mean().detach(),
            grpo_count_gain_mean=rollout['reward_count_gain'][valid_group_mask].mean().detach(),
            grpo_purity_gain_mean=rollout['reward_purity_gain'][valid_group_mask].mean().detach(),
            grpo_leak_gain_mean=rollout['reward_leak_gain'][valid_group_mask].mean().detach())
        losses.update(reg_losses)
        return losses

    @staticmethod
    def _average_log_vars(log_vars_list):
        if len(log_vars_list) == 1:
            return log_vars_list[0]
        averaged = OrderedDict()
        for key in log_vars_list[0].keys():
            averaged[key] = sum(log_vars[key] for log_vars in log_vars_list) / len(log_vars_list)
        return averaged

    def _standard_train_step(self, data: Dict, optim_wrapper: OptimWrapper):
        with optim_wrapper.optim_context(self):
            losses = self._run_forward(data, mode='loss')
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def _supervised_refiner_train_step(self, data: Dict, optim_wrapper: OptimWrapper):
        with optim_wrapper.optim_context(self):
            losses = self._compute_refiner_supervised_losses(
                batch_inputs=data['inputs'],
                batch_data_samples=data['data_samples'],
            )
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def train_step(self, data: Dict, optim_wrapper: OptimWrapper):
        data = self.data_preprocessor(data, True)
        if self.training_stage == 'warmup':
            return self._standard_train_step(data, optim_wrapper)
        if self.training_stage == 'refiner_supervised':
            return self._supervised_refiner_train_step(data, optim_wrapper)
        rollout = self._collect_grpo_rollout(
            batch_inputs=data['inputs'], batch_data_samples=data['data_samples'])
        if rollout['policy_valid_mask'].numel() == 0 or not bool(rollout['policy_valid_mask'].any()):
            zero = self._policy_zero_loss()
            return dict(
                grpo_loss_policy=zero.detach(), grpo_loss_entropy=zero.detach(), grpo_loss_shift_reg=zero.detach(),
                grpo_loss_margin_reg=zero.detach(), grpo_loss_cover_reg=zero.detach(), grpo_loss_area_reg=zero.detach(),
                grpo_loss_shift_mag=zero.detach(), grpo_loss_margin_mag=zero.detach(), grpo_loss_kl=zero.detach(),
                grpo_reward_mean=zero.detach(), grpo_ratio_mean=zero.detach(), grpo_clipfrac=zero.detach(),
                grpo_approx_kl=zero.detach(), grpo_approx_kl_raw=zero.detach(),
                grpo_det_gain_mean=zero.detach(), grpo_cover_gain_mean=zero.detach(),
                grpo_cover_penalty_mean=zero.detach(), grpo_area_penalty_mean=zero.detach(), grpo_count_gain_mean=zero.detach(),
                grpo_purity_gain_mean=zero.detach(), grpo_leak_gain_mean=zero.detach())
        log_vars_list = []
        for _ in range(self.grpo_update_steps):
            with optim_wrapper.optim_context(self):
                losses = self._compute_grpo_losses(rollout)
            parsed_losses, log_vars = self.parse_losses(losses)
            optim_wrapper.update_params(parsed_losses)
            log_vars_list.append(log_vars)
        return self._average_log_vars(log_vars_list)

    def _merge_instances_by_class(self, all_boxes, all_scores, all_labels):
        device = all_boxes[0].device if all_boxes else torch.device('cpu')
        boxes = torch.cat(all_boxes, dim=0) if all_boxes else torch.zeros((0, 4), device=device)
        scores = torch.cat(all_scores, dim=0) if all_scores else torch.zeros((0,), device=device)
        labels = torch.cat(all_labels, dim=0) if all_labels else torch.zeros((0,), dtype=torch.long, device=device)
        merged = InstanceData()
        if boxes.numel() == 0:
            merged.bboxes = boxes; merged.scores = scores; merged.labels = labels
            return merged
        keep_parts = []
        for label in torch.unique(labels):
            mask = labels == label
            label_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            keep_parts.append(label_indices[torchvision_nms(boxes[label_indices], scores[label_indices], self.merge_nms_iou_thr)])
        keep = torch.cat(keep_parts, dim=0)
        keep = keep[torch.argsort(scores[keep], descending=True)]
        max_per_img = self.test_cfg.get('max_per_img', None) if isinstance(self.test_cfg, dict) else getattr(self.test_cfg, 'max_per_img', None)
        if max_per_img is not None and max_per_img > 0:
            keep = keep[:int(max_per_img)]
        merged.bboxes = boxes[keep]
        merged.scores = scores[keep]
        merged.labels = labels[keep]
        return merged

    @staticmethod
    def _rescale_instances_to_ori(pred_instances_list, batch_data_samples):
        for pred_instances, data_sample in zip(pred_instances_list, batch_data_samples):
            if pred_instances.bboxes.numel() == 0 or not hasattr(data_sample, 'ori_shape'):
                continue
            scale_factor = getattr(data_sample, 'scale_factor', None)
            if scale_factor is None and hasattr(data_sample, 'metainfo'):
                scale_factor = data_sample.metainfo.get('scale_factor', None)
            if scale_factor is not None:
                if hasattr(scale_factor, 'tolist'):
                    scale_factor = scale_factor.tolist()
                if not isinstance(scale_factor, (list, tuple)):
                    scale_factor = [float(scale_factor)]
                w_scale = float(scale_factor[0])
                h_scale = float(scale_factor[1] if len(scale_factor) >= 2 else scale_factor[0])
                if abs(w_scale - 1.0) < 1e-6 and abs(h_scale - 1.0) < 1e-6:
                    continue
                w_scale = 1.0 / w_scale
                h_scale = 1.0 / h_scale
            else:
                ori_h, ori_w = data_sample.ori_shape[:2]
                img_h, img_w = data_sample.img_shape[:2]
                if ori_h == img_h and ori_w == img_w:
                    continue
                w_scale = float(ori_w) / float(img_w)
                h_scale = float(ori_h) / float(img_h)
            pred_instances.bboxes[:, 0] *= w_scale
            pred_instances.bboxes[:, 2] *= w_scale
            pred_instances.bboxes[:, 1] *= h_scale
            pred_instances.bboxes[:, 3] *= h_scale
        return pred_instances_list

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        losses = self._base_detector_loss(batch_inputs, batch_data_samples)
        self._external_local_holder['needs_sync'] = True
        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = False):
        backbone_feats, det_feats = self._extract_backbone_and_det_feats(batch_inputs)
        c5_feats = (backbone_feats[-1],)
        cluster_logits = self.cluster_head(c5_feats) if self.use_cluster_head else None
        global_instances = self.global_bbox_head.predict(det_feats, batch_data_samples, rescale=False)
        if not self.use_cluster_head:
            cluster_instances = self._empty_instance_list(len(batch_data_samples), batch_inputs.device)
            refined_instances = cluster_instances
            local_instances = self._empty_instance_list(len(batch_data_samples), batch_inputs.device)
            merged_instances = global_instances
        else:
            cluster_instances = self._get_merged_cluster_instances_by_feat(cluster_logits, batch_data_samples)
            if self.use_refiner:
                if self._is_template_policy_refiner():
                    if self.training_stage == 'grpo':
                        center_refiner = self._get_center_anchor_refiner()
                        centered_instances = center_refiner.predict_center_instances(
                            feats=det_feats,
                            proposal_instances=cluster_instances,
                            batch_data_samples=batch_data_samples,
                            use_expected=True,
                        )
                        refined_instances = self.refiner.predict_scale_context_instances(
                            feats=det_feats,
                            proposal_instances=centered_instances,
                            batch_data_samples=batch_data_samples,
                            use_expected=False,
                        )
                    else:
                        refined_instances = self.refiner.predict_center_instances(
                            feats=det_feats,
                            proposal_instances=cluster_instances,
                            batch_data_samples=batch_data_samples,
                            use_expected=True,
                        )
                else:
                    refined_instances = self.refiner.predict_instances(
                        feats=det_feats, proposal_instances=cluster_instances, batch_data_samples=batch_data_samples)
            else:
                refined_instances = cluster_instances
            resolved_source = self._resolve_roi_source()
            roi_instances = refined_instances if resolved_source == 'refined' else cluster_instances
            expand_ratio = 1.0 if resolved_source == 'refined' else self.raw_roi_expand_ratio
            rois = self._build_rois_from_instances(roi_instances, batch_data_samples, expand_raw_ratio=expand_ratio)
            local_instances = self._predict_external_local_instances(
                batch_inputs=batch_inputs, rois=rois, batch_data_samples=batch_data_samples)
            merged_instances = []
            for global_pred, local_pred in zip(global_instances, local_instances):
                all_boxes = [global_pred.bboxes]
                all_scores = [global_pred.scores]
                all_labels = [global_pred.labels]
                if local_pred.bboxes.numel() > 0:
                    all_boxes.append(local_pred.bboxes)
                    all_scores.append(local_pred.scores)
                    all_labels.append(local_pred.labels)
                merged_instances.append(self._merge_instances_by_class(all_boxes, all_scores, all_labels))
        if rescale:
            global_instances = self._rescale_instances_to_ori(global_instances, batch_data_samples)
            cluster_instances = self._rescale_instances_to_ori(cluster_instances, batch_data_samples)
            refined_instances = self._rescale_instances_to_ori(refined_instances, batch_data_samples)
            local_instances = self._rescale_instances_to_ori(local_instances, batch_data_samples)
            merged_instances = self._rescale_instances_to_ori(merged_instances, batch_data_samples)
        for idx, data_sample in enumerate(batch_data_samples):
            data_sample.pred_global_instances = global_instances[idx]
            data_sample.pred_cluster_instances = cluster_instances[idx]
            data_sample.pred_refined_cluster_instances = refined_instances[idx]
            data_sample.pred_local_instances = local_instances[idx]
        return self.add_pred_to_datasample(batch_data_samples, merged_instances)
