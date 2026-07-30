from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import DeformConv2d, batched_nms
from mmengine.config import ConfigDict
from mmengine.model import bias_init_with_prob, normal_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.models.utils import (gaussian_radius, gen_gaussian_target,
                                get_local_maximum, multi_apply,
                                transpose_and_gather_feat)
from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig)


def _gaussian_blur2d(x: Tensor, kernel_size: int = 3, sigma: float = 1.0) -> Tensor:
    """Small torch-only replacement for kornia gaussian blur."""
    if kernel_size <= 1:
        return x

    radius = kernel_size // 2
    coords = torch.arange(
        kernel_size, device=x.device, dtype=x.dtype) - float(radius)
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(
        x.size(1), 1, 1, 1)
    return F.conv2d(x, kernel, padding=radius, groups=x.size(1))


@MODELS.register_module()
class YOLCHead(BaseDenseHead):
    """mmdet3-compatible YOLC head for tiny-object infrared detection."""

    def __init__(self,
                 in_channels: int,
                 feat_channels: int,
                 num_classes: int,
                 loss_center_local: ConfigType = dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 loss_xywh: ConfigType = dict(
                     type='YOLCGWDLoss', loss_weight=2.0),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.local_head = self._build_local_head(in_channels, num_classes)
        self._build_reg_head(in_channels, feat_channels)

        self.loss_center_local = MODELS.build(loss_center_local)
        self.loss_xywh_coarse = MODELS.build(loss_xywh)
        self.loss_xywh_refine = MODELS.build(loss_xywh)
        self.loss_xywh_coarse_l1 = MODELS.build(
            dict(type='L1Loss', loss_weight=0.5))
        self.loss_xywh_refine_l1 = MODELS.build(
            dict(type='L1Loss', loss_weight=0.5))

        self.prior_generator = MlvlPointGenerator(strides=[1], offset=0)

        dcn_base = torch.arange(-1, 2, dtype=torch.float32)
        dcn_base_y = dcn_base.repeat_interleave(3)
        dcn_base_x = dcn_base.repeat(3)
        dcn_base_offset = torch.stack([dcn_base_y, dcn_base_x],
                                      dim=1).reshape(-1)
        self.register_buffer(
            'dcn_base_offset',
            dcn_base_offset.view(1, -1, 1, 1),
            persistent=False,
        )

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _build_local_head(self, in_channels: int,
                          out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, self.num_classes * 8, 4, stride=2,
                               padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.num_classes * 8, self.num_classes * 8, 4,
                               stride=2, padding=1),
            nn.Conv2d(
                self.num_classes * 8,
                out_channels,
                kernel_size=1,
                groups=self.num_classes,
            ),
        )

    def _build_reg_head(self, in_channels: int, feat_channels: int) -> None:
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1),
        )
        self.xywh_init = nn.Conv2d(feat_channels, 4, kernel_size=1)
        self.bbox_offset = nn.Conv2d(feat_channels, 18, kernel_size=1)
        self.xywh_refine = DeformConv2d(
            feat_channels, 4, kernel_size=3, padding=1)

    def init_weights(self) -> None:
        super().init_weights()
        for head in [
                self.local_head, self.reg_conv, self.xywh_init,
                self.bbox_offset, self.xywh_refine
        ]:
            for module in head.modules():
                if isinstance(module, (nn.Conv2d, DeformConv2d)):
                    normal_init(module, std=0.001)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.ConvTranspose2d):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

        if self.local_head[-1].bias is not None:
            self.local_head[-1].bias.data.fill_(bias_init_with_prob(0.1))

    def forward(self, feats: Tuple[Tensor, ...]) -> Tuple[List[Tensor], ...]:
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        center_local_pred = self.local_head(feat).sigmoid()

        reg_feat = self.reg_conv(feat).contiguous()
        xywh_pred_coarse = self.xywh_init(reg_feat)

        featmap_size = xywh_pred_coarse.shape[-2:]
        center_points = self.prior_generator.grid_priors(
            [featmap_size],
            device=xywh_pred_coarse.device,
            with_stride=False)[0].to(dtype=xywh_pred_coarse.dtype)
        center_grid = center_points.transpose(0, 1).reshape(
            1, 2, featmap_size[0], featmap_size[1])
        bbox_for_dcn = torch.cat(
            (xywh_pred_coarse[:, 0:2] + center_grid, xywh_pred_coarse[:, 2:4]),
            dim=1).detach()

        offset = self.bbox_offset(reg_feat).sigmoid()
        dcn_offset = self.gen_dcn_offset(bbox_for_dcn, offset, center_points)
        xywh_pred_refine = self.xywh_refine(reg_feat, dcn_offset)
        return center_local_pred, xywh_pred_coarse, xywh_pred_refine

    def gen_dcn_offset(self, bbox_pred: Tensor, offset: Tensor,
                       center_points: Tensor) -> Tensor:
        b, _, h, w = offset.shape
        dcn_offset = offset.new_zeros(b, 18, h, w)
        box = bbox_pred.clone()
        box[:, 0:2] = box[:, 0:2] - box[:, 2:4]
        box[:, 2:4] = 2 * box[:, 2:4]

        dcn_offset[:, 0::2] = box[:, 0:1] + box[:, 2:3] * offset[:, 0::2]
        dcn_offset[:, 1::2] = box[:, 1:2] + box[:, 3:4] * offset[:, 1::2]

        anchor_offset = center_points.reshape(h, w, 2).repeat(b, 1, 1, 9)
        anchor_offset = anchor_offset.permute(0, 3, 1, 2).contiguous()
        anchor_offset = anchor_offset + self.dcn_base_offset.to(
            device=dcn_offset.device, dtype=dcn_offset.dtype)
        return dcn_offset - anchor_offset

    def loss_by_feat(
            self,
            center_local_preds: List[Tensor],
            xywh_preds_coarse: List[Tensor],
            xywh_preds_refine: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        assert len(center_local_preds) == len(xywh_preds_coarse) == len(
            xywh_preds_refine) == 1

        center_local_pred = center_local_preds[0]
        xywh_pred_coarse = xywh_preds_coarse[0]
        xywh_pred_refine = xywh_preds_refine[0]

        gt_bboxes = [
            get_box_tensor(gt_instances.bboxes).to(center_local_pred.device)
            for gt_instances in batch_gt_instances
        ]
        gt_labels = [
            gt_instances.labels.to(center_local_pred.device)
            for gt_instances in batch_gt_instances
        ]
        img_shape = batch_img_metas[0].get('batch_input_shape',
                                           batch_img_metas[0]['img_shape'])
        target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels,
                                                     xywh_pred_coarse.shape,
                                                     img_shape)

        center_points = self.prior_generator.grid_priors(
            [xywh_pred_coarse.shape[-2:]],
            device=xywh_pred_coarse.device,
            with_stride=False)[0].to(dtype=xywh_pred_coarse.dtype)

        bbox_pred_coarse = self._decode_bbox_map(xywh_pred_coarse, center_points)
        bbox_pred_refine = self._decode_bbox_map(xywh_pred_refine, center_points)

        xywh_target = target_result['xywh_target'].reshape(
            xywh_pred_coarse.size(0), -1, 4)
        xywh_target_weight = target_result['xywh_target_weight'].reshape(
            xywh_pred_coarse.size(0), -1)
        xywh_l1target_weight = target_result['xywh_l1target_weight'].reshape(
            xywh_pred_coarse.size(0), -1, 4)

        loss_center_heatmap = self.loss_center_local(
            center_local_pred,
            target_result['center_heatmap_target'],
            avg_factor=avg_factor,
        )
        loss_xywh_coarse = self.loss_xywh_coarse(
            bbox_pred_coarse,
            xywh_target,
            xywh_target_weight,
            avg_factor=avg_factor,
        )
        loss_xywh_refine = self.loss_xywh_refine(
            bbox_pred_refine,
            xywh_target,
            xywh_target_weight,
            avg_factor=avg_factor,
        )
        loss_xywh_coarse_l1 = self.loss_xywh_coarse_l1(
            bbox_pred_coarse,
            xywh_target,
            xywh_l1target_weight,
            avg_factor=avg_factor,
        )
        loss_xywh_refine_l1 = self.loss_xywh_refine_l1(
            bbox_pred_refine,
            xywh_target,
            xywh_l1target_weight,
            avg_factor=avg_factor,
        )
        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_xywh_coarse=loss_xywh_coarse,
            loss_xywh_coarse_l1=loss_xywh_coarse_l1,
            loss_xywh_refine=loss_xywh_refine,
            loss_xywh_refine_l1=loss_xywh_refine_l1,
        )

    def get_targets(self, gt_bboxes: List[Tensor], gt_labels: List[Tensor],
                    feat_shape: Tuple[int, ...],
                    img_shape: Tuple[int, ...]) -> Tuple[dict, int]:
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        template = gt_bboxes[0]
        center_heatmap_target = template.new_zeros(
            (bs, self.num_classes, feat_h * 4, feat_w * 4))
        xywh_target = template.new_zeros((bs, feat_h, feat_w, 4))
        xywh_target_weight = template.new_zeros((bs, feat_h, feat_w))
        xywh_l1target_weight = template.new_zeros((bs, feat_h, feat_w, 4))

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            if gt_bbox.numel() == 0:
                continue
            gt_label = gt_labels[batch_id]

            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            origin_center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) / 2
            origin_center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) / 2
            origin_gt_centers = torch.cat((origin_center_x, origin_center_y),
                                          dim=1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.long()
                ctx, cty = ct
                box_h = gt_bbox[j][3] - gt_bbox[j][1]
                box_w = gt_bbox[j][2] - gt_bbox[j][0]
                scale_box_h = box_h * height_ratio
                scale_box_w = box_w * width_ratio

                radius = max(0, int(gaussian_radius([box_h, box_w],
                                                    min_overlap=0.3)))
                cls_id = int(gt_label[j].item())
                ori_ctx_int, ori_cty_int = origin_gt_centers[j].long()
                if 0 <= ori_ctx_int < feat_w * 4 and 0 <= ori_cty_int < feat_h * 4:
                    gen_gaussian_target(
                        center_heatmap_target[batch_id, cls_id],
                        [int(ori_ctx_int.item()), int(ori_cty_int.item())],
                        radius,
                    )

                if cty_int >= feat_h or ctx_int >= feat_w or cty_int < 0 or ctx_int < 0:
                    continue

                xywh_target[batch_id, cty_int, ctx_int, 0] = ctx
                xywh_target[batch_id, cty_int, ctx_int, 1] = cty
                xywh_target[batch_id, cty_int, ctx_int, 2] = scale_box_w / 2
                xywh_target[batch_id, cty_int, ctx_int, 3] = scale_box_h / 2
                xywh_target_weight[batch_id, cty_int, ctx_int] = 1
                xywh_l1target_weight[batch_id, cty_int, ctx_int, 0:2] = 1.0
                xywh_l1target_weight[batch_id, cty_int, ctx_int, 2:4] = 0.2

        avg_factor = max(1, int(center_heatmap_target.eq(1).sum().item()))
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            xywh_target=xywh_target,
            xywh_target_weight=xywh_target_weight,
            xywh_l1target_weight=xywh_l1target_weight,
        )
        return target_result, avg_factor

    def predict_by_feat(self,
                        center_local_preds: List[Tensor],
                        xywh_preds_coarse: List[Tensor],
                        xywh_preds_refine: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> InstanceList:
        assert len(center_local_preds) == len(xywh_preds_refine) == 1
        cfg = self.test_cfg if cfg is None else cfg

        center_local_pred = _gaussian_blur2d(center_local_preds[0], 3, 1.0)
        result_list = []
        for img_id in range(len(batch_img_metas)):
            result_list.append(
                self._predict_by_feat_single(
                    center_local_pred[img_id:img_id + 1],
                    xywh_preds_refine[0][img_id:img_id + 1],
                    batch_img_metas[img_id],
                    cfg=cfg,
                    rescale=rescale,
                    with_nms=with_nms,
                ))
        return result_list

    def _predict_by_feat_single(self,
                                center_local_pred: Tensor,
                                xywh_pred: Tensor,
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = True,
                                with_nms: bool = True) -> InstanceData:
        batch_det_bboxes, batch_labels = self._decode_heatmap(
            center_local_pred,
            xywh_pred,
            img_meta.get('batch_input_shape', img_meta['img_shape']),
            k=cfg.get('topk', 100),
            kernel=cfg.get('local_maximum_kernel', 3),
        )

        det_bboxes = batch_det_bboxes.view(-1, 5)
        det_labels = batch_labels.view(-1)

        score_thr = float(cfg.get('score_thr', 0.0))
        if score_thr > 0:
            keep = det_bboxes[:, 4] >= score_thr
            det_bboxes = det_bboxes[keep]
            det_labels = det_labels[keep]

        border = img_meta.get('border', [0, 0, 0, 0])
        det_bboxes[..., :4] -= det_bboxes.new_tensor(border)[[2, 0, 2, 0]]

        if rescale and 'scale_factor' in img_meta:
            scale_factor = det_bboxes.new_tensor(img_meta['scale_factor'])
            if scale_factor.numel() == 4:
                det_bboxes[..., :4] /= scale_factor
            else:
                det_bboxes[..., :4] /= scale_factor.repeat(2)

        if with_nms:
            det_bboxes, det_labels = self._bboxes_nms(det_bboxes, det_labels,
                                                      cfg)

        results = InstanceData()
        results.bboxes = det_bboxes[..., :4]
        results.scores = det_bboxes[..., 4]
        results.labels = det_labels
        return results

    def _decode_heatmap(self, center_heatmap_pred: Tensor, xywh_pred: Tensor,
                        img_shape: Tuple[int, ...], k: int = 100,
                        kernel: int = 3) -> Tuple[Tensor, Tensor]:
        coarse_h, coarse_w = xywh_pred.shape[2:]
        inp_h, inp_w = img_shape[:2]

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)
        batch_scores, batch_index, batch_topk_labels, topk_ys, topk_xs = (
            self._get_topk_from_local_heatmap(center_heatmap_pred, k=k))

        xywh = transpose_and_gather_feat(xywh_pred, batch_index)
        topk_xs = topk_xs + xywh[..., 0]
        topk_ys = topk_ys + xywh[..., 1]
        tl_x = (topk_xs - xywh[..., 2]) * (inp_w / coarse_w)
        tl_y = (topk_ys - xywh[..., 3]) * (inp_h / coarse_h)
        br_x = (topk_xs + xywh[..., 2]) * (inp_w / coarse_w)
        br_y = (topk_ys + xywh[..., 3]) * (inp_h / coarse_h)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        return batch_bboxes, batch_topk_labels

    def _get_topk_from_local_heatmap(self, center_heatmap: Tensor,
                                     k: int = 20) -> Tuple[Tensor, ...]:
        batch, _, height, width = center_heatmap.size()
        k = min(k, height * width * self.num_classes)
        topk_scores, topk_inds = torch.topk(center_heatmap.view(batch, -1), k)
        topk_clses = topk_inds // (height * width)
        topk_inds = topk_inds % (height * width)

        local_topk_ys = topk_inds // width
        local_topk_xs = topk_inds % width
        topk_ys = torch.div(local_topk_ys, 4, rounding_mode='floor').float()
        topk_xs = torch.div(local_topk_xs, 4, rounding_mode='floor').float()

        coarse_width = width // 4
        topk_inds = (coarse_width * topk_ys + topk_xs).long()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

    def _decode_bbox_map(self, xywh_pred: Tensor,
                         center_points: Tensor) -> Tensor:
        batch_size = xywh_pred.size(0)
        bbox_pred = xywh_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        bbox_pred = bbox_pred.contiguous()
        bbox_pred[:, :, 0:2] = bbox_pred[:, :, 0:2] + center_points.unsqueeze(0)
        return bbox_pred

    def _bboxes_nms(self, bboxes: Tensor, labels: Tensor,
                    cfg: ConfigDict) -> Tuple[Tensor, Tensor]:
        if labels.numel() == 0:
            return bboxes, labels

        nms_cfg = cfg.get('nms', cfg.get('nms_cfg', None))
        max_num = int(cfg.get('max_per_img', len(bboxes)))
        if nms_cfg is None:
            if max_num > 0:
                order = torch.argsort(bboxes[:, -1], descending=True)[:max_num]
                return bboxes[order], labels[order]
            return bboxes, labels

        bboxes, keep = batched_nms(
            bboxes[:, :4],
            bboxes[:, -1].contiguous(),
            labels,
            nms_cfg,
        )
        if max_num > 0:
            bboxes = bboxes[:max_num]
            labels = labels[keep][:max_num]
        else:
            labels = labels[keep]
        return bboxes, labels
