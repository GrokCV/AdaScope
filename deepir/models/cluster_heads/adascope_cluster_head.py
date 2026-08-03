import torch
import torch.nn as nn
from typing import List, Tuple

from mmengine.structures import InstanceData
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.registry import MODELS


@MODELS.register_module()
class C5ClusterHead(BaseDenseHead):
    """C5 binary grid classifier head with grid-to-bbox decoding helpers."""

    def __init__(self,
                 in_channels,
                 feat_channels=256,
                 num_convs=4,
                 core_stride=2,
                 core_size=512,
                 loss_cls: dict = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 threshold=0.5,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     std=0.01,
                     override=[dict(name='cls_pred')])):
        super().__init__(init_cfg=init_cfg)

        self.loss_cls = MODELS.build(loss_cls)
        self.threshold = float(threshold)
        self.core_stride = core_stride
        self.core_size = core_size

        cls_subnet = [
            nn.Conv2d(in_channels, feat_channels, 3, stride=self.core_stride, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_convs - 1):
            cls_subnet.extend([
                nn.Conv2d(feat_channels, feat_channels, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ])
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.cls_pred = nn.Conv2d(feat_channels, 1, 1)

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        c5_feat = feats[0]
        cls_feat = self.cls_subnet(c5_feat)
        return self.cls_pred(cls_feat)

    def loss(self, x: Tuple[torch.Tensor], batch_data_samples: List[InstanceData]) -> dict:
        cls_score = self(x)
        return self.loss_by_feat(cls_score, batch_data_samples)

    def loss_by_feat(self, cls_score: torch.Tensor, batch_data_samples: List[InstanceData]) -> dict:
        gt_cls_maps = torch.stack([
            torch.as_tensor(ds.gt_cls_map, device=cls_score.device, dtype=torch.float32)
            for ds in batch_data_samples
        ]).unsqueeze(1)

        loss_cls = self.loss_cls(cls_score, gt_cls_maps)
        return dict(loss_cls=loss_cls)

    def _build_mask(self,
                    prob_map: torch.Tensor,
                    data_sample: InstanceData,
                    use_gt_mask: bool) -> torch.Tensor:
        if not use_gt_mask:
            return prob_map > self.threshold

        if not hasattr(data_sample, 'gt_cls_map'):
            raise KeyError('gt_cls_map is required when use_gt_mask=True')

        gt_map = torch.as_tensor(data_sample.gt_cls_map, device=prob_map.device, dtype=prob_map.dtype)
        if gt_map.shape != prob_map.shape:
            gt_map = torch.nn.functional.interpolate(
                gt_map.unsqueeze(0).unsqueeze(0), size=prob_map.shape, mode='nearest'
            ).squeeze(0).squeeze(0)
        return gt_map > 0

    def _decode_grid(self,
                     prob_map: torch.Tensor,
                     mask: torch.Tensor,
                     img_h: int,
                     img_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device = prob_map.device
        foreground_y, foreground_x = torch.where(mask)

        if foreground_y.numel() == 0:
            return torch.zeros((0, 4), device=device), torch.zeros((0,), device=device)

        feat_h, feat_w = prob_map.shape
        stride_h = float(img_h) / float(feat_h)
        stride_w = float(img_w) / float(feat_w)

        scores = prob_map[foreground_y, foreground_x]

        x1 = foreground_x.float() * stride_w
        y1 = foreground_y.float() * stride_h
        x2 = x1 + stride_w
        y2 = y1 + stride_h

        bboxes = torch.stack([x1, y1, x2, y2], dim=1)
        bboxes[:, 0::2] = bboxes[:, 0::2].clamp(0, float(img_w))
        bboxes[:, 1::2] = bboxes[:, 1::2].clamp(0, float(img_h))

        valid = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
        return bboxes[valid], scores[valid]

    def get_grid_bboxes_by_feat(self,
                                cls_score: torch.Tensor,
                                batch_data_samples: List[InstanceData],
                                use_gt_mask: bool = False) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        cls_prob = cls_score.sigmoid()

        grid_bboxes: List[torch.Tensor] = []
        grid_scores: List[torch.Tensor] = []

        for i, data_sample in enumerate(batch_data_samples):
            prob_map = cls_prob[i, 0]
            mask = self._build_mask(prob_map, data_sample, use_gt_mask=use_gt_mask)

            if hasattr(data_sample, 'img_shape'):
                img_h, img_w = data_sample.img_shape[:2]
            else:
                img_h = int(self.core_size)
                img_w = int(self.core_size)

            bboxes, scores = self._decode_grid(prob_map, mask, img_h=img_h, img_w=img_w)
            grid_bboxes.append(bboxes)
            grid_scores.append(scores)

        return grid_bboxes, grid_scores, cls_prob

    def get_grid_bboxes(self,
                        feats: Tuple[torch.Tensor],
                        batch_data_samples: List[InstanceData],
                        use_gt_mask: bool = False) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        cls_score = self(feats)
        return self.get_grid_bboxes_by_feat(
            cls_score, batch_data_samples, use_gt_mask=use_gt_mask)

    def predict_by_feat(self,
                        cls_score: torch.Tensor,
                        batch_data_samples: List[InstanceData],
                        rescale: bool = True) -> List[InstanceData]:
        grid_bboxes, grid_scores, cls_prob = self.get_grid_bboxes_by_feat(
            cls_score, batch_data_samples, use_gt_mask=False)

        results = []
        for i, data_sample in enumerate(batch_data_samples):
            bboxes = grid_bboxes[i]
            scores = grid_scores[i]

            if rescale and hasattr(data_sample, 'ori_shape') and bboxes.numel() > 0:
                ori_h, ori_w = data_sample.ori_shape[:2]
                img_h, img_w = data_sample.img_shape[:2]

                w_scale = float(ori_w) / float(img_w)
                h_scale = float(ori_h) / float(img_h)
                bboxes = bboxes.clone()
                bboxes[:, 0] *= w_scale
                bboxes[:, 2] *= w_scale
                bboxes[:, 1] *= h_scale
                bboxes[:, 3] *= h_scale

            pred_instances = InstanceData()
            pred_instances.bboxes = bboxes
            pred_instances.scores = scores
            pred_instances.labels = torch.zeros(len(scores), dtype=torch.long, device=bboxes.device)

            data_sample.pred_cls_heatmap = cls_prob[i]
            results.append(pred_instances)

        return results

    def predict(self,
                feats: Tuple[torch.Tensor],
                batch_data_samples: List[InstanceData],
                rescale: bool = True) -> List[InstanceData]:
        cls_score = self(feats)
        return self.predict_by_feat(cls_score, batch_data_samples, rescale=rescale)
