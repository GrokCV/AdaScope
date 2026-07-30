from typing import Optional

import numpy as np
import torch

from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results


@METRICS.register_module()
class SelectiveCocoMetric(CocoMetric):
    """COCO metric that can evaluate a chosen prediction branch."""

    default_prefix: Optional[str] = 'selective_coco'

    def __init__(
        self,
        pred_key: str = 'pred_instances',
        gt_source: str = 'instance',
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pred_key = str(pred_key)
        self.gt_source = str(gt_source)
        if self.gt_source not in ('instance', 'cluster'):
            raise ValueError(
                f'gt_source must be "instance" or "cluster", got {self.gt_source!r}.')

    @staticmethod
    def _get_value(data_sample, key: str):
        if isinstance(data_sample, dict) and key in data_sample:
            return data_sample[key]
        if hasattr(data_sample, key):
            return getattr(data_sample, key)
        if hasattr(data_sample, 'metainfo') and key in data_sample.metainfo:
            return data_sample.metainfo[key]
        return None

    @staticmethod
    def _to_numpy(value, dtype=None):
        if value is None:
            arr = np.empty((0,), dtype=np.float32 if dtype is None else dtype)
        elif hasattr(value, 'detach'):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def _build_coco_gt(self, data_sample) -> dict:
        ori_shape = self._get_value(data_sample, 'ori_shape')
        img_id = self._get_value(data_sample, 'img_id')
        if ori_shape is None:
            raise KeyError('ori_shape is required for COCO evaluation.')
        if img_id is None:
            raise KeyError('img_id is required for COCO evaluation.')

        gt = dict(width=int(ori_shape[1]), height=int(ori_shape[0]), img_id=img_id)
        if self._coco_api is not None:
            return gt

        anns = []
        if self.gt_source == 'instance':
            instances = self._get_value(data_sample, 'instances')
            if instances is not None:
                gt['anns'] = instances
                return gt

            gt_instances = self._get_value(data_sample, 'gt_instances')
            if gt_instances is not None:
                bboxes = self._to_numpy(gt_instances['bboxes'], dtype=np.float32)
                labels = self._to_numpy(gt_instances['labels'], dtype=np.int64)
                for bbox, label in zip(bboxes, labels):
                    anns.append(
                        dict(
                            bbox=bbox,
                            bbox_label=int(label),
                            ignore_flag=0,
                        ))

            ignored_instances = self._get_value(data_sample, 'ignored_instances')
            if ignored_instances is not None:
                bboxes = self._to_numpy(ignored_instances['bboxes'], dtype=np.float32)
                labels = self._to_numpy(ignored_instances['labels'], dtype=np.int64)
                for bbox, label in zip(bboxes, labels):
                    anns.append(
                        dict(
                            bbox=bbox,
                            bbox_label=int(label),
                            ignore_flag=1,
                        ))
        else:
            cluster_boxes = self._get_value(data_sample, 'gt_cluster_bboxes')
            cluster_boxes = self._to_numpy(cluster_boxes, dtype=np.float32)
            if cluster_boxes.ndim == 1 and cluster_boxes.size > 0:
                cluster_boxes = cluster_boxes.reshape(1, -1)
            for bbox in cluster_boxes[:, :4]:
                anns.append(
                    dict(
                        bbox=bbox,
                        bbox_label=0,
                        ignore_flag=0,
                    ))

        gt['anns'] = anns
        return gt

    def process(self, data_batch: dict, data_samples) -> None:
        for data_sample in data_samples:
            pred = self._get_value(data_sample, self.pred_key)
            if pred is None:
                raise KeyError(
                    f'Prediction field {self.pred_key!r} not found in data sample.')

            result = dict()
            result['img_id'] = self._get_value(data_sample, 'img_id')
            result['bboxes'] = self._to_numpy(pred['bboxes'], dtype=np.float32)
            result['scores'] = self._to_numpy(pred['scores'], dtype=np.float32)
            result['labels'] = self._to_numpy(pred['labels'], dtype=np.int64)

            if 'masks' in pred:
                masks = pred['masks']
                result['masks'] = encode_mask_results(
                    masks.detach().cpu().numpy()) if isinstance(
                        masks, torch.Tensor) else masks
            if 'mask_scores' in pred:
                result['mask_scores'] = self._to_numpy(
                    pred['mask_scores'], dtype=np.float32)

            gt = self._build_coco_gt(data_sample)
            self.results.append((gt, result))
