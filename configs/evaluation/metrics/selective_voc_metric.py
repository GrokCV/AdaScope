import copy
from typing import Optional

import numpy as np

from mmdet.evaluation.metrics import VOCMetric
from mmdet.registry import METRICS


@METRICS.register_module()
class SelectiveVOCMetric(VOCMetric):
    """VOC metric that can target a chosen prediction field and GT source."""

    default_prefix: Optional[str] = 'selective_voc'

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

    def _build_ann(self, data_sample) -> dict:
        if self.gt_source == 'instance':
            gt = copy.deepcopy(data_sample)
            # data_sample is already a dict (converted by evaluator)
            gt_instances = gt['gt_instances']
            gt_ignore_instances = gt['ignored_instances']
            return dict(
                labels=gt_instances['labels'].cpu().numpy(),
                bboxes=gt_instances['bboxes'].cpu().numpy(),
                bboxes_ignore=gt_ignore_instances['bboxes'].cpu().numpy(),
                labels_ignore=gt_ignore_instances['labels'].cpu().numpy(),
            )

        cluster_boxes = self._get_value(data_sample, 'gt_cluster_bboxes')
        cluster_boxes = self._to_numpy(cluster_boxes, dtype=np.float32)
        if cluster_boxes.size == 0:
            cluster_boxes = np.zeros((0, 4), dtype=np.float32)
        elif cluster_boxes.ndim == 1:
            cluster_boxes = cluster_boxes.reshape(1, -1)
        cluster_boxes = cluster_boxes[:, :4]
        return dict(
            labels=np.zeros((cluster_boxes.shape[0],), dtype=np.int64),
            bboxes=cluster_boxes,
            bboxes_ignore=np.zeros((0, 4), dtype=np.float32),
            labels_ignore=np.zeros((0,), dtype=np.int64),
        )

    def process(self, data_batch: dict, data_samples) -> None:
        for data_sample in data_samples:
            ann = self._build_ann(data_sample)

            pred = self._get_value(data_sample, self.pred_key)
            if pred is None:
                raise KeyError(
                    f'Prediction field {self.pred_key!r} not found in data sample.')

            pred_bboxes = self._to_numpy(pred['bboxes'], dtype=np.float32)
            pred_scores = self._to_numpy(pred['scores'], dtype=np.float32)
            pred_labels = self._to_numpy(pred['labels'], dtype=np.int64)

            dets = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(pred_labels == label)[0]
                pred_bbox_scores = np.hstack(
                    [pred_bboxes[index], pred_scores[index].reshape((-1, 1))])
                dets.append(pred_bbox_scores)

            self.results.append((ann, dets))
