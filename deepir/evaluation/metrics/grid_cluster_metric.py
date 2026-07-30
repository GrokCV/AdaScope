from typing import Optional

import numpy as np

from mmengine.evaluator import BaseMetric

from mmdet.registry import METRICS


@METRICS.register_module()
class GridClusterMetric(BaseMetric):
    """Evaluate grid-cluster binary classification quality."""

    default_prefix: Optional[str] = 'grid_cluster'

    def __init__(
        self,
        cls_threshold: float = 0.3,
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.cls_threshold = float(cls_threshold)

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
    def _to_numpy(value):
        if value is None:
            return None
        if hasattr(value, 'detach'):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def process(self, data_batch: dict, data_samples) -> None:
        for data_sample in data_samples:
            pred = self._to_numpy(self._get_value(data_sample, 'pred_cls_heatmap'))
            gt = self._to_numpy(self._get_value(data_sample, 'gt_cls_map'))
            if pred is None or gt is None:
                continue

            pred = np.asarray(pred).squeeze()
            gt = np.asarray(gt).squeeze()
            self.results.append(
                dict(
                    pred=pred.astype(np.float32),
                    gt=gt.astype(np.float32),
                ))

    def compute_metrics(self, results: list) -> dict:
        if not results:
            return {}

        pred_list = []
        gt_list = []
        for item in results:
            pred = item['pred']
            gt = item['gt']
            if pred.shape != gt.shape:
                raise ValueError(
                    f'Grid shape mismatch: pred={pred.shape}, gt={gt.shape}.')
            pred_list.append(pred.reshape(-1))
            gt_list.append(gt.reshape(-1))

        all_pred = np.concatenate(pred_list, axis=0)
        all_gt = np.concatenate(gt_list, axis=0)

        pred_bin = (all_pred >= self.cls_threshold).astype(np.int32)
        gt_bin = (all_gt > 0.5).astype(np.int32)

        tp = int(np.logical_and(pred_bin == 1, gt_bin == 1).sum())
        fp = int(np.logical_and(pred_bin == 1, gt_bin == 0).sum())
        fn = int(np.logical_and(pred_bin == 0, gt_bin == 1).sum())

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-12)

        return dict(
            precision=round(float(precision), 4),
            recall=round(float(recall), 4),
            f1=round(float(f1), 4),
            tp=tp,
            fp=fp,
            fn=fn,
        )
