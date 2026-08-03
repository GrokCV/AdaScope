import numpy as np
import torch
from typing import List, Optional, Sequence, Union

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet.evaluation.functional import eval_map
from mmdet.registry import METRICS

try:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@METRICS.register_module()
class GridAndDetMetric(BaseMetric):
    """Evaluate grid classification and VOC-style detection mAP together."""

    default_prefix: Optional[str] = 'grid_det'

    def __init__(self,
                 cls_threshold: float = 0.5,
                 iou_thrs: Union[float, List[float]] = 0.5,
                 scale_ranges: Optional[List[tuple]] = None,
                 metric: Union[str, List[str]] = 'mAP',
                 eval_mode: str = '11points',
                 det_gt_source: str = 'instance',
                 align_to: str = 'img',
                 pred_box_space: str = 'auto',
                 gt_box_space: str = 'auto',
                 debug_max_samples: int = 5,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.cls_threshold = float(cls_threshold)
        if isinstance(iou_thrs, (float, int)):
            self.iou_thrs = [float(iou_thrs)]
        else:
            self.iou_thrs = [float(v) for v in iou_thrs]
        self.scale_ranges = scale_ranges

        if isinstance(metric, list):
            if len(metric) > 1:
                import warnings
                warnings.warn(
                    f'GridAndDetMetric only supports one metric at a time. '
                    f'Got {metric}, using "{metric[0]}" and ignoring the rest.',
                    UserWarning)
            metric = metric[0]
        self.metric = metric
        self.eval_mode = eval_mode

        self.det_gt_source = str(det_gt_source)
        if self.det_gt_source not in ('instance', 'cluster'):
            raise ValueError(
                f'det_gt_source must be \"instance\" or \"cluster\", got {self.det_gt_source!r}')
        self.align_to = str(align_to)
        self.pred_box_space = str(pred_box_space)
        self.gt_box_space = str(gt_box_space)
        self.debug_max_samples = max(int(debug_max_samples), 0)

    @staticmethod
    def _to_numpy(arr, dtype=None):
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        else:
            arr = np.asarray(arr)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @staticmethod
    def _get_meta_value(data_sample, key: str):
        if isinstance(data_sample, dict):
            return data_sample.get(key, None)
        if hasattr(data_sample, 'metainfo') and key in data_sample.metainfo:
            return data_sample.metainfo[key]
        if hasattr(data_sample, key):
            return getattr(data_sample, key)
        try:
            return data_sample[key]
        except Exception:
            return None

    @staticmethod
    def _infer_box_space(
        boxes: np.ndarray,
        img_h: float,
        img_w: float,
        ori_h: float,
        ori_w: float,
        prefer: str,
    ) -> str:
        if boxes.size == 0:
            return prefer

        max_x = float(np.max(boxes[:, 2]))
        max_y = float(np.max(boxes[:, 3]))
        tol = 1.0

        in_img = (max_x <= img_w + tol) and (max_y <= img_h + tol)
        in_ori = (max_x <= ori_w + tol) and (max_y <= ori_h + tol)

        if in_img and not in_ori:
            return 'img'
        if in_ori and not in_img:
            return 'ori'
        if in_img and in_ori:
            return prefer
        return prefer

    def _align_boxes_to_target(self, boxes: np.ndarray, data_sample, box_role: str) -> np.ndarray:
        boxes = self._to_numpy(boxes, dtype=np.float32)
        if boxes.size == 0 or self.align_to == 'img':
            return boxes

        if self.align_to != 'ori':
            raise ValueError(f'Unsupported align_to={self.align_to!r}, expected "img" or "ori".')

        img_shape = self._get_meta_value(data_sample, 'img_shape')
        ori_shape = self._get_meta_value(data_sample, 'ori_shape')
        if img_shape is None or ori_shape is None:
            return boxes

        img_h, img_w = float(img_shape[0]), float(img_shape[1])
        ori_h, ori_w = float(ori_shape[0]), float(ori_shape[1])
        if img_h <= 0 or img_w <= 0:
            return boxes
        if abs(img_h - ori_h) < 1e-6 and abs(img_w - ori_w) < 1e-6:
            return boxes

        mode = self.gt_box_space if box_role == 'gt' else self.pred_box_space
        prefer = 'img' if box_role == 'gt' else 'ori'
        if mode == 'auto':
            mode = self._infer_box_space(boxes, img_h, img_w, ori_h, ori_w, prefer=prefer)

        if mode != 'img':
            return boxes

        out = boxes.copy()
        sx = ori_w / img_w
        sy = ori_h / img_h
        out[:, 0::2] *= sx
        out[:, 1::2] *= sy
        return out

    @staticmethod
    def _binary_confusion_counts(gt_binary: np.ndarray, pred_binary: np.ndarray):
        tp = int(np.logical_and(gt_binary == 1, pred_binary == 1).sum())
        tn = int(np.logical_and(gt_binary == 0, pred_binary == 0).sum())
        fp = int(np.logical_and(gt_binary == 0, pred_binary == 1).sum())
        fn = int(np.logical_and(gt_binary == 1, pred_binary == 0).sum())
        return tn, fp, fn, tp

    def _extract_cluster_gt(self, data_sample) -> np.ndarray:
        cluster_boxes = None
        if isinstance(data_sample, dict):
            cluster_boxes = data_sample.get('gt_cluster_bboxes', None)
        if cluster_boxes is None and hasattr(data_sample, 'metainfo'):
            cluster_boxes = data_sample.metainfo.get('gt_cluster_bboxes', None)
        if cluster_boxes is None:
            cluster_boxes = self._get_meta_value(data_sample, 'gt_cluster_bboxes')
        if cluster_boxes is None:
            return np.zeros((0, 4), dtype=np.float32)
        boxes = self._to_numpy(cluster_boxes, dtype=np.float32)
        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        return boxes[:, :4]

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            result = {}

            if self.det_gt_source == 'cluster':
                gt_bboxes = self._extract_cluster_gt(data_sample)
                gt_labels = np.zeros((gt_bboxes.shape[0],), dtype=np.int64)
                ignore_bboxes = np.zeros((0, 4), dtype=np.float32)
                ignore_labels = np.zeros((0,), dtype=np.int64)
            else:
                gt_instances = data_sample['gt_instances']
                gt_bboxes = self._to_numpy(gt_instances['bboxes'], dtype=np.float32)
                gt_labels = self._to_numpy(gt_instances['labels'], dtype=np.int64)

                if 'ignored_instances' in data_sample and data_sample['ignored_instances'] is not None:
                    ignored = data_sample['ignored_instances']
                    ignore_bboxes = self._to_numpy(ignored['bboxes'], dtype=np.float32)
                    ignore_labels = self._to_numpy(ignored['labels'], dtype=np.int64)
                else:
                    ignore_bboxes = np.zeros((0, 4), dtype=np.float32)
                    ignore_labels = np.zeros((0,), dtype=np.int64)

            gt_bboxes = self._align_boxes_to_target(gt_bboxes, data_sample, box_role='gt')
            ignore_bboxes = self._align_boxes_to_target(ignore_bboxes, data_sample, box_role='gt')

            ann = dict(
                labels=gt_labels,
                bboxes=gt_bboxes,
                bboxes_ignore=ignore_bboxes,
                labels_ignore=ignore_labels,
            )

            pred_instances = data_sample['pred_instances']
            pred_bboxes = self._to_numpy(pred_instances['bboxes'], dtype=np.float32)
            pred_scores = self._to_numpy(pred_instances['scores'], dtype=np.float32)
            pred_labels = self._to_numpy(pred_instances['labels'], dtype=np.int64)
            pred_bboxes = self._align_boxes_to_target(pred_bboxes, data_sample, box_role='pred')

            if self.dataset_meta and 'classes' in self.dataset_meta:
                num_classes = len(self.dataset_meta['classes'])
            else:
                num_classes = 1

            dets = []
            for label in range(num_classes):
                index = np.where(pred_labels == label)[0]
                if len(index) > 0:
                    pred_bbox_scores = np.hstack([
                        pred_bboxes[index].astype(np.float32),
                        pred_scores[index].reshape((-1, 1)).astype(np.float32),
                    ])
                else:
                    pred_bbox_scores = np.zeros((0, 5), dtype=np.float32)
                dets.append(pred_bbox_scores)

            result['ann'] = ann
            result['dets'] = dets

            pred_cls_grid = None
            gt_cls_grid = None
            if 'pred_cls_heatmap' in data_sample:
                pred_cls_heatmap = data_sample['pred_cls_heatmap']
                pred_cls_grid = self._to_numpy(pred_cls_heatmap).squeeze()
            if 'gt_cls_map' in data_sample:
                gt_cls_map = data_sample['gt_cls_map']
                gt_cls_grid = self._to_numpy(gt_cls_map).squeeze()

            result['pred_cls'] = pred_cls_grid
            result['gt_cls'] = gt_cls_grid

            result['debug'] = {
                'num_gt': len(gt_bboxes),
                'num_pred': len(pred_bboxes),
                'gt_bboxes': gt_bboxes[:5] if len(gt_bboxes) > 0 else np.zeros((0, 4)),
                'pred_bboxes': pred_bboxes[:5] if len(pred_bboxes) > 0 else np.zeros((0, 4)),
                'pred_scores': pred_scores[:5] if len(pred_scores) > 0 else np.zeros((0,)),
            }
            self.results.append(result)

    def compute_metrics(self, results: list) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()
        final_metrics = {}

        logger.info('\n' + '=' * 70)
        logger.info('[DEBUG] ========== COORDINATE SCALE CHECK ==========')

        total_gt = sum(r['debug']['num_gt'] for r in results)
        total_pred = sum(r['debug']['num_pred'] for r in results)
        logger.info(f'Total images: {len(results)}')
        logger.info(f'Total GT boxes: {total_gt}')
        logger.info(f'Total Pred boxes: {total_pred}')

        for i, res in enumerate(results[:self.debug_max_samples]):
            gt_bboxes = res['debug']['gt_bboxes']
            pred_bboxes = res['debug']['pred_bboxes']
            pred_scores = res['debug']['pred_scores']

            logger.info(f'\n--- Sample {i} ---')
            logger.info(f"  Num GT: {res['debug']['num_gt']}, Num Pred: {res['debug']['num_pred']}")

            if len(gt_bboxes) > 0:
                gt_all = res['ann']['bboxes']
                logger.info(
                    f'  GT range:   x=[{gt_all[:, 0].min():.1f}, {gt_all[:, 2].max():.1f}], '
                    f'y=[{gt_all[:, 1].min():.1f}, {gt_all[:, 3].max():.1f}]')
                logger.info(f'  GT example: {gt_bboxes[0].tolist()}')

            if len(pred_bboxes) > 0:
                pred_all = np.vstack([d[:, :4] for d in res['dets'] if len(d) > 0])
                logger.info(
                    f'  Pred range: x=[{pred_all[:, 0].min():.1f}, {pred_all[:, 2].max():.1f}], '
                    f'y=[{pred_all[:, 1].min():.1f}, {pred_all[:, 3].max():.1f}]')
                logger.info(f'  Pred example: {pred_bboxes[0].tolist()}')
                logger.info(f'  Pred scores: {pred_scores.tolist()}')

        iou_stats = self._compute_iou_statistics(results[:100])
        logger.info('\n[DEBUG] IoU Statistics (first 100 samples):')
        logger.info(f"  Mean max IoU per GT: {iou_stats['mean_max_iou']:.4f}")
        logger.info(f"  Matched ratio (IoU>0.1): {iou_stats['matched_01']:.2%}")
        logger.info(f"  Matched ratio (IoU>0.3): {iou_stats['matched_03']:.2%}")
        logger.info(f"  Matched ratio (IoU>0.5): {iou_stats['matched_05']:.2%}")
        logger.info('=' * 70 + '\n')

        valid_cls_pairs = [
            (r['pred_cls'], r['gt_cls']) for r in results
            if r['pred_cls'] is not None and r['gt_cls'] is not None
        ]
        pred_cls_list = [v[0] for v in valid_cls_pairs]
        gt_cls_list = [v[1] for v in valid_cls_pairs]

        if pred_cls_list and gt_cls_list:
            try:
                pred_sizes = [p.flatten().shape[0] for p in pred_cls_list]
                gt_sizes = [g.flatten().shape[0] for g in gt_cls_list]
                if pred_sizes != gt_sizes:
                    raise ValueError(
                        f'Grid size mismatch between pred and gt across samples. '
                        f'pred sizes: {pred_sizes[:5]}..., gt sizes: {gt_sizes[:5]}...')

                all_pred_flat = np.concatenate([p.flatten() for p in pred_cls_list])
                all_gt_flat = np.concatenate([g.flatten() for g in gt_cls_list])
                pred_binary = (all_pred_flat > self.cls_threshold).astype(np.int32)
                gt_binary = (all_gt_flat > self.cls_threshold).astype(np.int32)

                if HAS_SKLEARN:
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        gt_binary, pred_binary, average='binary', zero_division=0)
                    try:
                        tn, fp, fn, tp = confusion_matrix(
                            gt_binary, pred_binary, labels=[0, 1]).ravel()
                    except ValueError as e:
                        logger.warning(
                            f'confusion_matrix ravel() failed: {e}. '
                            f'gt unique={np.unique(gt_binary).tolist()}, '
                            f'pred unique={np.unique(pred_binary).tolist()}. '
                            f'Confusion matrix counts will be reported as 0.')
                        tn, fp, fn, tp = 0, 0, 0, 0
                else:
                    tn, fp, fn, tp = self._binary_confusion_counts(gt_binary, pred_binary)
                    precision = tp / (tp + fp + 1e-12)
                    recall = tp / (tp + fn + 1e-12)
                    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)

                final_metrics.update({
                    'grid_precision': round(float(precision), 4),
                    'grid_recall': round(float(recall), 4),
                    'grid_f1': round(float(f1), 4),
                    'grid_TP': int(tp),
                    'grid_FP': int(fp),
                    'grid_FN': int(fn),
                })
            except Exception as e:
                logger.warning(f'Grid metric calculation failed: {e}')

        if self.metric == 'mAP':
            voc_anns = [res['ann'] for res in results]
            voc_dets = [res['dets'] for res in results]

            if self.dataset_meta and 'classes' in self.dataset_meta:
                dataset_name = self.dataset_meta['classes']
            else:
                dataset_name = None

            mean_aps = []
            for iou_thr in self.iou_thrs:
                logger.info(f'\n{"-" * 20} iou_thr: {iou_thr} {"-" * 20}')
                try:
                    mean_ap, _ = eval_map(
                        voc_dets,
                        voc_anns,
                        scale_ranges=self.scale_ranges,
                        iou_thr=iou_thr,
                        dataset=dataset_name,
                        logger=logger,
                        eval_mode=self.eval_mode,
                        use_legacy_coordinate=False)
                except Exception as e:
                    logger.warning(f'eval_map failed: {e}')
                    mean_ap = 0.0

                mean_aps.append(mean_ap)
                final_metrics[f'AP{int(iou_thr * 100):02d}'] = round(float(mean_ap), 4)

            final_metrics['mAP'] = round(float(sum(mean_aps) / len(mean_aps)), 4)

        return final_metrics

    def _compute_iou_statistics(self, results: list) -> dict:
        all_max_ious = []
        total_gt = 0
        matched_01 = 0
        matched_03 = 0
        matched_05 = 0

        for res in results:
            gt_bboxes = res['ann']['bboxes']
            pred_list = [d[:, :4] for d in res['dets'] if len(d) > 0]

            if len(gt_bboxes) == 0:
                continue
            if not pred_list:
                total_gt += len(gt_bboxes)
                continue

            pred_bboxes = np.vstack(pred_list)
            if len(pred_bboxes) == 0:
                total_gt += len(gt_bboxes)
                continue

            ious = self._compute_iou_matrix(pred_bboxes, gt_bboxes)
            if ious.size > 0:
                max_ious_per_gt = ious.max(axis=0)
                all_max_ious.extend(max_ious_per_gt.tolist())
                total_gt += len(gt_bboxes)
                matched_01 += int((max_ious_per_gt > 0.1).sum())
                matched_03 += int((max_ious_per_gt > 0.3).sum())
                matched_05 += int((max_ious_per_gt > 0.5).sum())

        return {
            'mean_max_iou': float(np.mean(all_max_ious)) if all_max_ious else 0.0,
            'matched_01': matched_01 / total_gt if total_gt > 0 else 0.0,
            'matched_03': matched_03 / total_gt if total_gt > 0 else 0.0,
            'matched_05': matched_05 / total_gt if total_gt > 0 else 0.0,
        }

    def _compute_iou_matrix(self, bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
        if len(bboxes1) == 0 or len(bboxes2) == 0:
            return np.zeros((len(bboxes1), len(bboxes2)), dtype=np.float32)

        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)

        xi1 = np.maximum(x11, x21.T)
        yi1 = np.maximum(y11, y21.T)
        xi2 = np.minimum(x12, x22.T)
        yi2 = np.minimum(y12, y22.T)

        inter_w = np.maximum(0, xi2 - xi1)
        inter_h = np.maximum(0, yi2 - yi1)
        inter_area = inter_w * inter_h

        area1 = (x12 - x11) * (y12 - y11)
        area2 = (x22 - x21) * (y22 - y21)
        union_area = area1 + area2.T - inter_area

        iou = inter_area / np.maximum(union_area, 1e-6)
        return iou.astype(np.float32)
