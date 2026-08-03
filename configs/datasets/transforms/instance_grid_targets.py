import json
import os.path as osp
from typing import Dict

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class GenerateC5InstanceGridTargets(BaseTransform):
    """Generate C5 binary grid targets from instance boxes.

    Grid labeling rule:
    - A grid cell is positive if any instance box overlaps that cell.
    - Otherwise it is negative.

    Cluster boxes are optional and only kept as metadata for downstream
    analysis/evaluation. They do not decide the binary classification labels.
    """

    def __init__(
        self,
        stride: int = 32,
        cluster_json: str = '',
        missing_policy: str = 'empty',
    ) -> None:
        self.stride = int(stride)
        self.cluster_json = str(cluster_json)
        self.missing_policy = str(missing_policy)
        if self.missing_policy not in ('empty', 'error'):
            raise ValueError("missing_policy must be 'empty' or 'error'.")
        self._cluster_db = self._load_cluster_json(self.cluster_json) if self.cluster_json else {}

    def _load_cluster_json(self, path: str) -> Dict[str, np.ndarray]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        db: Dict[str, np.ndarray] = {}
        for item in data.get('images', []):
            boxes = np.asarray(item.get('cluster_bboxes', []), dtype=np.float32)
            if boxes.size == 0:
                boxes = np.zeros((0, 4), dtype=np.float32)
            elif boxes.ndim == 1:
                boxes = boxes.reshape(1, 4)
            key = str(item.get('id'))
            db[key] = boxes
            file_name = item.get('file_name', None)
            if file_name:
                db[file_name] = boxes
                db[osp.basename(file_name)] = boxes
        return db

    @staticmethod
    def _safe_shape(results, key, fallback):
        value = results.get(key, None)
        return fallback if value is None else value

    def _lookup_cluster_boxes(self, results: dict) -> np.ndarray:
        if not self._cluster_db:
            return np.zeros((0, 4), dtype=np.float32)

        keys = []
        if 'img_id' in results:
            keys.append(str(results['img_id']))
        if 'img_path' in results:
            keys.append(results['img_path'])
            keys.append(osp.basename(results['img_path']))
        if 'file_name' in results:
            keys.append(results['file_name'])
            keys.append(osp.basename(results['file_name']))

        for key in keys:
            if key in self._cluster_db:
                return self._cluster_db[key].copy()

        if self.missing_policy == 'error':
            raise KeyError(f'No cluster GT found for keys={keys} in {self.cluster_json}')
        return np.zeros((0, 4), dtype=np.float32)

    def _rescale_boxes_to_current_img(self, boxes: np.ndarray, results: dict) -> np.ndarray:
        if boxes.shape[0] == 0:
            return boxes

        img_h, img_w = self._safe_shape(results, 'img_shape', (0, 0))[:2]
        ori_h, ori_w = self._safe_shape(results, 'ori_shape', (img_h, img_w))[:2]
        if ori_h <= 0 or ori_w <= 0 or img_h <= 0 or img_w <= 0:
            return boxes

        scale_factor = results.get('scale_factor', None)
        if scale_factor is not None:
            scale_factor = np.asarray(scale_factor, dtype=np.float32).reshape(-1)
            if scale_factor.size >= 2:
                w_scale = float(scale_factor[0])
                h_scale = float(scale_factor[1])
            else:
                w_scale = float(scale_factor[0])
                h_scale = float(scale_factor[0])
        else:
            w_scale = float(img_w) / float(ori_w)
            h_scale = float(img_h) / float(ori_h)

        scaled = boxes.copy()
        scaled[:, 0] *= w_scale
        scaled[:, 2] *= w_scale
        scaled[:, 1] *= h_scale
        scaled[:, 3] *= h_scale
        scaled[:, 0::2] = np.clip(scaled[:, 0::2], 0.0, float(img_w))
        scaled[:, 1::2] = np.clip(scaled[:, 1::2], 0.0, float(img_h))
        valid = (scaled[:, 2] > scaled[:, 0]) & (scaled[:, 3] > scaled[:, 1])
        return scaled[valid]

    @staticmethod
    def _to_numpy_boxes(gt_bboxes) -> np.ndarray:
        if gt_bboxes is None:
            return np.zeros((0, 4), dtype=np.float32)
        if hasattr(gt_bboxes, 'tensor'):
            boxes = gt_bboxes.tensor.detach().cpu().numpy()
        else:
            boxes = np.asarray(gt_bboxes)
        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return boxes.astype(np.float32).reshape(-1, 4)

    def _extract_instance_boxes(self, results: dict) -> np.ndarray:
        boxes = self._to_numpy_boxes(results.get('gt_bboxes', None))
        if boxes.shape[0] == 0:
            return boxes

        ignore_flags = results.get('gt_ignore_flags', None)
        if ignore_flags is not None and len(ignore_flags) == len(boxes):
            keep = np.logical_not(np.asarray(ignore_flags, dtype=np.bool_))
            boxes = boxes[keep]

        img_h, img_w = results['img_shape'][:2]
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0.0, float(img_w))
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0.0, float(img_h))
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes[valid]

    def _fill_grid_from_instances(self, gt_cls_map: np.ndarray, instance_boxes: np.ndarray) -> None:
        if instance_boxes.shape[0] == 0:
            return

        grid_h, grid_w = gt_cls_map.shape
        for bbox in instance_boxes:
            gx1 = int(np.floor(float(bbox[0]) / float(self.stride)))
            gy1 = int(np.floor(float(bbox[1]) / float(self.stride)))
            gx2 = int(np.ceil(float(bbox[2]) / float(self.stride)) - 1)
            gy2 = int(np.ceil(float(bbox[3]) / float(self.stride)) - 1)

            gx1 = int(np.clip(gx1, 0, grid_w - 1))
            gy1 = int(np.clip(gy1, 0, grid_h - 1))
            gx2 = int(np.clip(gx2, 0, grid_w - 1))
            gy2 = int(np.clip(gy2, 0, grid_h - 1))
            if gx2 < gx1 or gy2 < gy1:
                continue
            gt_cls_map[gy1:gy2 + 1, gx1:gx2 + 1] = 1.0

    def transform(self, results: dict) -> dict:
        img_h, img_w = results['img_shape'][:2]
        grid_h = img_h // self.stride
        grid_w = img_w // self.stride

        gt_cls_map = np.zeros((grid_h, grid_w), dtype=np.float32)
        gt_offset_map = np.zeros((2, grid_h, grid_w), dtype=np.float32)
        gt_offset_weight = np.zeros((1, grid_h, grid_w), dtype=np.float32)
        gt_scale_map = np.zeros((1, grid_h, grid_w), dtype=np.float32)
        gt_scale_weight = np.zeros((1, grid_h, grid_w), dtype=np.float32)
        gt_rf_level = np.full((1, grid_h, grid_w), -1, dtype=np.int64)
        gt_rf_weight = np.zeros((1, grid_h, grid_w), dtype=np.float32)

        instance_boxes = self._extract_instance_boxes(results)
        self._fill_grid_from_instances(gt_cls_map, instance_boxes)

        cluster_boxes = self._lookup_cluster_boxes(results)
        cluster_boxes = self._rescale_boxes_to_current_img(cluster_boxes, results)

        results['gt_cls_map'] = gt_cls_map
        results['gt_offset_map'] = gt_offset_map
        results['gt_offset_weight'] = gt_offset_weight
        results['gt_scale_map'] = gt_scale_map
        results['gt_scale_weight'] = gt_scale_weight
        results['gt_rf_level'] = gt_rf_level
        results['gt_rf_weight'] = gt_rf_weight
        results['gt_cluster_bboxes'] = cluster_boxes.astype(np.float32)
        return results
