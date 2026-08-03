import json
import os.path as osp
from typing import Dict

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class GenerateC5TargetsFromClusterJSON(BaseTransform):
    """Generate C5 targets from precomputed cluster boxes stored in JSON.

    Expected JSON format:
        {
          "images": [
            {
              "id": 1,
              "file_name": "xxx.jpg",
              "cluster_bboxes": [[x1, y1, x2, y2], ...]
            },
            ...
          ]
        }
    """

    def __init__(
        self,
        cluster_json: str,
        stride: int = 32,
        rf_scale_bins=(3.0, 6.0),
        missing_policy: str = 'empty',
        scale_eps: float = 1e-6,
    ) -> None:
        self.cluster_json = str(cluster_json)
        self.stride = int(stride)
        self.rf_scale_bins = tuple(float(x) for x in rf_scale_bins)
        self.missing_policy = str(missing_policy)
        self.scale_eps = float(scale_eps)
        if self.missing_policy not in ('empty', 'error'):
            raise ValueError("missing_policy must be 'empty' or 'error'.")
        self._cluster_db = self._load_cluster_json(self.cluster_json)

    def _load_cluster_json(self, path: str) -> Dict[str, np.ndarray]:
        with open(path, 'r') as f:
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

    def _rf_level_from_scale_ratio(self, scale_ratio: float) -> int:
        b0, b1 = self.rf_scale_bins[:2]
        if scale_ratio < b0:
            return 0
        if scale_ratio < b1:
            return 1
        return 2

    def _lookup_cluster_boxes(self, results: dict) -> np.ndarray:
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

    @staticmethod
    def _safe_shape(results, key, fallback):
        value = results.get(key, None)
        return fallback if value is None else value

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
        scaled[:, 0::2] = np.clip(scaled[:, 0::2], 0, float(img_w))
        scaled[:, 1::2] = np.clip(scaled[:, 1::2], 0, float(img_h))
        valid = (scaled[:, 2] > scaled[:, 0]) & (scaled[:, 3] > scaled[:, 1])
        return scaled[valid]

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

        cluster_boxes = self._lookup_cluster_boxes(results)
        cluster_boxes = self._rescale_boxes_to_current_img(cluster_boxes, results)

        for bbox in cluster_boxes:
            x_center = 0.5 * (bbox[0] + bbox[2])
            y_center = 0.5 * (bbox[1] + bbox[3])
            gx_center = int(x_center / self.stride)
            gy_center = int(y_center / self.stride)

            if not (0 <= gx_center < grid_w and 0 <= gy_center < grid_h):
                continue

            gt_cls_map[gy_center, gx_center] = 1.0

            grid_cx = (gx_center + 0.5) * self.stride
            grid_cy = (gy_center + 0.5) * self.stride
            gt_offset_map[0, gy_center, gx_center] = (x_center - grid_cx) / self.stride
            gt_offset_map[1, gy_center, gx_center] = (y_center - grid_cy) / self.stride
            gt_offset_weight[0, gy_center, gx_center] = 1.0

            box_w = max(float(bbox[2] - bbox[0]), self.scale_eps)
            box_h = max(float(bbox[3] - bbox[1]), self.scale_eps)
            cluster_size = np.sqrt(box_w * box_h)
            gt_scale_map[0, gy_center, gx_center] = np.log(
                max(cluster_size / self.stride, self.scale_eps))
            gt_scale_weight[0, gy_center, gx_center] = 1.0

            gt_rf_level[0, gy_center, gx_center] = self._rf_level_from_scale_ratio(
                cluster_size / self.stride)
            gt_rf_weight[0, gy_center, gx_center] = 1.0

        results['gt_cls_map'] = gt_cls_map
        results['gt_offset_map'] = gt_offset_map
        results['gt_offset_weight'] = gt_offset_weight
        results['gt_scale_map'] = gt_scale_map
        results['gt_scale_weight'] = gt_scale_weight
        results['gt_rf_level'] = gt_rf_level
        results['gt_rf_weight'] = gt_rf_weight
        results['gt_cluster_bboxes'] = cluster_boxes.astype(np.float32)
        return results
