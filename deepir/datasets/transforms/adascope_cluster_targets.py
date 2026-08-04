import os.path as osp
import xml.etree.ElementTree as ET

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class AdaScopeClusterTargets(BaseTransform):
    """Generate C5 targets from pre-annotated cluster XML boxes.

    This transform keeps the same supervision outputs as AdaScopeClusterTargets,
    but uses cluster boxes parsed from sibling XML files:
    `{img_id}_with_clusters.xml`.
    """

    def __init__(self,
                 stride=32,
                 rf_scale_bins=(1.5, 3.0),
                 cluster_xml_suffix='_with_clusters.xml',
                 cluster_tag='cluster',
                 cluster_name='Target',
                 cluster_xml_subdir='',
                 missing_policy='empty',
                 minus_one=True,
                 scale_eps=1e-6):
        self.stride = int(stride)
        self.rf_scale_bins = rf_scale_bins
        self.cluster_xml_suffix = str(cluster_xml_suffix)
        self.cluster_tag = str(cluster_tag)
        self.cluster_name = str(cluster_name)
        self.cluster_xml_subdir = str(cluster_xml_subdir)
        self.missing_policy = str(missing_policy)
        self.minus_one = bool(minus_one)
        self.scale_eps = float(scale_eps)

        if self.missing_policy not in ('empty', 'error'):
            raise ValueError("missing_policy must be 'empty' or 'error'.")

    def _rf_level_from_scale_ratio(self, scale_ratio):
        if len(self.rf_scale_bins) < 2:
            raise ValueError('rf_scale_bins must provide two thresholds.')
        bin0 = float(self.rf_scale_bins[0])
        bin1 = float(self.rf_scale_bins[1])
        if scale_ratio < bin0:
            return 0
        if scale_ratio < bin1:
            return 1
        return 2

    def _cluster_xml_path(self, results):
        xml_path = results.get('xml_path', None)
        if not xml_path:
            return None

        img_id = results.get('img_id', None)
        if img_id is None:
            img_id = osp.splitext(osp.basename(xml_path))[0]

        xml_dir = osp.dirname(xml_path)
        if self.cluster_xml_subdir:
            xml_dir = osp.join(xml_dir, self.cluster_xml_subdir)
        return osp.join(xml_dir, f'{img_id}{self.cluster_xml_suffix}')

    def _load_cluster_boxes(self, results):
        cluster_xml_path = self._cluster_xml_path(results)
        if cluster_xml_path is None:
            if self.missing_policy == 'error':
                raise KeyError('results does not contain xml_path for cluster GT loading.')
            return np.zeros((0, 4), dtype=np.float32)

        if not osp.isfile(cluster_xml_path):
            if self.missing_policy == 'error':
                raise FileNotFoundError(f'Cluster XML not found: {cluster_xml_path}')
            return np.zeros((0, 4), dtype=np.float32)

        try:
            root = ET.parse(cluster_xml_path).getroot()
        except Exception as exc:
            if self.missing_policy == 'error':
                raise RuntimeError(f'Failed to parse cluster XML: {cluster_xml_path}') from exc
            return np.zeros((0, 4), dtype=np.float32)

        boxes = []
        for node in root.findall(self.cluster_tag):
            name_node = node.find('name')
            if name_node is not None and name_node.text is not None:
                if name_node.text != self.cluster_name:
                    continue

            bndbox = node.find('bndbox')
            if bndbox is None:
                continue

            xmin_node = bndbox.find('xmin')
            ymin_node = bndbox.find('ymin')
            xmax_node = bndbox.find('xmax')
            ymax_node = bndbox.find('ymax')
            if None in (xmin_node, ymin_node, xmax_node, ymax_node):
                continue

            bbox = np.array([
                float(xmin_node.text),
                float(ymin_node.text),
                float(xmax_node.text),
                float(ymax_node.text),
            ], dtype=np.float32)
            if self.minus_one:
                bbox = bbox - 1.0
            boxes.append(bbox)

        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return np.stack(boxes, axis=0).astype(np.float32)

    @staticmethod
    def _safe_shape(results, key, fallback):
        value = results.get(key, None)
        if value is None:
            return fallback
        return value

    def _rescale_boxes_to_current_img(self, boxes, results):
        if boxes.shape[0] == 0:
            return boxes

        img_h, img_w = self._safe_shape(results, 'img_shape', (0, 0))[:2]
        ori_h, ori_w = self._safe_shape(results, 'ori_shape', (img_h, img_w))[:2]

        if ori_h <= 0 or ori_w <= 0 or img_h <= 0 or img_w <= 0:
            return boxes

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

    def transform(self, results):
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

        cluster_boxes = self._load_cluster_boxes(results)
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
            dx = (x_center - grid_cx) / self.stride
            dy = (y_center - grid_cy) / self.stride
            gt_offset_map[0, gy_center, gx_center] = dx
            gt_offset_map[1, gy_center, gx_center] = dy
            gt_offset_weight[0, gy_center, gx_center] = 1.0

            box_w = max(float(bbox[2] - bbox[0]), self.scale_eps)
            box_h = max(float(bbox[3] - bbox[1]), self.scale_eps)
            cluster_size = np.sqrt(box_w * box_h)
            scale_target = np.log(max(cluster_size / self.stride, self.scale_eps))
            gt_scale_map[0, gy_center, gx_center] = scale_target
            gt_scale_weight[0, gy_center, gx_center] = 1.0

            rf_level = self._rf_level_from_scale_ratio(cluster_size / self.stride)
            gt_rf_level[0, gy_center, gx_center] = rf_level
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
