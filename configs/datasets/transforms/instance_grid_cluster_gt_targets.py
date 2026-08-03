import os.path as osp
import xml.etree.ElementTree as ET

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class GenerateC5InstanceGridTargetsFromClusterGT(BaseTransform):
    """Generate binary grid targets from instance boxes while keeping cluster GT.

    Label rule:
    - A grid cell is positive if any instance box overlaps that cell.
    - Otherwise it is negative.

    Cluster XML boxes are still loaded into ``gt_cluster_bboxes`` so downstream
    refiner supervision/evaluation can stay unchanged.
    """

    def __init__(
        self,
        stride=32,
        cluster_xml_suffix='_with_clusters.xml',
        cluster_tag='cluster',
        cluster_name='Target',
        cluster_xml_subdir='',
        missing_policy='empty',
        minus_one=True,
    ) -> None:
        self.stride = int(stride)
        self.cluster_xml_suffix = str(cluster_xml_suffix)
        self.cluster_tag = str(cluster_tag)
        self.cluster_name = str(cluster_name)
        self.cluster_xml_subdir = str(cluster_xml_subdir)
        self.missing_policy = str(missing_policy)
        self.minus_one = bool(minus_one)

        if self.missing_policy not in ('empty', 'error'):
            raise ValueError("missing_policy must be 'empty' or 'error'.")

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
        return fallback if value is None else value

    def _rescale_boxes_to_current_img(self, boxes, results):
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

    def _extract_instance_boxes(self, results):
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

    def _fill_grid_from_instances(self, gt_cls_map, instance_boxes):
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

        instance_boxes = self._extract_instance_boxes(results)
        self._fill_grid_from_instances(gt_cls_map, instance_boxes)

        cluster_boxes = self._load_cluster_boxes(results)
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
