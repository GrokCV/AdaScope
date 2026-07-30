import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from deepir.utils import cluster_boxes_from_gt_bboxes

@TRANSFORMS.register_module()
class GenerateC5Targets(BaseTransform):
    """
    为 C5 + Refiner 生成训练目标。

    先将多个单目标框进行成簇，再在同一网格上生成：
    - gt_cls_map:         [H, W]   簇中心分类图（0/1）
    - gt_offset_map:      [2, H, W]中心偏移监督 (dx, dy)，已按 stride 归一化
    - gt_offset_weight:   [1, H, W]偏移损失掩码（仅正样本为 1）
    - gt_scale_map:       [1, H, W]面积缩放监督（log-scale）
    - gt_scale_weight:    [1, H, W]缩放损失掩码（仅正样本为 1）
    - gt_rf_level:        [1, H, W]感受野三档标签（-1/0/1/2）
    - gt_rf_weight:       [1, H, W]感受野损失掩码（仅正样本为 1）
    - gt_cluster_bboxes:  [K, 4]   成簇后的簇框（xyxy）
    """
    def __init__(self,
                 stride=32,
                 cluster_gt=True,
                 cluster_grid_size=(64, 40),
                 cluster_topk=15,
                 cluster_thresh_ratio=10.0 / 11.0,
                 cluster_min_overlap=0.3,
                 cluster_min_radius=1,
                 rf_scale_bins=(1.5, 3.0),
                 scale_eps=1e-6):
        self.stride = stride
        self.cluster_gt = cluster_gt
        self.cluster_grid_size = cluster_grid_size
        self.cluster_topk = cluster_topk
        self.cluster_thresh_ratio = cluster_thresh_ratio
        self.cluster_min_overlap = cluster_min_overlap
        self.cluster_min_radius = cluster_min_radius
        self.rf_scale_bins = rf_scale_bins
        self.scale_eps = scale_eps

    def _build_cluster_boxes_from_heatmap(self, boxes_xyxy: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
        return cluster_boxes_from_gt_bboxes(
            gt_bboxes_xyxy=boxes_xyxy,
            img_shape=(img_h, img_w),
            grid_size=tuple(self.cluster_grid_size),
            topk=int(self.cluster_topk),
            threshold_ratio=float(self.cluster_thresh_ratio),
            min_overlap=float(self.cluster_min_overlap),
            min_radius=int(self.cluster_min_radius),
        )

    def _build_cluster_boxes(self, boxes_xyxy: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
        if not self.cluster_gt:
            return boxes_xyxy
        return self._build_cluster_boxes_from_heatmap(boxes_xyxy, img_h, img_w)

    def _rf_level_from_scale_ratio(self, scale_ratio: float) -> int:
        if len(self.rf_scale_bins) < 2:
            raise ValueError('rf_scale_bins must have two thresholds for 3-level RF supervision.')
        b0, b1 = float(self.rf_scale_bins[0]), float(self.rf_scale_bins[1])
        if scale_ratio < b0:
            return 0
        if scale_ratio < b1:
            return 1
        return 2

    def transform(self, results: dict) -> dict:
        h, w = results['img_shape'][:2]
        gt_bboxes = results['gt_bboxes']

        grid_h, grid_w = h // self.stride, w // self.stride

        gt_cls_map = np.zeros((grid_h, grid_w), dtype=np.float32)
        gt_offset_map = np.zeros((2, grid_h, grid_w), dtype=np.float32)
        gt_offset_weight = np.zeros((1, grid_h, grid_w), dtype=np.float32)
        gt_scale_map = np.zeros((1, grid_h, grid_w), dtype=np.float32)
        gt_scale_weight = np.zeros((1, grid_h, grid_w), dtype=np.float32)
        gt_rf_level = np.full((1, grid_h, grid_w), -1, dtype=np.int64)
        gt_rf_weight = np.zeros((1, grid_h, grid_w), dtype=np.float32)

        boxes_xyxy = gt_bboxes.tensor.detach().cpu().numpy().astype(np.float32)
        cluster_boxes = self._build_cluster_boxes(boxes_xyxy, h, w)

        # 遍历所有簇框，构造簇中心分类/偏移/缩放监督
        for bbox in cluster_boxes:
            x_center = 0.5 * (bbox[0] + bbox[2])
            y_center = 0.5 * (bbox[1] + bbox[3])
            gx_c = int(x_center / self.stride)
            gy_c = int(y_center / self.stride)

            if 0 <= gx_c < grid_w and 0 <= gy_c < grid_h:
                gt_cls_map[gy_c, gx_c] = 1.0

                grid_cx = (gx_c + 0.5) * self.stride
                grid_cy = (gy_c + 0.5) * self.stride

                dx = (x_center - grid_cx) / self.stride
                dy = (y_center - grid_cy) / self.stride
                gt_offset_map[0, gy_c, gx_c] = dx
                gt_offset_map[1, gy_c, gx_c] = dy
                gt_offset_weight[0, gy_c, gx_c] = 1.0

                bw = max(float(bbox[2] - bbox[0]), self.scale_eps)
                bh = max(float(bbox[3] - bbox[1]), self.scale_eps)
                cluster_size = np.sqrt(bw * bh)
                scale_target = np.log(max(cluster_size / self.stride, self.scale_eps))
                gt_scale_map[0, gy_c, gx_c] = scale_target
                gt_scale_weight[0, gy_c, gx_c] = 1.0

                rf_level = self._rf_level_from_scale_ratio(cluster_size / self.stride)
                gt_rf_level[0, gy_c, gx_c] = rf_level
                gt_rf_weight[0, gy_c, gx_c] = 1.0

        results['gt_cls_map'] = gt_cls_map
        results['gt_offset_map'] = gt_offset_map
        results['gt_offset_weight'] = gt_offset_weight
        results['gt_scale_map'] = gt_scale_map
        results['gt_scale_weight'] = gt_scale_weight
        results['gt_rf_level'] = gt_rf_level
        results['gt_rf_weight'] = gt_rf_weight
        results['gt_cluster_bboxes'] = cluster_boxes
        return results
