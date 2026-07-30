from collections import deque
from typing import List, Tuple

import numpy as np
import torch
from mmdet.models.utils import gaussian_radius, gen_gaussian_target


def _gen_heatmap_from_xywh(
    gt_bboxes_xywh: torch.Tensor,
    img_shape: Tuple[int, int],
    min_overlap: float = 0.3,
    min_radius: int = 1,
) -> torch.Tensor:
    """Generate a single-channel center heatmap from xywh boxes."""
    h, w = img_shape
    heatmap = torch.zeros((1, 1, h, w), dtype=torch.float32)
    if gt_bboxes_xywh.numel() == 0:
        return heatmap

    for box in gt_bboxes_xywh:
        bw = float(box[2].item())
        bh = float(box[3].item())
        if bw <= 0 or bh <= 0:
            continue

        cx = float(box[0].item()) + bw * 0.5
        cy = float(box[1].item()) + bh * 0.5
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            continue

        center = [int(cx), int(cy)]
        radius = gaussian_radius((bh, bw), min_overlap=min_overlap)
        radius = max(min_radius, int(radius))
        gen_gaussian_target(heatmap[0, 0], center, radius)

    return heatmap


def _split_overlay_map(grid: np.ndarray) -> List[List[int]]:
    """Eight-connected component merge on the selected density grid."""
    if grid.size == 0:
        return []

    m, n = grid.shape
    visited = np.zeros((m, n), dtype=np.uint8)
    result: List[List[int]] = []

    for i in range(m):
        for j in range(n):
            if visited[i, j]:
                continue
            visited[i, j] = 1
            if grid[i, j] == 0:
                continue

            q = deque([(i, j)])
            top, left = i, j
            bottom, right = i, j

            while q:
                x, y = q.popleft()
                top = min(top, x)
                left = min(left, y)
                bottom = max(bottom, x)
                right = max(right, y)

                for nx, ny in (
                    (x + 1, y),
                    (x - 1, y),
                    (x, y + 1),
                    (x, y - 1),
                    (x + 1, y + 1),
                    (x + 1, y - 1),
                    (x - 1, y + 1),
                    (x - 1, y - 1),
                ):
                    if nx < 0 or ny < 0 or nx >= m or ny >= n:
                        continue
                    if visited[nx, ny]:
                        continue
                    visited[nx, ny] = 1
                    if grid[nx, ny] != 0:
                        q.append((nx, ny))

            result.append([top, left, min(bottom + 1, m), min(right + 1, n)])

    return result


def _find_clusters_from_heatmap(
    heatmap: np.ndarray,
    grid_size: Tuple[int, int] = (64, 40),
    topk: int = 15,
    threshold_ratio: float = 10.0 / 11.0,
) -> np.ndarray:
    """
    Reproduce the core cluster.py logic on a center heatmap.

    Returns:
        np.ndarray of shape [N, 4] in xywh.
    """
    if heatmap.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    grid_w, grid_h = grid_size
    if grid_w <= 0 or grid_h <= 0:
        raise ValueError("grid_size must be positive")

    work = 1.0 - heatmap
    max_val = float(np.max(work))
    if max_val <= 1e-12:
        return np.zeros((0, 4), dtype=np.float32)
    work = 255.0 * work / max_val
    gray = work[0, 0].astype(np.uint8)

    thresh = float(threshold_ratio) * 255.0
    binary = np.where(gray <= thresh, 255, 0).astype(np.uint8)
    binmap = (binary == 255).astype(np.float32)

    h, w = binary.shape
    x_edges = np.linspace(0, w, grid_w + 1)
    y_edges = np.linspace(0, h, grid_h + 1)

    density_map = np.zeros((grid_w, grid_h), dtype=np.float32)
    for i in range(grid_w):
        for j in range(grid_h):
            x1 = int(np.floor(x_edges[i]))
            y1 = int(np.floor(y_edges[j]))
            x2 = int(np.ceil(x_edges[i + 1]))
            y2 = int(np.ceil(y_edges[j + 1]))
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue
            density_map[i, j] = float(binmap[y1:y2, x1:x2].sum())

    flat = density_map.reshape(-1)
    k = min(max(int(topk), 1), flat.size)
    idx = flat.argsort()[-k:][::-1]

    grid = np.zeros((grid_w, grid_h), dtype=np.uint8)
    for item in idx:
        x_idx = item // grid_h
        y_idx = item % grid_h
        grid[x_idx, y_idx] = 255

    comps = _split_overlay_map(grid)
    if not comps:
        return np.zeros((0, 4), dtype=np.float32)

    result = np.array(comps, dtype=np.float32)
    x_idx1 = np.clip(result[:, 0].astype(np.int64), 0, grid_w)
    x_idx2 = np.clip(result[:, 2].astype(np.int64), 0, grid_w)
    y_idx1 = np.clip(result[:, 1].astype(np.int64), 0, grid_h)
    y_idx2 = np.clip(result[:, 3].astype(np.int64), 0, grid_h)

    result[:, 0] = x_edges[x_idx1]
    result[:, 2] = x_edges[x_idx2]
    result[:, 1] = y_edges[y_idx1]
    result[:, 3] = y_edges[y_idx2]
    result[:, 2] = result[:, 2] - result[:, 0]
    result[:, 3] = result[:, 3] - result[:, 1]
    return result


def cluster_boxes_from_gt_bboxes(
    gt_bboxes_xyxy: np.ndarray,
    img_shape: Tuple[int, int],
    grid_size: Tuple[int, int] = (64, 40),
    topk: int = 15,
    threshold_ratio: float = 10.0 / 11.0,
    min_overlap: float = 0.3,
    min_radius: int = 1,
) -> np.ndarray:
    """
    Convert single-object GT boxes into cluster GT boxes with cluster.py-like flow.

    Args:
        gt_bboxes_xyxy: [N, 4] boxes in xyxy.
        img_shape: (H, W)
    Returns:
        [K, 4] cluster boxes in xyxy.
    """
    if gt_bboxes_xyxy.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    h, w = int(img_shape[0]), int(img_shape[1])
    boxes = np.asarray(gt_bboxes_xyxy, dtype=np.float32)
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    xywh = np.zeros_like(boxes, dtype=np.float32)
    xywh[:, 0] = boxes[:, 0]
    xywh[:, 1] = boxes[:, 1]
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

    heatmap = _gen_heatmap_from_xywh(
        torch.as_tensor(xywh, dtype=torch.float32),
        img_shape=(h, w),
        min_overlap=min_overlap,
        min_radius=min_radius,
    )
    clusters_xywh = _find_clusters_from_heatmap(
        heatmap.numpy(),
        grid_size=grid_size,
        topk=topk,
        threshold_ratio=threshold_ratio,
    )
    if clusters_xywh.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    clusters_xyxy = np.zeros_like(clusters_xywh, dtype=np.float32)
    clusters_xyxy[:, 0] = clusters_xywh[:, 0]
    clusters_xyxy[:, 1] = clusters_xywh[:, 1]
    clusters_xyxy[:, 2] = clusters_xywh[:, 0] + clusters_xywh[:, 2]
    clusters_xyxy[:, 3] = clusters_xywh[:, 1] + clusters_xywh[:, 3]

    clusters_xyxy[:, 0::2] = np.clip(clusters_xyxy[:, 0::2], 0, w)
    clusters_xyxy[:, 1::2] = np.clip(clusters_xyxy[:, 1::2], 0, h)
    valid = (clusters_xyxy[:, 2] > clusters_xyxy[:, 0]) & (clusters_xyxy[:, 3] > clusters_xyxy[:, 1])
    return clusters_xyxy[valid].astype(np.float32)
