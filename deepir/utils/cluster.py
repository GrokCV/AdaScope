import os
import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET
from xml.dom import minidom
from mmdet.models.utils import gaussian_radius, gen_gaussian_target


num_classes=1


def gen_heatmap(gt_bboxes, gt_labels, img_shape):
    '''
    gt_bboxes: [x, y, w, h]  # 假设xywh格式，从你的代码推测
    '''
    H, W = img_shape
    center_heatmap_target = torch.zeros([1, num_classes, H, W], dtype=torch.float32)  # 初始化（batch=1）

    if gt_bboxes.numel() == 0 or gt_labels.numel() == 0:  # 新增：如果无GT，返回全零热图
        return center_heatmap_target

    gt_bboxes = gt_bboxes.clone().detach()  # 修复警告：clone().detach() 替换 torch.tensor
    gt_labels = gt_labels.clone().detach() - 1  # 同上，假设label从1开始

    for j in range(len(gt_bboxes)):
        gt_bbox = gt_bboxes[j]
        gt_label = gt_labels[j]

        # 新增检查：跳过无效bbox（w/h <=0 或 出界）
        if gt_bbox[2] <= 0 or gt_bbox[3] <= 0:  # w/h 无效
            continue
        ct_x = gt_bbox[0] + gt_bbox[2] / 2
        ct_y = gt_bbox[1] + gt_bbox[3] / 2
        if ct_x < 0 or ct_y < 0 or ct_x >= W or ct_y >= H:  # 中心出界
            continue

        ct_int = torch.tensor([int(ct_x), int(ct_y)], dtype=torch.long)  # 整数中心

        # 计算radius（添加min_overlap调整，确保radius>0）
        radius = gaussian_radius((gt_bbox[3], gt_bbox[2]), min_overlap=0.3)  # 宽高顺序根据你的bbox格式
        radius = max(1, int(radius))  # 新增：最小radius=1，避免0

        # 新增检查：计算切片范围，确保非空
        left = min(ct_int[0], radius)
        right = min(W - ct_int[0] - 1, radius)
        top = min(ct_int[1], radius)
        bottom = min(H - ct_int[1] - 1, radius)
        if left + right < 1 or top + bottom < 1:  # 切片太小或负，跳过
            continue

        ind = gt_label.long()  # 类别索引
        gen_gaussian_target(center_heatmap_target[0, ind], ct_int.tolist(), radius)  # 调用，现在大小匹配

    return center_heatmap_target

def boxes_intersect(boxA, boxB):
    """
    Checks if two bounding boxes overlap.
    Boxes are in [x1, y1, x2, y2] format.
    """
    # Determine the coordinates of the intersection rectangle
    x_inter1 = max(boxA[0], boxB[0])
    y_inter1 = max(boxA[1], boxB[1])
    x_inter2 = min(boxA[2], boxB[2])
    y_inter2 = min(boxA[3], boxB[3])

    # A positive intersection area means they overlap
    return x_inter2 > x_inter1 and y_inter2 > y_inter1        


def LSM(center_heatmap_preds, img_metas):  # Removed 'self' assuming it's not a class method
    '''
    Args:
        center_heatmap_preds (list[Tensor]):  (N, C, H, W)
    '''
    center_heatmap_pred = center_heatmap_preds[0]
    locmap = torch.max(center_heatmap_pred, dim=1, keepdim=True)[0].cpu().numpy()
    
    coord = findclusters(locmap, find_max=True, fname=["test"])  # Removed 'self'

    '''for visualization'''
    border_pixs = [img_meta['border'] for img_meta in img_metas]
    # coord [x, y, w, h]
    coord[:, 0] = coord[:, 0] - border_pixs[0][2]
    coord[:, 1] = coord[:, 1] - border_pixs[0][0]
    return coord

def findclusters(heatmap, find_max, fname): # 确保移除了 self
    # --- 主要修改点：将网格尺寸参数化 ---
    grid_w, grid_h = 64, 40  # 尝试将 16x10 提高到 32x20，你也可以尝试 64x40 等
    topk = 15                # 你也可以尝试减少 topk 的值，比如 10，让筛选更严格
    
    heatmap = 1 - heatmap
    heatmap = 255 * heatmap / np.max(heatmap)
    heatmap = heatmap[0][0]

    gray = heatmap.astype(np.uint8)
    Thresh = 10.0 / 11.0 * 255.0
    ret, binary = cv2.threshold(gray, Thresh, 255, cv2.THRESH_BINARY_INV)

    binmap = binary.copy()
    binmap[binmap == 255] = 1
    
    # 使用新的网格尺寸
    density_map = np.zeros((grid_w, grid_h))
    w_stride = binary.shape[1] // grid_w
    h_stride = binary.shape[0] // grid_h
    for i in range(grid_w):
        for j in range(grid_h):
            x1 = w_stride * i
            y1 = h_stride * j
            x2 = min(x1 + w_stride, binary.shape[1])
            y2 = min(y1 + h_stride, binary.shape[0])
            density_map[i][j] = binmap[y1:y2, x1:x2].sum()

    d = density_map.flatten()
    idx = d.argsort()[-topk:][::-1]
    grid_idx = idx.copy()
    
    # 使用新的网格尺寸
    grid = np.zeros((grid_w, grid_h))
    for item in grid_idx:
        x1 = item // grid_h
        y1 = item % grid_h
        grid[x1, y1] = 255
        
    result = split_overlay_map(grid)
    if not result: # 检查 result 是否为空
        return np.empty((0, 4))
        
    result = np.array(result)
    result[:, 0::2] = np.clip(result[:, 0::2] * w_stride, 0, binary.shape[1])
    result[:, 1::2] = np.clip(result[:, 1::2] * h_stride, 0, binary.shape[0])

    for i in range(len(result)):
        # 注意：result[i, 0] 等已经是整数，无需再次转换
        p1 = (result[i, 0], result[i, 1])
        p2 = (result[i, 2], result[i, 3])
        cv2.rectangle(binary, p1, p2, (255, 0, 0), 2)

    cv2.imwrite("binary_heatmap_%s_grid%dx%d.jpg" % (fname[0], grid_w, grid_h), binary)

    result[:, 2] = result[:, 2] - result[:, 0]
    result[:, 3] = result[:, 3] - result[:, 1]
    return result

def split_overlay_map(grid):
    # This function is modified from https://github.com/Cli98/DMNet
    """
        Conduct eight-connected-component methods on grid to connnect all pixel within the similar region
        :param grid: desnity mask to connect
        :return: merged regions for cropping purpose
    """
    if grid is None or grid[0] is None:
        return 0
    # Assume overlap_map is a 2d feature map
    m, n = grid.shape
    visit = [[0 for _ in range(n)] for _ in range(m)]
    count, queue, result = 0, [], []
    for i in range(m):
        for j in range(n):
            if not visit[i][j]:
                if grid[i][j] == 0:
                    visit[i][j] = 1
                    continue
                queue.append([i, j])
                top, left = float("inf"), float("inf")
                bot, right = float("-inf"), float("-inf")
                while queue:
                    i_cp, j_cp = queue.pop(0)
                    if 0 <= i_cp < m and 0 <= j_cp < n and grid[i_cp][j_cp] == 255:
                        top = min(i_cp, top)
                        left = min(j_cp, left)
                        bot = max(i_cp, bot)
                        right = max(j_cp, right)
                    if 0 <= i_cp < m and 0 <= j_cp < n and not visit[i_cp][j_cp]:
                        visit[i_cp][j_cp] = 1
                        if grid[i_cp][j_cp] == 255:
                            queue.append([i_cp, j_cp + 1])
                            queue.append([i_cp + 1, j_cp])
                            queue.append([i_cp, j_cp - 1])
                            queue.append([i_cp - 1, j_cp])

                            queue.append([i_cp - 1, j_cp - 1])
                            queue.append([i_cp - 1, j_cp + 1])
                            queue.append([i_cp + 1, j_cp - 1])
                            queue.append([i_cp + 1, j_cp + 1])
                count += 1
                result.append([max(0, top), max(0, left), min(bot+1, m), min(right+1, n)])

    return result

def LSM(center_heatmap_preds, img_metas):  # Removed 'self' assuming it's not a class method
    '''
    Args:
        center_heatmap_preds (list[Tensor]):  (N, C, H, W)
    '''
    center_heatmap_pred = center_heatmap_preds[0]
    locmap = torch.max(center_heatmap_pred, dim=1, keepdim=True)[0].cpu().numpy()
    
    coord = findclusters(locmap, find_max=True, fname=["test"])  # Removed 'self'

    '''for visualization'''
    border_pixs = [img_meta['border'] for img_meta in img_metas]
    # coord [x, y, w, h]
    coord[:, 0] = coord[:, 0] - border_pixs[0][2]
    coord[:, 1] = coord[:, 1] - border_pixs[0][0]
    return coord

def findclusters(heatmap, find_max, fname): # 确保移除了 self
    # --- 主要修改点：将网格尺寸参数化 ---
    grid_w, grid_h = 64, 40  # 尝试将 16x10 提高到 32x20，你也可以尝试 64x40 等
    topk = 15                # 你也可以尝试减少 topk 的值，比如 10，让筛选更严格
    
    heatmap = 1 - heatmap
    heatmap = 255 * heatmap / np.max(heatmap)
    heatmap = heatmap[0][0]

    gray = heatmap.astype(np.uint8)
    Thresh = 10.0 / 11.0 * 255.0
    ret, binary = cv2.threshold(gray, Thresh, 255, cv2.THRESH_BINARY_INV)

    binmap = binary.copy()
    binmap[binmap == 255] = 1
    
    # 使用新的网格尺寸
    density_map = np.zeros((grid_w, grid_h))
    w_stride = binary.shape[1] // grid_w
    h_stride = binary.shape[0] // grid_h
    for i in range(grid_w):
        for j in range(grid_h):
            x1 = w_stride * i
            y1 = h_stride * j
            x2 = min(x1 + w_stride, binary.shape[1])
            y2 = min(y1 + h_stride, binary.shape[0])
            density_map[i][j] = binmap[y1:y2, x1:x2].sum()

    d = density_map.flatten()
    idx = d.argsort()[-topk:][::-1]
    grid_idx = idx.copy()
    
    # 使用新的网格尺寸
    grid = np.zeros((grid_w, grid_h))
    for item in grid_idx:
        x1 = item // grid_h
        y1 = item % grid_h
        grid[x1, y1] = 255
        
    result = split_overlay_map(grid)
    if not result: # 检查 result 是否为空
        return np.empty((0, 4))
        
    result = np.array(result)
    result[:, 0::2] = np.clip(result[:, 0::2] * w_stride, 0, binary.shape[1])
    result[:, 1::2] = np.clip(result[:, 1::2] * h_stride, 0, binary.shape[0])

    for i in range(len(result)):
        # 注意：result[i, 0] 等已经是整数，无需再次转换
        p1 = (result[i, 0], result[i, 1])
        p2 = (result[i, 2], result[i, 3])
        cv2.rectangle(binary, p1, p2, (255, 0, 0), 2)

    cv2.imwrite("binary_heatmap_%s_grid%dx%d.jpg" % (fname[0], grid_w, grid_h), binary)

    result[:, 2] = result[:, 2] - result[:, 0]
    result[:, 3] = result[:, 3] - result[:, 1]
    return result

def split_overlay_map(grid):
    # This function is modified from https://github.com/Cli98/DMNet
    """
        Conduct eight-connected-component methods on grid to connnect all pixel within the similar region
        :param grid: desnity mask to connect
        :return: merged regions for cropping purpose
    """
    if grid is None or grid[0] is None:
        return 0
    # Assume overlap_map is a 2d feature map
    m, n = grid.shape
    visit = [[0 for _ in range(n)] for _ in range(m)]
    count, queue, result = 0, [], []
    for i in range(m):
        for j in range(n):
            if not visit[i][j]:
                if grid[i][j] == 0:
                    visit[i][j] = 1
                    continue
                queue.append([i, j])
                top, left = float("inf"), float("inf")
                bot, right = float("-inf"), float("-inf")
                while queue:
                    i_cp, j_cp = queue.pop(0)
                    if 0 <= i_cp < m and 0 <= j_cp < n and grid[i_cp][j_cp] == 255:
                        top = min(i_cp, top)
                        left = min(j_cp, left)
                        bot = max(i_cp, bot)
                        right = max(j_cp, right)
                    if 0 <= i_cp < m and 0 <= j_cp < n and not visit[i_cp][j_cp]:
                        visit[i_cp][j_cp] = 1
                        if grid[i_cp][j_cp] == 255:
                            queue.append([i_cp, j_cp + 1])
                            queue.append([i_cp + 1, j_cp])
                            queue.append([i_cp, j_cp - 1])
                            queue.append([i_cp - 1, j_cp])

                            queue.append([i_cp - 1, j_cp - 1])
                            queue.append([i_cp - 1, j_cp + 1])
                            queue.append([i_cp + 1, j_cp - 1])
                            queue.append([i_cp + 1, j_cp + 1])
                count += 1
                result.append([max(0, top), max(0, left), min(bot+1, m), min(right+1, n)])

    return result
