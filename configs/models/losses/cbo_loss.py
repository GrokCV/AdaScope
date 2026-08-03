# mmdet/models/losses/center_objectness_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepir.registry import MODELS


@MODELS.register_module()
class CenterBasedObjectnessLoss(nn.Module):
    """自定义objectness损失：基于bbox中心点生成GT heatmap，并计算分类损失。
    
    Args:
        stride (int): P5的stride，默认32。
        base_loss_type (str): 底层损失类型，支持 'FocalLoss' 或 'CrossEntropyLoss'。
        gamma (float): FocalLoss的gamma，默认2.0。
        alpha (float): FocalLoss的alpha，默认0.25。
        reduction (str): 损失归约方式，默认 'mean'。
        loss_weight (float): 损失权重，默认1.0（在head中乘以）。
    """
    def __init__(self,
                 stride=32,
                 base_loss_type='FocalLoss',
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(CenterBasedObjectnessLoss, self).__init__()
        self.stride = stride
        self.base_loss_type = base_loss_type
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        
        if base_loss_type not in ['FocalLoss', 'CrossEntropyLoss']:
            raise ValueError(f"Unsupported base_loss_type: {base_loss_type}")

    # mmdet/models/losses/center_objectness_loss.py（修改 forward）
    # deepir/models/losses/cbo_loss.py 的 forward（完整 batch 版）
    def forward(self,
                pred,                      # [B, 1, H, W] batch pred
                batch_gt_instances,         # list[InstanceData] for batch
                img_shapes,                 # list[tuple (H,W)] for batch
                device=None,
                **kwargs):
        if device is None:
            device = pred.device
        batch_size = pred.size(0)
    
    # 初始化 batch gt_objectness [B,1,H,W]
        _,map_h, map_w = pred.shape  # 假设 all images same size
        gt_objectness = torch.zeros((batch_size, 1, map_h, map_w), dtype=torch.float32, device=device)
    
        losses = []
        for i in range(batch_size):
            single_pred = pred[i]  # [1, H, W]
            gt_instances = batch_gt_instances[i]  # single InstanceData
            single_img_shape = img_shapes[i]  # (img_h, img_w)
        
            single_gt_bboxes = gt_instances.bboxes  # Tensor [num_gts,4]
        
            if single_gt_bboxes.numel() == 0:
                single_loss = self._compute_base_loss(single_pred, gt_objectness[i]) * self.loss_weight
                losses.append(single_loss)
                continue
        
        # 计算 centers（不变）
            centers_x = (single_gt_bboxes[:, 0] + single_gt_bboxes[:, 2]) / 2.0
            centers_y = (single_gt_bboxes[:, 1] + single_gt_bboxes[:, 3]) / 2.0
            grid_x = torch.floor(centers_x / self.stride).long()
            grid_y = torch.floor(centers_y / self.stride).long()
            grid_x = torch.clamp(grid_x, 0, map_w - 1)
            grid_y = torch.clamp(grid_y, 0, map_h - 1)
        
            unique_grids = torch.unique(torch.stack([grid_y, grid_x], dim=1), dim=0)
            for gy, gx in unique_grids:
                gt_objectness[i, 0, gy, gx] = 1.0  # set to batch dim i
        
            single_loss = self._compute_base_loss(single_pred, gt_objectness[i]) * self.loss_weight
            losses.append(single_loss)
    
        return torch.mean(torch.stack(losses))  # 平均 batch loss
        
    
    def _compute_base_loss(self, pred, target):
        """计算底层损失。"""
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
        if self.base_loss_type == 'FocalLoss':
            # 简单PyTorch实现FocalLoss（假设pred已sigmoid）
            prob = pred
            p_t = prob * target + (1 - prob) * (1 - target)
            alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
            modulating_factor = (1.0 - p_t) ** self.gamma
            loss = -alpha_factor * modulating_factor * F.binary_cross_entropy(prob, target, reduction='none')
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        
        elif self.base_loss_type == 'CrossEntropyLoss':
            # 用BCE（假设pred已sigmoid）
            loss = F.binary_cross_entropy(pred, target, reduction=self.reduction)
            return loss
        
        raise NotImplementedError(f"Base loss {self.base_loss_type} not implemented.")