import torch
import torch.nn as nn
from torch import Tensor

from mmdet.models.losses.utils import weighted_loss
from mmdet.registry import MODELS


@weighted_loss
def yolc_gwd_loss(pred: Tensor, target: Tensor) -> Tensor:
    """A lightweight distance loss used by YOLC for box center/size targets."""
    if target.numel() == 0:
        return pred.sum() * 0

    distance = torch.square(pred - target).sum(dim=-1).sqrt()
    distance = torch.log1p(distance)
    return 1 - 1 / (1.0 + distance)


@MODELS.register_module()
class YOLCGWDLoss(nn.Module):
    def __init__(self, reduction: str = 'mean', loss_weight: float = 1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Tensor = None,
                avg_factor: float = None,
                reduction_override: str = None) -> Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override or self.reduction
        loss_bbox = self.loss_weight * yolc_gwd_loss(
            pred,
            target,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return loss_bbox
