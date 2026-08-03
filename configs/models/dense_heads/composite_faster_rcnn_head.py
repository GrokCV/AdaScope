from typing import List, Tuple

import copy
import torch
import torch.nn as nn
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from torch import Tensor


@MODELS.register_module()
class CompositeFasterRCNNHead(nn.Module):
    """Wrap ``rpn_head + roi_head`` as a composite global-head interface.

    The module intentionally exposes only ``loss`` and ``predict`` so it can be
    consumed by the existing plugin detector the same way as a dense bbox head.
    """

    def __init__(
        self,
        rpn_head,
        roi_head,
        train_cfg=None,
        test_cfg=None,
    ) -> None:
        super().__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        rpn_cfg = copy.deepcopy(rpn_head)
        rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
        rpn_test_cfg = test_cfg.rpn if test_cfg is not None else None
        rpn_cfg.update(train_cfg=rpn_train_cfg, test_cfg=rpn_test_cfg)
        if rpn_cfg.get('num_classes', None) is None:
            rpn_cfg.update(num_classes=1)
        else:
            rpn_cfg.update(num_classes=1)
        self.rpn_head = MODELS.build(rpn_cfg)

        roi_cfg = copy.deepcopy(roi_head)
        roi_train_cfg = train_cfg.rcnn if train_cfg is not None else None
        roi_test_cfg = test_cfg.rcnn if test_cfg is not None else None
        roi_cfg.update(train_cfg=roi_train_cfg, test_cfg=roi_test_cfg)
        self.roi_head = MODELS.build(roi_cfg)

    @property
    def with_rpn(self) -> bool:
        return self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        return self.roi_head is not None

    def _rpn_data_samples(self, batch_data_samples: SampleList) -> SampleList:
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        for data_sample in rpn_data_samples:
            if hasattr(data_sample, 'gt_instances') and hasattr(
                    data_sample.gt_instances, 'labels'):
                data_sample.gt_instances.labels = torch.zeros_like(
                    data_sample.gt_instances.labels)
        return rpn_data_samples

    def loss(self, feats: Tuple[Tensor, ...],
             batch_data_samples: SampleList) -> dict:
        losses = {}
        proposal_cfg = None
        if self.train_cfg is not None:
            proposal_cfg = self.train_cfg.get(
                'rpn_proposal', self.test_cfg.rpn if self.test_cfg else None)

        rpn_data_samples = self._rpn_data_samples(batch_data_samples)
        rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
            feats, rpn_data_samples, proposal_cfg=proposal_cfg)
        for key, value in rpn_losses.items():
            if 'loss' in key and 'rpn' not in key:
                losses[f'rpn_{key}'] = value
            else:
                losses[key] = value

        roi_losses = self.roi_head.loss(feats, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)
        return losses

    def predict(self,
                feats: Tuple[Tensor, ...],
                batch_data_samples: SampleList,
                rescale: bool = False) -> List[InstanceData]:
        rpn_results_list = self.rpn_head.predict(
            feats, batch_data_samples, rescale=False)
        return self.roi_head.predict(
            feats, rpn_results_list, batch_data_samples, rescale=rescale)
