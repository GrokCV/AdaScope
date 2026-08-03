# c5_parallel_head.py
import torch
import torch.nn as nn
from typing import Tuple, List 
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead

@MODELS.register_module()
class C5ParallelHead(BaseDenseHead):
    """
    一个并行的检测头:
    - 输入 (来自 C5): 
        - 分支 1: 0/1 分类
        - 分支 2: 目标计数
    """
    def __init__(self,
                 in_channels,
                 feat_channels=256,
                 # --- 核心修改：增加 num_convs 参数 ---
                 num_convs=4, # 默认堆叠 4 层
                 # --- 修改结束 ---
                 loss_cls: dict = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
             
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Normal', 
                     std=0.01, 
                     override=[dict(name='cls_pred')])
                ):
        super().__init__(init_cfg=init_cfg)
        
        self.loss_cls = MODELS.build(loss_cls)
        

        # --- 核心修改：构建更深的子网络 ---
        
        # 1. 分类子网络 (Cls Subnet)
        cls_subnet = []
        # 第一层：从 in_channels (2048) 降到 feat_channels (256)
        cls_subnet.append(nn.Conv2d(in_channels, feat_channels, 3, stride=2, padding=1))
        cls_subnet.append(nn.ReLU())
        # 堆叠 num_convs - 1 层
        for _ in range(num_convs - 1):
            cls_subnet.append(nn.Conv2d(feat_channels, feat_channels, 3, stride=1, padding=1))
            cls_subnet.append(nn.ReLU())
        # 最终的预测层
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.cls_pred = nn.Conv2d(feat_channels, 1, 1)

       

    def forward(self, feats: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        c5_feat = feats[0] # (B, 2048, 32, 32)
        
        # --- 分类分支 ---
        # --- 核心修改：使用子网络 ---
        cls_feat = self.cls_subnet(c5_feat) 
        cls_score = self.cls_pred(cls_feat)
        
      
        
        return (cls_score)

    # ... (loss, loss_by_feat, predict 等其他方法保持不变) ...
    
    # ... (确保 loss, loss_by_feat, predict 方法存在且正确) ...

    def loss(self, 
             x: Tuple[torch.Tensor], 
             batch_data_samples: List[InstanceData]) -> dict:
        cls_score = self(x)
        losses = self.loss_by_feat(
            (cls_score),
            batch_data_samples
        )
        return losses
    
    def loss_by_feat(self,
                     preds: Tuple[torch.Tensor],
                     batch_data_samples: List[InstanceData]) -> dict:
        cls_score = preds
        gt_cls_maps = torch.stack([
            torch.tensor(ds.gt_cls_map, device=cls_score.device) 
            for ds in batch_data_samples
        ]).unsqueeze(1) 

       

        loss_cls = self.loss_cls(cls_score, gt_cls_maps)

        
        

            
        return dict(loss_cls=loss_cls)

    def predict(self, 
                feats: Tuple[torch.Tensor], 
                batch_data_samples: List[InstanceData], 
                rescale: bool = False):
        
        cls_score = self(feats)
        cls_prob = cls_score.sigmoid()
        predictions_list = [] 
        
        for i in range(len(batch_data_samples)):
            data_sample = batch_data_samples[i]
            pred_instances = InstanceData() 
            pred_instances.cls_heatmap = cls_prob[i]
            
            predictions_list.append(pred_instances)
            
            gt_instances = InstanceData()
            gt_instances.gt_cls_map = torch.tensor(data_sample.gt_cls_map)
            
            data_sample.gt_instances = gt_instances
            
        return predictions_list