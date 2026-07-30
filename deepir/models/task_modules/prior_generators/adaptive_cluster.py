import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import RoIAlign
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from mmdet.models.losses import IoULoss
from mmengine.model import BaseModule
from mmdet.structures.bbox import bbox_overlaps
from mmengine.logging import MessageHub  # <--- 关键导入

@MODELS.register_module()
class ClusterAdaptiveModule(BaseModule):
    def __init__(self, 
                 in_channels=2048,   
                 feat_channels=256,  
                 roi_size=7,         
                 # 修改配置：支持初始权重、最终权重和衰减周期
                 loss_cfg=dict(
                     center_init=5.0, center_min=0.5, # 中心点权重：从 5.0 降到 0.5
                     iou_init=5.0,    iou_min=1.0,    # IoU权重：从 5.0 降到 1.0
                     decay_epochs=8                   # 前 8 个 Epoch 线性衰减
                 ),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.loss_cfg = loss_cfg
        
        # --- 网络层定义 (保持不变) ---
        self.roi_align = RoIAlign(
            output_size=(roi_size, roi_size),
            spatial_scale=1.0/32.0, 
            sampling_ratio=0, aligned=True)
        
        self.vis_project = ConvModule(in_channels, feat_channels, 1, norm_cfg=dict(type='BN'))
        
        self.vis_fc = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(feat_channels * roi_size * roi_size, 1024),
            nn.ReLU(inplace=True))
        
        self.geo_fc = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(inplace=True))
        
        self.refine_head = nn.Sequential(
            nn.Linear(1024 + 64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4))
        
        self.iou_loss_func = IoULoss(loss_type='diou')

    def _get_current_weights(self):
        """
        核心逻辑：根据当前 Epoch 动态计算权重
        使用线性衰减策略: Weight = Init - (Init - Min) * (Current / Decay_Max)
        """
        # 获取消息中心，从中读取当前 epoch
        message_hub = MessageHub.get_current_instance()
        # 注意：MMDet 的 epoch 通常从 0 或 1 开始，这里要做个安全获取
        current_epoch = message_hub.get_info('epoch') if message_hub is not None else 0
        
        # 如果获取不到 epoch (比如刚初始化)，默认使用初始权重
        if current_epoch is None:
            current_epoch = 0

        # 读取配置
        decay_steps = self.loss_cfg.get('decay_epochs', 8)
        
        # 计算进度 (0.0 ~ 1.0)
        progress = min(current_epoch / decay_steps, 1.0)
        
        # 计算 Center Loss 权重
        c_init = self.loss_cfg['center_init']
        c_min = self.loss_cfg['center_min']
        w_center = c_init - (c_init - c_min) * progress
        
        # 计算 IoU Loss 权重
        i_init = self.loss_cfg['iou_init']
        i_min = self.loss_cfg['iou_min']
        w_iou = i_init - (i_init - i_min) * progress
        
        return w_center, w_iou

    def loss(self, x, cluster_proposals, batch_data_samples):
        # 1. 获取当前动态权重
        w_center, w_iou = self._get_current_weights()
        
        # 假设用最后一层特征 (C5)
        c5_feat = x[-1]
        batch_gt_instances = [ds.gt_instances for ds in batch_data_samples]
        batch_img_metas = [ds.metainfo for ds in batch_data_samples]
        
        all_initial_rois = []
        all_target_bboxes = []
        
        # --- A. 动态匹配与 RoI 构造 ---
        for i, (proposals, gt_instances) in enumerate(zip(cluster_proposals, batch_gt_instances)):
            initial_bboxes = proposals.bboxes
            gt_bboxes = gt_instances.bboxes
            
            if len(initial_bboxes) == 0:
                continue

            pos_initial_bboxes, target_cluster_bboxes = self._get_targets(initial_bboxes, gt_bboxes)
            
            if len(pos_initial_bboxes) > 0:
                batch_inds = torch.full((len(pos_initial_bboxes), 1), i, device=c5_feat.device)
                rois = torch.cat([batch_inds, pos_initial_bboxes], dim=1)
                all_initial_rois.append(rois)
                all_target_bboxes.append(target_cluster_bboxes)
        
        if len(all_initial_rois) == 0:
            return {
                'loss_refine_center': c5_feat.sum() * 0,
                'loss_refine_iou': c5_feat.sum() * 0
            }
            
        all_initial_rois = torch.cat(all_initial_rois, dim=0)
        all_target_bboxes = torch.cat(all_target_bboxes, dim=0)
        
        # --- B. 前向传播 ---
        deltas = self.forward(c5_feat, all_initial_rois, batch_img_metas)
        
        # --- C. 计算具体 Loss (传入动态权重) ---
        return self._loss_by_deltas(deltas, all_initial_rois, all_target_bboxes, w_center, w_iou)

    def _loss_by_deltas(self, deltas, initial_rois, gt_cluster_bboxes, w_center, w_iou):
        """具体的 Loss 计算"""
        losses = dict()
        refined_rois = self.apply_deltas(initial_rois, deltas)
        refined_bboxes = refined_rois[:, 1:]
        initial_bboxes = initial_rois[:, 1:]
        
        # Center Loss
        w_init = initial_bboxes[:, 2] - initial_bboxes[:, 0]
        h_init = initial_bboxes[:, 3] - initial_bboxes[:, 1]
        c_init_x = (initial_bboxes[:, 0] + initial_bboxes[:, 2]) / 2
        c_init_y = (initial_bboxes[:, 1] + initial_bboxes[:, 3]) / 2
        c_ref_x = (refined_bboxes[:, 0] + refined_bboxes[:, 2]) / 2
        c_ref_y = (refined_bboxes[:, 1] + refined_bboxes[:, 3]) / 2
        
        diff_x = (c_ref_x - c_init_x) / (w_init + 1e-6)
        diff_y = (c_ref_y - c_init_y) / (h_init + 1e-6)
        
        loss_center = (F.smooth_l1_loss(diff_x, torch.zeros_like(diff_x)) + 
                       F.smooth_l1_loss(diff_y, torch.zeros_like(diff_y)))
        
        # IoU Loss
        loss_iou = self.iou_loss_func(refined_bboxes, gt_cluster_bboxes)
        
        # 应用动态权重
        losses['loss_refine_center'] = loss_center * w_center
        losses['loss_refine_iou'] = loss_iou * w_iou
        
        return losses

    # ... (forward, apply_deltas, _get_targets 保持不变) ...
    def forward(self, c5_feat, rois, img_metas):
        roi_feats = self.roi_align(c5_feat, rois)
        roi_feats = self.vis_project(roi_feats)
        vis_vec = self.vis_fc(roi_feats)
        
        batch_inds = rois[:, 0].long()
        img_shapes = torch.stack([
            torch.tensor(m['img_shape'][:2], device=rois.device) for m in img_metas
        ])[batch_inds]
        img_h, img_w = img_shapes[:, 0], img_shapes[:, 1]
        x1, y1, x2, y2 = rois[:, 1], rois[:, 2], rois[:, 3], rois[:, 4]
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        area = w * h
        
        geo_input = torch.stack([
            cx/img_w, cy/img_h, w/img_w, h/img_h, torch.log(area+1e-6)/10.0
        ], dim=1)
        geo_vec = self.geo_fc(geo_input)
        
        fusion_vec = torch.cat([vis_vec, geo_vec], dim=1)
        return self.refine_head(fusion_vec)

    def apply_deltas(self, rois, deltas):
        batch_ind = rois[:, 0:1]
        x1, y1, x2, y2 = rois[:, 1], rois[:, 2], rois[:, 3], rois[:, 4]
        w, h = x2 - x1, y2 - y1
        dx1, dy1, dx2, dy2 = deltas.unbind(dim=1)
        nx1 = x1 + dx1 * w; ny1 = y1 + dy1 * h
        nx2 = x2 + dx2 * w; ny2 = y2 + dy2 * h
        return torch.cat([batch_ind, nx1.unsqueeze(1), ny1.unsqueeze(1), nx2.unsqueeze(1), ny2.unsqueeze(1)], dim=1)

    def _get_targets(self, initial_bboxes, gt_bboxes, iou_thr=0.1):
        ious = bbox_overlaps(initial_bboxes, gt_bboxes)
        pos_list, target_list = [], []
        for i in range(initial_bboxes.shape[0]):
            matched = torch.where(ious[i] > iou_thr)[0]
            if len(matched) > 0:
                gts = gt_bboxes[matched]
                target_box = torch.stack([gts[:,0].min(), gts[:,1].min(), gts[:,2].max(), gts[:,3].max()])
                pos_list.append(initial_bboxes[i]); target_list.append(target_box)
        if len(pos_list) > 0: return torch.stack(pos_list), torch.stack(target_list)
        return initial_bboxes.new_zeros((0,4)), initial_bboxes.new_zeros((0,4))