import torch
from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS
# 我们需要从 sklearn 导入更多工具
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error, confusion_matrix, accuracy_score
import numpy as np

@METRICS.register_module()
class MyGridMetric(BaseMetric):
    default_prefix = 'MyGridMetric' 

    def __init__(self, cls_threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.cls_threshold = cls_threshold 

    def process(self, data_batch, data_samples):
        # data_samples 是一个 List[dict]

        # --- 1. 使用字典的 [] 访问 ---
        preds = [d['pred_instances'] for d in data_samples]
        gts = [d['gt_instances'] for d in data_samples]
        # --- 修改结束 ---
        
        for pred, gt in zip(preds, gts):
            # --- 2. 这里的 pred 和 gt 也是 dict, 也要用 [] 访问 ---
            self.results.append({
                'pred_cls': pred['cls_heatmap'].squeeze().cpu().numpy(),
                'gt_cls': gt['gt_cls_map'].cpu().numpy(),
                
            })

    def compute_metrics(self, results: list) -> dict:
        all_pred_cls = []
        all_gt_cls = []
     
        
       

        # --- 1. 汇总所有数据 ---
        for res in results:
           
            
            all_pred_cls.append(res['pred_cls'].flatten())
            all_gt_cls.append(res['gt_cls'].flatten())


        if not all_pred_cls:
            return {}

        # --- 2. 计算分类指标 (二分类精度) ---
        all_pred_cls_flat = np.concatenate(all_pred_cls)
        all_gt_cls_flat = np.concatenate(all_gt_cls)
        pred_cls_binary = (all_pred_cls_flat > self.cls_threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_gt_cls_flat, pred_cls_binary, average='binary', zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(all_gt_cls_flat, pred_cls_binary).ravel()
        
        metrics = {
            'cls_precision': precision,
            'cls_recall': recall,
            'cls_f1_score': f1,
            'cls_TP': float(tp),
            'cls_FP': float(fp),
            'cls_FN': float(fn),
        }

       


        return metrics