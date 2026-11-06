# AdaScope-TOOD

`AdaScope_TOOD.py` keeps the AdaScope detector, cluster head and ARFR refiner,
but replaces the global FCOS head with a TOOD head.

## Files

```text
configs/detection/cluster/AdaScope_TOOD.py
```

No new detector/refiner class is needed: `AdaScopeDetector` accepts
`global_bbox_head` and uses it for the global branch. The config supplies
`TOODHead` and its task-alignment training configuration.

## Train

```shell
CUDA_VISIBLE_DEVICES=0 python tools/train_det.py \
  configs/detection/cluster/AdaScope_TOOD.py
```

Default: 20 epochs, SGD lr 5e-4, MultiStepLR milestones 12/18.

## Test

```shell
CUDA_VISIBLE_DEVICES=0 python tools/test_det.py \
  configs/detection/cluster/AdaScope_TOOD.py \
  work_dirs/adascope_tood_densesirst/best_merged_voc_mAP_epoch_XX.pth
```

The stage hook resolves the test stage from the checkpoint epoch, so the
trained refiner stage is used during evaluation.

## Metrics

The default evaluator is VOC 11-point `merged_voc/mAP` on DenseSIRST.
