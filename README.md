<a id="top"></a>

# Beyond the Sparsity Assumption: Arbitrary-Distribution Infrared Small Target Detection

Official implementation of **AdaScope** — an asynchronous coarse-to-fine framework
designed for **arbitrary-distribution infrared small target detection (IRSTD)**.
AdaScope decouples target localization from context allocation via a
divide-and-conquer strategy, reformulating coarse localization as a **spatial
existence classification** problem and learning distribution-adaptive observation
windows through **reinforcement learning**.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
- [Repository Structure](#repository-structure)

---

<a id="overview"></a>

## Overview

![AdaScope Pipeline](./docs/pipeline.png)

AdaScope is an asynchronous coarse-to-fine detector for arbitrary-distribution
infrared small target detection: **GEDM** performs coarse localization by
classifying spatial existence on a semantic grid (instead of fitting unstable
cluster boundaries), and **ARFR** uses reinforcement learning (GRPO) to adaptively
refine the observation window of each candidate, with the downstream detector's
performance gain as the reward. Global and local predictions are fused by NMS.

[Back to top](#top)

---

<a id="installation"></a>

## Installation

<a id="step-1-clone-the-repository"></a>

### Step 1: Clone the Repository

```shell
git clone git@github.com:GrokCV/AdaScope.git
cd AdaScope
```

> All commands below must be run from the repository root.

<a id="step-2-create-a-conda-environment"></a>

### Step 2: Create a Conda Environment

```shell
conda create --name deepir python=3.9 -y
conda activate deepir
```

<a id="step-3-install-pytorch"></a>

### Step 3: Install PyTorch

Install PyTorch with CUDA 11.8 (or a version matching your driver):

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 \
  -c pytorch \
  -c nvidia
```

<a id="step-4-install-openmmlab-codebases"></a>

### Step 4: Install OpenMMLab Codebases

```shell
pip install -U openmim

mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"
```

<a id="step-5-install-adascope"></a>

### Step 5: Install AdaScope

```shell
python setup.py develop
# or
pip install -e .
```

[Back to top](#top)

---

<a id="dataset-preparation"></a>

## Dataset Preparation

AdaScope is developed on the **DenseSIRST** infrared small-target dataset.

### Step 1: Download DenseSIRST

Download the dataset and place it under `data/` so that the layout is:

```
data/DenseSIRST/SIRSTdevkit/
├── PNGImages/                    # infrared images (*.png)
├── Splits/
│   ├── trainval_v2.txt           # training split
│   └── test_v2.txt               # testing split
└── SIRST/
    ├── BBox/                     # single-instance annotations (*.xml)
    └── Cluster_relabel_graph_v2/ # cluster annotations (*_with_clusters.xml)
```

If your data lives elsewhere, override `DATA_ROOT` in
`configs/detection/_base_/datasets/adascope_densesirst.py`.

### Step 2: Generate Cluster Annotations

AdaScope trains its cluster head and refiner against **cluster-level** annotations
(a box that encloses a group of small targets). Generate them from the
single-instance XMLs with the provided tools — **run the first, then the second**:

```shell
# (1) Graph-based clustering of the test split → Cluster_relabel_graph_v1
python tools/relabel_test_clusters_graph.py

# (2) Supplement Cluster_relabel_graph_v1 with singleton clusters so that every
#     instance is covered → Cluster_relabel_graph_v2
python tools/build_cluster2_from_cluster1.py
```

> These two tools reproduce the `Cluster_relabel_graph_v2` directory referenced
> by the training config. See the header of each script for its arguments.

[Back to top](#top)

---

<a id="training"></a>

## Training

AdaScope is trained in three stages automatically via the
`SynWarmupSupGRPOStageHook`:

| Stage | Epochs | What is trained |
|-------|--------|-----------------|
| Pretrain | 1–12 | backbone / neck / GEDM cluster head / global head |
| Center offset SFT | 13–16 | ARFR (center alignment, supervised) |
| RL postraining4Scope | 17–28 | ARFR (context allocation, Scope Adjustment) |

<a id="single-gpu-training"></a>

### Single-GPU Training

```shell
CUDA_VISIBLE_DEVICES=0 python tools/train_det.py configs/detection/cluster/AdaScope.py
```

<a id="multi-gpu-training"></a>

### Multi-GPU Training

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 \
tools/train_det.py configs/detection/cluster/AdaScope.py
```

Checkpoints are saved to `work_dirs/adascope_fcos_densesirst/`, with the best one
tracked by `merged_voc/mAP`.

### RL Policy Variants

The ARFR policy optimizer is pluggable. All variants share the same
coarse-to-fine pipeline (GEDM + ARFR + local FCOS) through the common base
detector `AdaScopeRLDetector` — only the detector/refiner classes and the
policy-stage name differ:

| Config | Policy optimizer | 
|--------|------------------|
| `AdaScope.py` | GRPO (default) | 
| `AdaScope_ppo.py` | PPO | 
| `AdaScope_sac.py` | SAC | 
| `AdaScope_trpo.py` | TRPO | 

Train any variant with the same entry point:

```shell
CUDA_VISIBLE_DEVICES=0 python tools/train_det.py configs/detection/cluster/AdaScope_ppo.py
CUDA_VISIBLE_DEVICES=0 python tools/train_det.py configs/detection/cluster/AdaScope_sac.py
CUDA_VISIBLE_DEVICES=0 python tools/train_det.py configs/detection/cluster/AdaScope_trpo.py
```

Test with the corresponding config and checkpoint:

```shell
CUDA_VISIBLE_DEVICES=0 python tools/test_det.py \
  configs/detection/cluster/AdaScope_ppo.py \
  work_dirs/adascope_ppo_densesirst/best_merged_voc_mAP_epoch_XX.pth
```

[Back to top](#top)

---

<a id="testing"></a>

## Testing

```shell
CUDA_VISIBLE_DEVICES=0 python tools/test_det.py \
  configs/detection/cluster/AdaScope.py \
  /path/to/best_merged_voc_mAP_epoch_XX.pth
```

By default the evaluator reports the **merged** metric (global + local fusion,
mAP @ IoU=0.5, 11-point VOC), which is also used for best-checkpoint selection.
To additionally report per-branch metrics (global / raw_cluster / refined_cluster
/ local), set `SHOW_ALL_METRICS = True` in
`configs/detection/cluster/AdaScope.py`.

[Back to top](#top)

---

<a id="results"></a>

## Results

Comparison with state-of-the-art methods on **DenseSIRST** (mAP and Recall under
the VOC 11-point protocol at IoU=0.5; PR / F1 / F2 at a confidence threshold):

| Method | Backbone | FLOPs ↓ | Params ↓ | mAP50 ↑ | Recall50 ↑ | PR ↑ | F1 ↑ | F2 ↑ |
|--------|----------|---------|----------|---------|------------|------|------|------|
| **One-stage** |
| FCOS | ResNet50 | 50.291G | 32.113M | 0.257 | 0.315 | 0.1571 | 0.2023 | 0.2445 |
| VFNet | ResNet50 | 48.317G | 32.709M | 0.253 | 0.336 | 0.1820 | 0.2251 | 0.2624 |
| YOLOX | — | 8.578G | 8.968M | 0.210 | 0.341 | 0.1450 | 0.1882 | 0.2291 |
| TOOD | ResNet50 | 50.456G | 32.018M | 0.256 | 0.355 | 0.1750 | 0.2224 | 0.2655 |
| DyHead | ResNet50 | 27.866G | 38.890M | 0.249 | 0.335 | 0.1650 | 0.2090 | 0.2488 |
| DDOD | ResNet50 | 46.514G | 32.378M | 0.253 | 0.335 | 0.1620 | 0.2079 | 0.2504 |
| GFL | ResNet50 | 52.296G | 32.258M | 0.264 | 0.367 | 0.1850 | 0.2331 | 0.2762 |
| **Two-stage** |
| Cascade R-CNN | ResNet50 | 90.978G | 69.152M | 0.136 | 0.188 | 0.1150 | 0.1338 | 0.1484 |
| SABL | ResNet50 | 0.125T | 42.213M | 0.124 | 0.104 | 0.1050 | 0.0969 | 0.0926 |
| Dynamic R-CNN | ResNet50 | 63.179G | 41.348M | 0.184 | 0.235 | 0.1420 | 0.1678 | 0.1883 |
| **End2End**|
| Sparse R-CNN | ResNet50 | 45.274G | 0.106G | 0.183 | 0.572 | 0.0850 | 0.1444 | 0.2488 |
| DAB-DETR | ResNet50 | 28.939G | 43.702M | 0.005 | 0.054 | 0.0100 | 0.0160 | 0.0250 |
| DQ-DETR | — | 783.57G | 58.68M | 0.0149 | 0.154 | 0.0200 | 0.0345 | 0.0610 |
| EFLNet | — | 65.49G | 38.335M | 0.152 | 0.1349 | 0.1400 | 0.1263 | 0.1193 |
| PConv | — | 65.32G | 37.136M | 0.164 | 0.179 | 0.1500 | 0.1525 | 0.1540 |
| **Coarse2Fine**|
| DMNet | ResNet50 | 144.45G | 63.18M | 0.2239 | 0.1259 | 0.2400 | 0.1412 | 0.1132 |
| AdaZoom | ResNet50 | 49.96G | 31.887M | 0.052 | 0.348 | 0.0360 | 0.0554 | 0.0819 |
| YOLC | HRNet | 121.59G | 67.55M | 0.335 | 0.477 | 0.3416 | 0.3527 | 0.3597 |
| BAFE-Net | ResNet50 | 71.639G | 35.626M | 0.283 | 0.335 | **0.4275** | 0.3302 | 0.2905 |
| YOLD | HRNet | 122.62G | 67.61M | 0.171 | 0.371 | 0.3312 | 0.2172 | 0.1800 |
| **AdaScope (Ours)** | **ResNet50** | 89.34G | 49.73M | **0.402** | **0.681** | 0.3284 | **0.3683** | **0.3972** |

[Back to top](#top)

---

<a id="repository-structure"></a>

## Repository Structure

```
AdaScope/
├── configs/
│   └── detection/
│       ├── _base_/
│       │   ├── datasets/adascope_densesirst.py   # dataset + pipelines
│       │   ├── schedules/schedule_1x.py          # LR schedule
│       │   └── default_runtime.py                # runtime defaults
│       └── cluster/
│           ├── AdaScope.py                       # main experiment config (GRPO)
│           ├── AdaScope_ppo.py                   # PPO variant config
│           ├── AdaScope_sac.py                   # SAC variant config
│           ├── AdaScope_trpo.py                  # TRPO variant config
│           └── adascope_fcos_local.py            # external local FCOS
├── deepir/
│   ├── models/
│   │   ├── detectors/
│   │   │   ├── adascope_detector.py              # FixedFlatSyncGRPODetector (GRPO)
│   │   │   ├── adascope_rl_detector.py           # AdaScopeRLDetector (shared base)
│   │   │   ├── adascope_ppo_detector.py          # AdaScopePPODetector
│   │   │   ├── adascope_sac_detector.py          # AdaScopeSACDetector
│   │   │   └── adascope_trpo_detector.py         # AdaScopeTRPODetector
│   │   ├── refine/
│   │   │   ├── adascope_refiner.py               # ARFR (GRPO refiner)
│   │   │   └── adascope_{ppo,sac,trpo}_refiner.py# RL variant refiners
│   │   └── cluster_heads/adascope_cluster_head.py# GEDM (C5 existence classifier)
│   ├── datasets/
│   │   ├── sirst_voc_det.py                      # DenseSIRST dataset
│   │   └── transforms/adascope_cluster_targets.py# cluster GT generation
│   ├── engine/hooks/adascope_stage_hook.py       # shared 3-stage RL training hook
│   └── evaluation/metrics/selective_voc_metric.py# per-branch VOC metrics
├── tools/
│   ├── train_det.py / test_det.py                # training & testing entry
│   ├── relabel_test_clusters_graph.py            # cluster generation (v1)
│   └── build_cluster2_from_cluster1.py           # cluster generation (v2)
└── docs/
    ├── pipeline.png
    └── ARFR.png
```

[Back to top](#top)

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{adascope,
  title={Beyond the Sparsity Assumption: Arbitrary-Distribution Infrared Small Target Detection},
  author={Jingtang Chen, ZhuLiu, Mingjian Fu,Yimian Dai*},
  journal={},
  year={2026}
}
```

## License

This project is released under the Apache 2.0 license.
