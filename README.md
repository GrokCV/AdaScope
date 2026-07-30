<a id="top"></a>

# AdaScope

This repository is the official implementation of **AdaScope**.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
  - [Step 1: Clone the Repository](#step-1-clone-the-repository)
  - [Step 2: Create a Conda Environment](#step-2-create-a-conda-environment)
  - [Step 3: Install PyTorch](#step-3-install-pytorch)
  - [Step 4: Install OpenMMLab Codebases](#step-4-install-openmmlab-codebases)
  - [Step 5: Install deepir](#step-5-install-deepir)
  - [Step 6: Cluster Generation](#step-6-cluster-generation)
- [Training](#training)
  - [Single-GPU Training](#single-gpu-training)
  - [Multi-GPU Training](#multi-gpu-training)
- [Testing](#testing)
- [Visualization](#visualization)

---

<a id="overview"></a>

## Overview

![AdaScope Pipeline](./docs/pipeline.png)

![AdaScope ARFR](./docs/ARFR.png)

[Back to top](#top)

---

<a id="installation"></a>

## Installation

<a id="step-1-clone-the-repository"></a>

### Step 1: Clone the Repository

Clone the repository and enter its root directory:

```shell
git clone git@github.com:GrokCV/AdaScope.git
cd AdaScope
```

> Make sure all subsequent commands are executed from the root directory of
> the AdaScope repository.

<a id="step-2-create-a-conda-environment"></a>

### Step 2: Create a Conda Environment

```shell
conda create --name deepir python=3.9 -y
conda activate deepir
```

<a id="step-3-install-pytorch"></a>

### Step 3: Install PyTorch

Install PyTorch with CUDA 11.8:

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

pip install "mmsegmentation>=1.0.0"
pip install dadaptation
```

<a id="step-5-install-deepir"></a>

### Step 5: Install deepir

Install the project in development mode:

```shell
python setup.py develop
```

Alternatively, you can use:

```shell
pip install -e .
```

<a id="step-6-cluster-generation"></a>

### Step 6: Cluster Generation

Before training AdaScope, generate the required cluster information according
to the dataset configuration.The first command is to generate Cluster-only lable ,the second one is to additionally lable the single instance as a 'cluster',meeting the definition of our new task,run the first one before the second one

```shell
/tools/relabel_test_clusters_graph.py

/tools/build_cluster2_from_cluster1.py
```



[Back to top](#top)

---

<a id="training"></a>

## Training

<a id="single-gpu-training"></a>

### Single-GPU Training

Run the following command:

```shell
CUDA_VISIBLE_DEVICES=0 python tools/train_det.py <CONFIG_FILE>
```

Example:

```shell
CUDA_VISIBLE_DEVICES=0 \
python tools/train_det.py configs/detection/cluster/adascope.py
```

<a id="multi-gpu-training"></a>

### Multi-GPU Training

Run the following command:

```shell
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 tools/train_det.py <CONFIG_FILE>
```

Example:

```shell
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 \
tools/train_det.py configs/detection/cluster/adascope.py
```

[Back to top](#top)

---

<a id="testing"></a>

## Testing

Run the following command:

```shell
CUDA_VISIBLE_DEVICES=0 \
python tools/test_det.py <CONFIG_FILE> <CHECKPOINT_FILE>
```

Example:

```shell
CUDA_VISIBLE_DEVICES=0 \
python tools/test_det.py \
configs/detection/cluster/adascope.py \
work_dirs/adascope/20260719_162542/best_pascal_voc_mAP_epoch_8.pth
```

[Back to top](#top)

---

<a id="visualization"></a>

## Visualization

To visualize the test results directly, add the `--show` option:

```shell
CUDA_VISIBLE_DEVICES=0 \
python tools/test_det.py \
configs/detection/cluster/adascope.py \
work_dirs/adascope/20260719_162542/best_pascal_voc_mAP_epoch_8.pth \
--show
```

You can use `--work-dir` to specify the test log directory:

```shell
CUDA_VISIBLE_DEVICES=0 \
python tools/test_det.py \
configs/detection/cluster/adascope.py \
<CHECKPOINT_FILE> \
--work-dir work_dirs/adascope_test
```

You can also use `--show-dir` to specify the directory where visualization
results will be saved:

```shell
CUDA_VISIBLE_DEVICES=0 \
python tools/test_det.py \
configs/detection/cluster/adascope.py \
<CHECKPOINT_FILE> \
--show-dir work_dirs/adascope_visualization
```

### Test Arguments

| Argument | Description |
| --- | --- |
| `--show` | Display visualization results directly. |
| `--work-dir` | Specify the test log and output directory. |
| `--show-dir` | Specify the directory used to save visualization results. |

[Back to top](#top)