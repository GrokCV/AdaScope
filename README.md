# AdaScope



This repository is the official implementation of AdaScope

- [AdaScope](#adascope)
    - [Installation](#installation)
        - [Step 1: Create a conda environment](#step-1-create-a-conda-environment)
        - [Step 2: Install PyTorch](#step-2-install-pytorch)
        - [Step 3: Install OpenMMLab Codebases](#step-3-install-openmmlab-codebases)
        - [Step 4: Install `deepir`](#step-4-install-deepir)
    - [Train](#train)
    - [Test](#test)





## AdaScope


![AdaScope](./docs/pipeline.png)

![AdaScope](./docs/ARFR.png)

### Installation

Step 1: Create a conda environment

```shell
$ conda create --name deepir python=3.9
$ conda activate deepir
```

Step 2: Install PyTorch

```shell
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Step 3: Install OpenMMLab Codebases

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"
pip install "mmsegmentation>=1.0.0"
pip install dadaptation
```

Step 4: Install `deepir`

```shell
$ python setup.py develop
```

**Note**: make sure you have `cd` to the root directory of `deepinfrared`

```shell
$ git clone git@github.com:GrokCV/GrokDet/AdaScope.git
$ cd AdaScope
```



### Train

**Single GPU Training**

```shell
$ CUDA_VISIBLE_DEVICES=0 python train.py <CONFIG_FILE>
```

For example:

```shell
$ CUDA_VISIBLE_DEVICES=0 python tools/train_det.py configs/detection/cluster/adascope.py
```

**Multi GPU Training**

```shell
$ CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py <CONFIG_FILE>
```

For example:

```shell
$ CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 ./tools/train_det.py configs/detection/cluster/adascope.py
```

### Test

```shell
$ CUDA_VISIBLE_DEVICES=0 python test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE>
```

For example:

```shell
$ CUDA_VISIBLE_DEVICES=0 python tools/test_det.py configs/detection/cluster/adascope.py work_dirs/adascope/20260719_162542/best_pascal_voc_mAP_epoch_8.pth
```

If you want to visualize the result, you only add ```--show``` at the end of the above command.

The default image save path is under <SEG_CHECKPOINT_FILE>. You can use `--work-dir` to specify the test log path, and the image save path is under this path by default. Of course, you can also use `--show-dir` to specify the image save path.
