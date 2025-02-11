# Alchemy-Data-Free-Adversarial-Training

This repo is the official Pytorch implementation for the paper Alchemy: Data-Free Adversarial Training.

_The current code is a temporary version. The complete and perfect code will be gradually modified and updated._

## Install

We suggest to install the dependencies using Anaconda or Miniconda.

```bash
conda env create -f requirements.yaml
```

## How to use Alchemy?

+ Step 1: Activate the python env.

```bash
conda activate DFAT
```

+ Step 2: Get the original model.

```bash
python train_from_scratch.py --model resnet18 --dataset cifar10 --gpu 0
```

+ Step 3: Data Free Adversarial Training

```bash
bash scripts/dfat_diff_attack.sh
```

## Reference

```
@inproceedings{10.1145/3658644.3670395,
author = {Bai, Yijie and Ma, Zhongming and Chen, Yanjiao and Deng, Jiangyi and Pang, Shengyuan and Liu, Yan and Xu, Wenyuan},
title = {Alchemy: Data-Free Adversarial Training},
year = {2024},
isbn = {9798400706363},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3658644.3670395},
doi = {10.1145/3658644.3670395},
booktitle = {Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security},
pages = {3808–3822},
numpages = {15},
keywords = {adversarial training, data-free, dataset reconstruction, robustness transferability},
location = {Salt Lake City, UT, USA},
series = {CCS '24}
}
```
