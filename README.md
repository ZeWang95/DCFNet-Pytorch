# DCFNet-Pytorch

This repository is the pytorch implementation for ICML 2018 paper DCFNet: Deep Neural Network with Decomposed Convolutional Filters

Note that this web page is still under construction (as life should be).

For details please refer to [paper](https://arxiv.org/pdf/1802.04145.pdf).

## Usage

Modify config.py first before running any experiments.

## Image Classification
The ImageClassification folder contains code for training DCFNet for image classification.

Note that this repository does not intend to reproduce the exactly the same results reported in the original paper, since the baseline performances are higher in our settings. And a ~1.5% accuracy growth is witnessed when using filter decomposition on CIFAR-10. 

We will update more results soon.

## Citing DCFNet

If you find this repo is helpful for your research, please kindly consider citing the paper as well.


```latex
@article{qiu2018dcfnet,
	title={{DCFNet}: Deep Neural Network with Decomposed Convolutional Filters},
	author={Qiu, Qiang and Cheng, Xiuyuan and Calderbank, Robert and Sapiro, Guillermo},
	journal={International Conference on Machine Learning},
	year={2018}
}
```

## Acknowledgements

The classification code borrows heavily from [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).