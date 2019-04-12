DCFNet-Pytorch
============================
This repository is the pytorch implementation for ICML 2018 paper DCFNet: Deep Neural Network with Decomposed Convolutional Filters

Note that this web page is still under construction (as life should be).

For details please refer to [paper](https://arxiv.org/pdf/1802.04145.pdf).

#Image Classification
The classification_DCF folder contains code for training DCFNet for image classification.

Note that this repository does not intend to reproduce the exact same results reported in the original paper, since the baseline performances are higher in our settings. And a ~1.5% accuracy growth is witnessed when using filter decomposition. 

We will update more results soon.

#Acknowledgements

The classification code borrows heavily from [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).