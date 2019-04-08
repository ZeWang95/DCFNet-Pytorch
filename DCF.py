import datetime
import os, sys
import random
import argparse
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
from torch import nn

import torch.nn.functional as F
import pdb

import time

from torch.nn.parameter import Parameter
import math
from fb import *

class Conv_DCF(nn.Module):
    r"""Pytorch implementation for 2D DCF Convolution operation.
    Link to ICML paper:
    https://arxiv.org/pdf/1802.04145.pdf


    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, optional): Zero-padding added to both sides of
            the input. Default: 0
        num_bases (int, optional): Number of basis elements for decomposition.
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        mode (optional): Either `mode0` for two-conv or `mode1` for reconstruction + conv.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 * \text{padding}[0] - \text{dilation}[0]
                        * (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

              W_{out} = \left\lfloor\frac{W_{in}  + 2 * \text{padding}[1] - \text{dilation}[1]
                        * (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size, kernel_size)
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::
        
        >>> from DCF import *
        >>> m = Conv_DCF(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    """
    __constants__ = ['kernel_size', 'stride', 'padding', 'num_bases',
                     'bases_grad', 'mode']
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
        num_bases=-1, bias=True,  bases_grad=False, dilation=1, initializer='FB', mode='mode1'):
        super(Conv_DCF, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # self.edge = (kernel_size-1)/2
        self.stride = stride
        self.padding = padding
        self.kernel_list = {}
        self.num_bases = num_bases
        assert mode in ['mode0', 'mode1'], 'Only mode0 and mode1 are available at this moment.'
        self.mode = mode
        self.bases_grad = bases_grad
        self.dilation = dilation

        assert initializer in ['FB', 'random'], 'Initializer should be either FB or random, other methods are not implemented yet'

        if initializer == 'FB':
            if kernel_size % 2 == 0:
                raise Exception('Kernel size for FB initialization only supports odd number for now.')
            base_np, _, _ = calculate_FB_bases(int((kernel_size-1)/2))
            if num_bases > base_np.shape[1]:
                raise Exception('The maximum number of bases for kernel size = %d is %d' %(kernel_size, base_np.shape[1]))
            elif num_bases == -1:
                num_bases = base_np.shape[1]
            else:
                base_np = base_np[:, :num_bases]
            base_np = base_np.reshape(kernel_size, kernel_size, num_bases)
            base_np = np.array(np.expand_dims(base_np.transpose(2,0,1), 1), np.float32)

        else:
            if num_bases <= 0:
                raise Exception('Number of basis elements must be positive when initialized randomly.')
            base_np = np.random.randn(num_bases, 1, kernel_size, kernel_size)

        if bases_grad:
            self.bases = Parameter(torch.tensor(base_np), requires_grad=bases_grad)
            self.bases.data.normal_(0, 1.0)
            # self.bases.data.uniform_(-1, 1)
        else:
            self.register_buffer('bases', torch.tensor(base_np, requires_grad=False).float())

        self.weight = Parameter(torch.Tensor(
                out_channels, in_channels*num_bases, 1, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        if self.mode == 'mode1':
            self.weight.data = self.weight.data.view(out_channels*in_channels, num_bases)
            self.bases.data = self.bases.data.view(num_bases, kernel_size*kernel_size)
            self.forward = self.forward_mode1
        else:
            self.forward = self.forward_mode0

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        self.weight.data.normal_(0, stdv) #Normal works better, working on more robust initializations
        if self.bias is not None:
            # self.bias.data.uniform_(-stdv, stdv)
            self.bias.data.zero_()

    def forward_mode0(self, input):
        FE_SIZE = input.size()
        feature_list = []
        input = input.view(FE_SIZE[0]*FE_SIZE[1], 1, FE_SIZE[2], FE_SIZE[3])
        
        feature = F.conv2d(input, self.bases,
            None, self.stride, self.padding, dilation=self.dilation)
        
        feature = feature.view(
            FE_SIZE[0], FE_SIZE[1]*self.num_bases, 
            int((FE_SIZE[2]-self.kernel_size+2*self.padding)/self.stride+1), 
            int((FE_SIZE[3]-self.kernel_size+2*self.padding)/self.stride+1))

        feature_out = F.conv2d(feature, self.weight, self.bias, 1, 0)

        return feature_out

    def forward_mode1(self, input):
        rec_kernel = torch.mm(self.weight, self.bases).view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        feature = F.conv2d(input, rec_kernel,
            self.bias, self.stride, self.padding, dilation=self.dilation)
        
        return feature

    def extra_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}, num_bases={num_bases}' \
            ', bases_grad={bases_grad}, mode={mode}'.format(**self.__dict__)