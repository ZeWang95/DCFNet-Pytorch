'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import sys
sys.path.append('/home/jacobwang/DCF/DCFNet')
from DCF import *
import pdb

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_DCF(nn.Module):
    def __init__(self, vgg_name, bases_grad, kernel_size=3, num_class=10):
        super(VGG_DCF, self).__init__()
        self.kernel_size = kernel_size
        self.bases_grad = bases_grad
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_class)
        
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        padding = int((self.kernel_size - 1) / 2)
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [Conv_DCF(in_channels, x, kernel_size=self.kernel_size, padding=padding, 
                    num_bases=-1, bias=False, bases_grad=self.bases_grad),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers += [nn.AdaptiveAvgPool2d(1)]
        return nn.Sequential(*layers)

def test():
    net = VGG_DCF('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

if __name__ == '__main__':
    test()
