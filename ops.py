import functools

import torch.nn as nn


def conv_bn_lrelu(in_dim, out_dim, kernel_size, stride, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.LeakyReLU(negative_slope=0.2))

def conv_bn_relu(in_dim, out_dim, kernel_size, stride, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU())

def dconv_bn_relu(in_dim, out_dim, kernel_size, stride, padding=0,
                   output_padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride,
                           padding, output_padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU())


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.res = nn.Sequential(nn.ReflectionPad2d(1),
                                conv_bn_relu(in_dim, out_dim, 3, 1),
                                nn.ReflectionPad2d(1),
                                nn.Conv2d(out_dim, out_dim, 3, 1),
                                nn.BatchNorm2d(out_dim))

    def forward(self, x):
        return x + self.res(x)
