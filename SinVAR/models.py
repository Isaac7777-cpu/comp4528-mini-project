import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel: int, out_channel: int, ker_size: int, padd: int, stride: int):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                          kernel_size=ker_size, stride=stride, padding=padd))
        self.add_module('norm', nn.BatchNorm2d(out_channel))
