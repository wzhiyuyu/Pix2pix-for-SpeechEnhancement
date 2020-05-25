from abc import ABC

import torch
from torch import nn


def gated_linear_layer(inputs, gates):
    # https://pytorch.org/docs/0.3.1/nn.html#torch.nn.functional.glu
    activation = inputs * torch.sigmoid(gates)
    return activation


def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2 * (padding)) / stride) + 1
    return (output)


class Conv2dLayer(nn.Module):
    """
    定义卷积单元，包含卷积层，Norm和激活函数
    """

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        # Default: [batch, in_channels, in_height, in_height] => [batch, out_channels, in_height/2, in_width/2]
        super().__init__()
        self.unit = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True)
        )

    def forward(self, data):
        return self.unit(data)


class DownSampleBlock(nn.Module):
    """
    Arch:
        k5x5, c256, s1x2
        Instance Norm
        GLU
        k5x5, c512, s1x2
        Intance Norm
        GLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=66):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm_conv = nn.InstanceNorm2d(out_channels)
        self.gates = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm_gates = nn.InstanceNorm2d(out_channels)

    def forward(self, input):
        h1 = self.conv(input)
        h1_norm = self.norm_conv(h1)

        h1_gates = self.gates(input)
        h1_norm_gates = self.norm_gates(h1_gates)

        return gated_linear_layer(h1_norm, h1_norm_gates)


class ResidualBlock(nn.Module):
    """
    Arch:
        k1x5, c1024, s1x1
        Instance Norm
        Gated Linear Unit
        k1x5, c512, s1x1
        Instance Norm
        Sum
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm_conv_1 = nn.InstanceNorm2d(out_channels)
        self.gates = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm_gates = nn.InstanceNorm2d(out_channels)

        self.conv_2 = nn.Conv2d(out_channels, out_channels // 2, kernel_size, stride=stride, padding=padding)
        self.norm_conv_2 = nn.InstanceNorm2d(out_channels)

    def forward(self, input):
        h1 = self.conv_1(input)
        h1_norm = self.norm_conv_1(h1)
        h1_gates = self.gates(input)
        h1_norm_gates = self.norm_gates(h1_gates)

        glu = gated_linear_layer(h1_norm, h1_norm_gates)

        h2 = self.conv_2(glu)
        h2_norm = self.norm_conv_2(h2)

        return input + h2_norm


class UpSampleBlock(nn.Module):
    """
    Arch (c1024 => c512):
        k5x5, c1024, s1x1
        Pixel Shuffler
        Instance Norm
        GLU
        k5x5, c512, s1x1
        Pixel shuffler
        Instance Norm
        GLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm_conv = nn.InstanceNorm2d(out_channels)
        self.gates = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm_gates = nn.InstanceNorm2d(out_channels)

    def forward(self, input):
        h1 = self.conv(input)
        # TODO h1 Pixel Shuffler
        h1_norm = self.norm_conv(h1)
        h1_gates = self.gates(input)
        # TODO h1_gates Pixel shuffler
        h1_norm_gates = self.norm_gates(h1_gates)

        return gated_linear_layer(h1_norm, h1_norm_gates)
