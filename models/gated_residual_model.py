import torch
from torch import nn

from models.module import (
    DownSampleBlock,
    UpSampleBlock,
    ResidualBlock,
    gated_linear_layer,
)


class Generator(nn.Module):
    """
    Define Generator 2D CNN
    """

    def __init__(self):
        super(Generator, self).__init__()

        self.conv = nn.Conv2d(1, 128, 15, stride=1, padding=7)
        self.norm_conv = nn.InstanceNorm2d(64)
        self.gates = nn.Conv2d(1, 128, 15, stride=1, padding=7)
        self.norm_gates = nn.InstanceNorm2d(64)

        self.down_sample = nn.Sequential(
            DownSampleBlock(128, 256, 5, stride=2, padding=66),
            DownSampleBlock(256, 512, 5, stride=2, padding=66),
        )

        self.residual_blocks = nn.Sequential(
            ResidualBlock(512, 1024, 3, stride=1, padding=1),
            ResidualBlock(512, 1024, 3, stride=1, padding=1),
            ResidualBlock(512, 1024, 3, stride=1, padding=1),
            ResidualBlock(512, 1024, 3, stride=1, padding=1),
        )  # 最终 output 为 1024 // 2

        # TODO Add Pixel Shuffler
        self.up_sample = nn.Sequential(
            UpSampleBlock(512, 256, 5, stride=1, padding=2),
            UpSampleBlock(256, 128, 5, stride=1, padding=2),
        )

        self.up_conv = nn.Sequential(
            nn.Conv2d(128, 1, 15, stride=1, padding=7),
            nn.ReLU(True)
        )


    def forward(self, input):
        h1 = self.conv(input)
        h1_norm = self.norm_conv(h1)
        h1_gates = self.gates(input)
        h1_norm_gates = self.norm_gates(h1_gates)
        h1_glu = gated_linear_layer(h1_norm, h1_norm_gates)

        down = self.down_sample(h1_glu)

        residual = self.residual_blocks(down)

        up = self.up_sample(residual)

        return self.up_conv(up)  # [bs, 1, 128, 128]


class Discriminator(nn.Module):
    """
    Arch:
        k3x3, c128, s2x2
        Instance Norm
        GLU
        k3x3, c256, s2x2
        Instance Norm
        GLU
        k3x3, c512, s2x2
        Instance Norm
        GLU
        k3x3, c512, s2x2
        Instance Norm
        GLU
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Conv2d(2, 128, 3, stride=2, padding=65)
        self.norm_conv = nn.InstanceNorm2d(64)
        self.gates = nn.Conv2d(2, 128, 3, stride=2, padding=65)
        self.norm_gates = nn.InstanceNorm2d(64)

        self.down_sample = nn.Sequential(
            DownSampleBlock(128, 256, 3, stride=2, padding=65),
            DownSampleBlock(256, 512, 3, stride=2, padding=65),
        )

        self.full = nn.Sequential(nn.Linear(128 * 128 * 512, 1), nn.Sigmoid())

    def forward(self, input):
        h1 = self.conv(input)
        h1_norm = self.norm_conv(h1)
        h1_gates = self.gates(input)
        h1_norm_gates = self.norm_gates(h1_gates)
        h1_glu = gated_linear_layer(h1_norm, h1_norm_gates)

        down = self.down_sample(h1_glu)

        down = down.view(down.size()[0], -1)

        return self.full(down)
