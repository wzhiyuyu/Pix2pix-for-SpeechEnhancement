import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class UNetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_channel, inner_channel, input_channel,
                 sub_module=None, inner_most=False, outer_most=False, drop_out=False):
        super(UNetSkipConnectionBlock, self).__init__()

        self.outer_most = outer_most

        if not input_channel:
            input_channel = outer_channel

        down_conv = nn.Conv2d(input_channel, inner_channel, kernel_size=4, stride=2, padding=1, bias=False)
        down_relu = nn.LeakyReLU(0.2, True)
        down_norm = nn.BatchNorm2d(inner_channel)

        up_relu = nn.ReLU(True)
        up_norm = nn.BatchNorm2d(outer_channel)

        if inner_most:
            """
            UNet 网络的最内侧无子网络
            """
            up_conv = nn.ConvTranspose2d(inner_channel, outer_channel, kernel_size=4, stride=2, padding=1, bias=False)
            down = [down_relu, down_conv]
            up = [up_relu, up_conv, up_norm]
            model = down + up
        elif outer_most:
            """
            UNet 最顶层的输入直接进行卷积
            最终的输出不进行 Normalization，直接在卷积后使用 Tanh()
            """
            up_conv = nn.ConvTranspose2d(inner_channel, outer_channel, kernel_size=4, stride=2, padding=1,
                                         bias=False)

            down = [down_conv]
            up = [up_relu, up_conv, nn.Tanh()]
            model = down + [sub_module] + up
        else:
            """
            UNet 中间层
            在过滤器数量不变时，上采样相关层可能会使用 drop_out
            """
            up_conv = nn.ConvTranspose2d(inner_channel, outer_channel, kernel_size=4, stride=2, padding=1,
                                         bias=False)
            up = [up_relu, up_conv, up_norm]
            down = [down_relu, down_conv, down_norm]

            if drop_out:
                model = down + [sub_module] + up + [nn.Dropout(0.5)]
            else:
                model = down + [sub_module] + up

        self.model = nn.Sequential(*model)

    def forward(self, ipt):
        return self.model(ipt)



class Generator(nn.Module):
    def __init__(self, n_filters_first=64, n_downsamplings_blocks=8):
        super(Generator, self).__init__()
        input_channel = 1

        # 定义最内层结构
        unet_block = UNetSkipConnectionBlock(
            outer_channel=n_filters_first * 8,
            inner_channel=n_filters_first * 8,
            input_channel=None,
            sub_module=None,
            inner_most=True,
            outer_most=False
        )
        # 无论是升采样还是降采样，均不改变网络的过滤器数量
        # 过滤器数量相同时，使用 drop_out
        for i in range(n_downsamplings_blocks - 5):
            unet_block = UNetSkipConnectionBlock(
                outer_channel=n_filters_first * 8,
                inner_channel=n_filters_first * 8,
                input_channel=None,
                sub_module=unet_block,
                inner_most=False,
                outer_most=False,
                drop_out=True
            )
        # 逐渐改变过滤器的数量
        # - 降采样：从 input_channel => n_filters_first * 8
        # - 升采样：从 n_filters_first * 8 => outer_channel
        unet_block = UNetSkipConnectionBlock(
            outer_channel=n_filters_first * 4,
            inner_channel=n_filters_first * 8,
            input_channel=None,
            sub_module=unet_block,
            inner_most=False,
            outer_most=False
        )
        unet_block = UNetSkipConnectionBlock(
            outer_channel=n_filters_first * 2,
            inner_channel=n_filters_first * 4,
            input_channel=None,
            sub_module=unet_block,
            inner_most=False,
            outer_most=False
        )
        unet_block = UNetSkipConnectionBlock(
            outer_channel=n_filters_first,
            inner_channel=n_filters_first * 2,
            input_channel=None,
            sub_module=unet_block,
            inner_most=False,
            outer_most=False
        )
        model = UNetSkipConnectionBlock(
            outer_channel=1,
            inner_channel=n_filters_first,
            input_channel=input_channel,
            sub_module=unet_block,
            inner_most=False,
            outer_most=True
        )

        self.model = model

    def forward(self, ipt):
        return self.model(ipt)

class Discriminator(nn.Module):
    def __init__(self, n_filters_first=64, use_bn_in_last_layer=True):
        super(Discriminator, self).__init__()
        network = [
            spectral_norm(nn.Conv2d(2, n_filters_first, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(n_filters_first, n_filters_first * 2, 4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(n_filters_first * 2),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(n_filters_first * 2, n_filters_first * 4, 4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(n_filters_first * 4),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        if use_bn_in_last_layer:
            network += [
                spectral_norm(nn.Conv2d(n_filters_first * 4, n_filters_first * 8, 4, stride=1, padding=1, bias=False))
            ]
        else:
            network += [nn.Conv2d(n_filters_first * 4, n_filters_first * 8, 4, stride=1, padding=1, bias=False)]

        network += [
            nn.BatchNorm2d(n_filters_first * 8),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        self.nn = nn.Sequential(*network)

    # forward method
    def forward(self, x):
        x = self.nn(x)
        return x
