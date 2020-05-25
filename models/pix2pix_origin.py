import torch
from torch import nn

class ConvUnit(nn.Module):
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


class UpConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.unit = nn.Sequential(
            nn.ReLU(True),
            # UNet 上卷积部分由两个部分沿 channels 维度叠加而成
            nn.ConvTranspose2d(in_channels * 2, out_channels, kernel_size, stride=2, padding=padding),
            nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True)
        )

    def forward(self, data):
        return self.unit(data)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义 UNet Generator 的卷积层
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=2, padding=67)
        )  # Same Padding
        self.down2 = ConvUnit(32, 64, kernel_size=8, stride=2, padding=67)  # Same Padding
        self.down3 = ConvUnit(64, 128)
        self.down4 = ConvUnit(128, 256)
        self.down5 = ConvUnit(256, 512)
        self.down6 = ConvUnit(512, 1024)
        self.down7 = ConvUnit(1024, 1024)
        self.down8 = ConvUnit(1024, 1024)
        self.down_last = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 1024, 4, stride=2, padding=1),
        )

        # 定义 UNet Generator 的反卷积层
        self.up1 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 1024, 4, stride=2, padding=1),
            nn.BatchNorm2d(1024, affine=True, track_running_stats=True)
        )
        self.up2 = UpConvUnit(1024, 1024)
        self.up3 = UpConvUnit(1024, 1024)
        self.up4 = UpConvUnit(1024, 512)
        self.up5 = UpConvUnit(512, 256)
        self.up6 = UpConvUnit(256, 128)
        self.up7 = UpConvUnit(128, 64)
        self.up8 = UpConvUnit(64, 32, kernel_size=8, stride=2, padding=67)
        self.up_last = nn.Sequential(
            nn.ConvTranspose2d(32 * 2, 1, 8, stride=2, padding=67),
            nn.Tanh()
        )  # 最后一个单元同样需要 skip connection

    def forward(self, data):
        x1 = self.down1(data)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x8 = self.down8(x7)
        x9 = self.down_last(x8)  # 1x1x1024
        x = self.up1(x9)
        x = self.up2(torch.cat((x, x8), 1))
        x = self.up3(torch.cat((x, x7), 1))
        x = self.up4(torch.cat((x, x6), 1))
        x = self.up5(torch.cat((x, x5), 1))
        x = self.up6(torch.cat((x, x4), 1))
        x = self.up7(torch.cat((x, x3), 1))
        x = self.up8(torch.cat((x, x2), 1))
        output = self.up_last(torch.cat((x, x1), 1))
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        sequence = [nn.Conv2d(2, 64, 4, stride=2, padding=1)]
        sequence += [
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(512, 1, 4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, data):
        return self.model(data)