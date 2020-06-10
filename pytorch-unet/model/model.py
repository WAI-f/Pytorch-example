import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class UNetModel(BaseModel):
    def __init__(self, in_channels, n_labels):
        super(UNetModel, self).__init__()
        self.conv1 = DoubleConv(in_channels, 32)
        self.conv2 = DoubleConv(32, 64)
        self.conv3 = DoubleConv(64, 128)
        self.conv4 = DoubleConv(128, 256)
        self.conv5 = DoubleConv(256, 512)

        self.up6 = Up(512, 256)
        self.up7 = Up(256, 128)
        self.up8 = Up(128, 64)
        self.up9 = Up(64, 32)

        self.conv10 = nn.Conv2d(32, n_labels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        pool1 = F.max_pool2d(x1, 2)
        x2 = self.conv2(pool1)
        pool2 = F.max_pool2d(x2, 2)
        x3 = self.conv3(pool2)
        pool3 = F.max_pool2d(x3, 2)
        x4 = self.conv4(pool3)
        pool4 = F.max_pool2d(x4, 2)

        x = self.conv5(pool4)
        x = self.up6(x, x4)
        x = self.up7(x, x3)
        x = self.up8(x, x2)
        x = self.up9(x, x1)

        output = self.conv10(x)
        return output
