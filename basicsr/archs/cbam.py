import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, r):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // r, in_channels, 1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        y1 = self.avg_pool(x)
        y1 = self.mlp(y1)

        y2 = self.max_pool(x)
        y2 = self.mlp(y2)

        y = self.act(y1 + y2)

        return x * y


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_mask, max_mask], dim=1)

        mask = self.layer(mask)
        return x * mask


class CBAM(nn.Module):
    def __init__(self, in_channels, r=16, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.cam = ChannelAttentionModule(in_channels, r)
        self.sam = SpatialAttentionModule(kernel_size)

    def forward(self, x):
        y = self.conv(x)
        y = self.cam(y)
        y = self.sam(y)

        return x + y
