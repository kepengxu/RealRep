import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbam import CBAM


class ConditionNet(nn.Module):
    def __init__(self, in_channels, cond_features=32, mode='luma'):
        super().__init__()

        if mode == 'luma':
            self.condition_net = LumaCondition(in_channels, cond_features)
        elif mode == 'chroma':
            self.condition_net = ChromaCondition(in_channels, cond_features)
        else:
            raise NotImplementedError

    def forward(self, x):
        cond = self.condition_net(x)

        return cond


class LumaCondition(nn.Module):
    def __init__(self, in_channels, n_features):
        super().__init__()

        self.condition = nn.Sequential(
            *csrnet_condition_block(in_channels, n_features, 7, stride=2, padding=1),
            *csrnet_condition_block(n_features, n_features, 3, stride=2, padding=1),
            *csrnet_condition_block(n_features, n_features, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        return self.condition(x)  # (N, n_features, 1, 1)


class ChromaCondition(nn.Module):
    def __init__(self, in_channels, n_features):
        super().__init__()

        self.condition = nn.Sequential(
            *hdrtvnet_color_condition_block(in_channels, 16, normalization=True),
            *hdrtvnet_color_condition_block(16, 32, normalization=True),
            *hdrtvnet_color_condition_block(32, 64, normalization=True),
            *hdrtvnet_color_condition_block(64, 128, normalization=True),
            *hdrtvnet_color_condition_block(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, n_features, 1),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.condition(x)  # (N, n_features, 1, 1)


'''
class CrossCondition(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
'''

'''
class SpatialCondition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.unet = UNet(in_channels, out_channels, init_mid_channels=16, bilinear=True)

    def forward(self, x):
        return self.unet(x)
'''


class ConditionedTransform(nn.Module):
    """
        (in_channels) -> Conv -> (n_features) -> Transform -> (n_features) -> Act -> (n_features)
                              (cond_channels) +|
    """

    def __init__(self,
                 in_channels,
                 n_features,
                 cond_channels,
                 transform='global',
                 ada_method='vanilla',
                 activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_features, kernel_size=1)
        if transform == 'global':
            self.transform = GlobalFeatureModulation(cond_channels, n_features)
        elif transform == 'spatial':
            self.transform = SpatialFeatureTransform(cond_channels, n_features, ada_method=ada_method)
        else:
            raise NotImplementedError
        self.act = nn.ReLU(inplace=True) if activation else None

    def forward(self, x, cond):
        out = self.conv(x)
        out = self.transform(out, cond)
        if self.act:
            out = self.act(out)

        return out


class GlobalFeatureModulation(nn.Module):
    def __init__(self, cond_channels, n_features, residual=True):
        super().__init__()
        self.cond_scale = nn.Linear(cond_channels, n_features)
        self.cond_shift = nn.Linear(cond_channels, n_features)
        self.n_features = n_features
        self.residual = residual

    def forward(self, x, cond):
        cond_vec = cond.squeeze(-1).squeeze(-1)  # (N, cond_channels)
        scale = self.cond_scale(cond_vec).view(-1, self.n_features, 1, 1)  # (N, n_features, 1, 1)
        shift = self.cond_shift(cond_vec).view(-1, self.n_features, 1, 1)  # (N, n_features, 1, 1)
        # print(x.shape, scale.shape, shift.shape)
        out = x * scale + shift

        if self.residual:
            return out + x
        else:
            return out


class SpatialFeatureTransform(nn.Module):
    def __init__(self, cond_channels, n_features, ada_method='vanilla', residual=True):
        super().__init__()
        if ada_method == 'vanilla':
            self.cond_scale = nn.Sequential(
                nn.Conv2d(cond_channels, n_features, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_features, n_features, 3, stride=1, padding=1)
            )
            self.cond_shift = nn.Sequential(
                nn.Conv2d(cond_channels, n_features, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_features, n_features, 3, stride=1, padding=1)
            )
        elif ada_method == 'cbam':
            self.cond_scale = nn.Sequential(
                nn.Conv2d(cond_channels, n_features, 1),
                nn.ReLU(inplace=True),
                CBAM(n_features)
            )
            self.cond_shift = nn.Sequential(
                nn.Conv2d(cond_channels, n_features, 1),
                nn.ReLU(inplace=True),
                CBAM(n_features)
            )

        self.residual = residual

    def forward(self, x, cond):
        scale = self.cond_scale(cond)  # (N, n_features, H, W)
        shift = self.cond_shift(cond)  # (N, n_features, H, W)
        out = x * scale + shift

        if self.residual:
            return out + x
        else:
            return out


def csrnet_condition_block(in_channels, n_features, kernel_size, stride, padding):
    conv = nn.Conv2d(in_channels, n_features, kernel_size, stride, padding)
    act = nn.ReLU(inplace=True)
    layers = [conv, act]
    return layers


def hdrtvnet_color_condition_block(in_channels, n_features, normalization=False):
    conv = nn.Conv2d(in_channels, n_features, kernel_size=1, stride=1, padding=0)
    pooling = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=True)
    act = nn.LeakyReLU(0.2)
    layers = [conv, pooling, act]
    if normalization:
        layers.append(nn.InstanceNorm2d(n_features, affine=True))
    return layers
