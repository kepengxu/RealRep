import torch.nn as nn
from .arch_util_hdrtvnet import ResidualBlock_noBN, make_layer, initialize_weights


def color_block(in_filters, out_filters, normalization=False):
    conv = nn.Conv2d(in_filters, out_filters, 1, stride=1, padding=0)
    pooling = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=True)
    act = nn.LeakyReLU(0.2)
    layers = [conv, pooling, act]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers


class Color_Condition(nn.Module):
    def __init__(self, in_channels=3, out_c=3):
        super(Color_Condition, self).__init__()

        self.model = nn.Sequential(
            *color_block(3, 16, normalization=True),
            *color_block(16, 32, normalization=True),
            *color_block(32, 64, normalization=True),
            *color_block(64, 128, normalization=True),
            *color_block(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, out_c, 1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, img_input):
        return self.model(img_input)


class ConditionNet(nn.Module):
    def __init__(self, nf=64, classifier='color_condition', cond_c=3):
        super(ConditionNet, self).__init__()

        if classifier=='color_condition':
            self.classifier = Color_Condition(out_c=cond_c)

        self.GFM_nf = 64

        self.cond_scale_first = nn.Linear(cond_c, nf)
        self.cond_scale_HR = nn.Linear(cond_c, nf)
        self.cond_scale_last = nn.Linear(cond_c, 3)

        self.cond_shift_first = nn.Linear(cond_c, nf)
        self.cond_shift_HR = nn.Linear(cond_c, nf)
        self.cond_shift_last = nn.Linear(cond_c, 3)

        self.conv_first = nn.Conv2d(3, nf, 1, 1)
        self.HRconv = nn.Conv2d(nf, nf, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x # 输入
        condition = x # cond数据
        fea = self.classifier(condition).squeeze(2).squeeze(2)

        scale_first = self.cond_scale_first(fea)
        shift_first = self.cond_shift_first(fea)

        scale_HR = self.cond_scale_HR(fea)
        shift_HR = self.cond_shift_HR(fea)

        scale_last = self.cond_scale_last(fea)
        shift_last = self.cond_shift_last(fea)

        out = self.conv_first(content)
        out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.HRconv(out)
        out = out * scale_HR.view(-1, self.GFM_nf, 1, 1) + shift_HR.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv_last(out)
        out = out * scale_last.view(-1, 3, 1, 1) + shift_last.view(-1, 3, 1, 1) + out

        return out
    
import functools
class SRResNet(nn.Module):
    ''' modified SRResNet for ITM'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, act_type='relu'):
        super(SRResNet, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 2, 1, bias=True)
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.recon_trunk = make_layer(basic_block, nb)

        self.upconv = nn.Conv2d(nf, nf*4, 3, 1, 1, bias=True)
        self.upsampler = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        # initialization
        initialize_weights([self.conv_first, self.upconv, self.HRconv, self.conv_last],
                                     0.1)

    def forward(self, x):
        fea = self.act(self.conv_first(x))
        out = self.recon_trunk(fea)
        out = self.act(self.upsampler(self.upconv(out)))
        out = self.conv_last(self.act(self.HRconv(out)))
        return out
    
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class HDRTVNet(nn.Module):
    def __init__(self, 
                 nf=64, 
                 classifier='color_condition', 
                 cond_c=6, 
                 in_nc=3, 
                 out_nc=3, 
                 nb=16, 
                 act_type='relu'):
        super(HDRTVNet, self).__init__()
        
        # 初始化 ConditionNet
        self.condition_net = ConditionNet(nf=nf, 
                                          classifier=classifier, 
                                          cond_c=cond_c)
        
        # 初始化 SRResNet
        self.sr_resnet = SRResNet(in_nc=in_nc, 
                                  out_nc=out_nc, 
                                  nf=nf, 
                                  nb=nb, 
                                  act_type=act_type)
        
    def forward(self, x):
        x = self.condition_net(x)
        
        out = self.sr_resnet(x)
        
        return out