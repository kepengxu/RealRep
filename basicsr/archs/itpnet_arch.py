import torch.nn as nn
import torch

from basicsr.utils.ICTCP_convert import SDR_to_ICTCP, ICTCP_to_HDR


class Condition(nn.Module):
    def __init__(self, in_nc=3, nf=32):
        super(Condition, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_nc, nf, 7, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1_out = self.act(self.conv1(self.pad(x)))
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))
        conv3_out = self.act(self.conv3(self.pad(conv2_out)))
        out = torch.mean(conv3_out, dim=[2, 3], keepdim=False)
        return out


# 3layers with control
class CSRNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, base_nf=64, cond_nf=32):
        super(CSRNet, self).__init__()

        self.base_nf = base_nf
        self.out_nc = out_nc

        self.cond_net = Condition(in_nc=in_nc, nf=cond_nf)

        self.cond_scale1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.conv1 = nn.Conv2d(in_nc, base_nf, 1, 1, bias=True)

        self.cond_scale2 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift2 = nn.Linear(cond_nf, base_nf, bias=True)
        self.conv2 = nn.Conv2d(base_nf, base_nf, 1, 1, bias=True)

        self.cond_scale3 = nn.Linear(cond_nf, out_nc, bias=True)
        self.cond_shift3 = nn.Linear(cond_nf, out_nc, bias=True)
        self.conv3 = nn.Conv2d(base_nf, out_nc, 1, 1, bias=True)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        cond = self.cond_net(x)

        scale1 = self.cond_scale1(cond)
        shift1 = self.cond_shift1(cond)

        scale2 = self.cond_scale2(cond)
        shift2 = self.cond_shift2(cond)

        scale3 = self.cond_scale3(cond)
        shift3 = self.cond_shift3(cond)

        out = self.conv1(x)
        out = out * scale1.view(-1, self.base_nf, 1, 1) + shift1.view(-1, self.base_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv2(out)
        out = out * scale2.view(-1, self.base_nf, 1, 1) + shift2.view(-1, self.base_nf, 1, 1) + out
        out = self.act(out)

        out = self.conv3(out)
        out = out * scale3.view(-1, self.out_nc, 1, 1) + shift3.view(-1, self.out_nc, 1, 1) + out
        return out
    
"""Resblock_A"""
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class CA(nn.Module):
    def __init__(self, channel, reduction):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(self, channel, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True)):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(default_conv(channel, channel, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(channel))
            if i == 0: modules_body.append(act)
        modules_body.append(CA(channel, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RCABGroup(nn.Module):
    def __init__(self, channel):
        super(RCABGroup, self).__init__()
        kernel_size = 3
        reduction = 16
        n_resblocks = 2

        modules_body = []
        for _ in range(n_resblocks):
            modules_body.append(RCAB(channel, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True)))
        modules_body.append(default_conv(channel, channel, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res



"""Resblock_B"""
class CB1x1(nn.Module):
    def __init__(self, channels):
        super(CB1x1, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True))

    def forward(self, x):
        return self.residual_function(x) + x


class CB1x1_Group(nn.Module):
    def __init__(self, channels, num):
        super(CB1x1_Group, self).__init__()
        fusion_conv = [CB1x1(channels=channels) for _ in range(num)]
        self.fusion = nn.Sequential(*fusion_conv)

    def forward(self, x):
        return self.fusion(x) + x



"""Resblock_C"""
class CB3x3(nn.Module):
    def __init__(self, channels):
        super(CB3x3, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True))

    def forward(self, x):
        return self.residual_function(x) + x




class ITP_fusionNet(nn.Module):
    def __init__(self, channels):
        super(ITP_fusionNet, self).__init__()
        self.fusion_conv = CB1x1_Group(channels=channels * 2, num=4)
        self.I_split_conv = CB3x3(channels=channels)
        self.TP_split_conv = CB3x3(channels=channels)

    def forward(self, I, TP):
        ITP = torch.cat([I, TP], dim=1)
        ITP1 = self.fusion_conv(ITP)
        I1, TP1 = torch.split(ITP1, split_size_or_sections=32, dim=1)
        I2 = self.I_split_conv(I1)
        TP2 = self.TP_split_conv(TP1)
        return I2, TP2


class ITP_fusionNet_last(nn.Module):
    def __init__(self, channel):
        super(ITP_fusionNet_last, self).__init__()
        self.fusion_conv = CB1x1_Group(channels=channel * 2, num=4)
        self.last_conv = CB3x3(channels=channel * 2)

    def forward(self, I, TP):
        ITP = torch.cat([I, TP], dim=1)
        ITP1 = self.fusion_conv(ITP)
        ITP2 = self.last_conv(ITP1)
        I2, TP2 = torch.split(ITP2, split_size_or_sections=32, dim=1)
        return I2, TP2


class RCAB_fusion_head(nn.Module):
    def __init__(self, channel):
        super(RCAB_fusion_head, self).__init__()
        self.channel = channel
        self.RCAB_I = RCABGroup(channel=channel)
        self.RCAB_TP = RCABGroup(channel=channel)
        self.fusion_ITP = ITP_fusionNet(channels=channel)

    def forward(self, ITP):
        I, TP = torch.split(ITP, split_size_or_sections=self.channel, dim=1)
        I1 = self.RCAB_I(I)
        TP1 = self.RCAB_TP(TP)
        I2, TP2 = self.fusion_ITP(I1, TP1)
        I3 = I1 + I2
        TP3 = TP1 + TP2
        ITP3 = torch.cat([I3, TP3], dim=1)
        return ITP3

class RCAB_fusion_last(nn.Module):
    def __init__(self, channel):
        super(RCAB_fusion_last, self).__init__()
        self.channel = channel
        self.RCAB_I = RCABGroup(channel=channel)
        self.RCAB_TP = RCABGroup(channel=channel)
        self.fusion_ITP = ITP_fusionNet_last(channel=channel)

    def forward(self, ITP):
        I, TP = torch.split(ITP, split_size_or_sections=self.channel, dim=1)
        I1 = self.RCAB_I(I)
        TP1 = self.RCAB_TP(TP)
        I2, TP2 = self.fusion_ITP(I1, TP1)
        return I2, TP2



"""LCATNet"""
class fusionNet(nn.Module):
    def __init__(self, num=3, channel=32):
        super(fusionNet, self).__init__()
        self.head_I = default_conv(1, channel, 3)
        self.head_TP = default_conv(2, channel, 3)
        self.tail_I = default_conv(channel, 1, 3)
        self.tail_TP = default_conv(channel, 2, 3)
        body = []
        for _ in range(num):
            body.append(RCAB_fusion_head(channel))
        body.append(RCAB_fusion_last(channel))
        self.body = nn.Sequential(*body)

    def forward(self, sdrI, sdrTP):
        I1 = self.head_I(sdrI)
        TP1 = self.head_TP(sdrTP)

        ITP1 = torch.cat([I1, TP1], dim=1)  # 看这个
        I2, TP2 = self.body(ITP1)

        I3 = self.tail_I(I2)
        TP3 = self.tail_TP(TP2)

        return I3, TP3

from basicsr.utils.registry import ARCH_REGISTRY
@ARCH_REGISTRY.register()
class ITPNetWholeNet(nn.Module):
    def __init__(self, num=3, channel=32):
        super(ITPNetWholeNet, self).__init__()
        self.CSRNet_I = CSRNet(in_nc=1, out_nc=1, base_nf=64, cond_nf=32)
        self.CSRNet_TP = CSRNet(in_nc=2, out_nc=2, base_nf=64, cond_nf=32)
        self.fusionNet = fusionNet(num=num, channel=channel)

    def forward(self, sdrRGB):
        sdrITP = SDR_to_ICTCP(sdrRGB,dim=1)
        sdrI, sdrT, sdrP = torch.split(sdrITP, 1, dim=1)
        sdrTP = torch.cat([sdrT, sdrP], dim=1)
        I_global = self.CSRNet_I(sdrI)
        TP_global= self.CSRNet_TP(sdrTP)
        ITP_global = torch.cat([I_global, TP_global], dim=1)
        RGB_global = ICTCP_to_HDR(ITP_global,dim=1)
        
        I_detail,TP_detail = self.fusionNet(sdrI, sdrTP)
        I_final = I_global + I_detail
        TP_final = TP_global + TP_detail
        ITP_final = torch.cat([I_final, TP_final], dim=1)
        RGB_final = ICTCP_to_HDR(ITP_final,dim=1)

        return [ITP_global,RGB_global], [ITP_final,RGB_final]






if __name__=="__main__":
    gpu_ids = [0]
    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
    model = ITPNetWholeNet(device)
    # d1 = torch.rand(1, 3, 480, 480)
    # o = model(d1)
    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
    print()




