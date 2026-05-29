import functools
from warnings import filters
import torch.nn as nn
import torch
import basicsr.archs.arch_util_fmnet as arch_util
import torch.nn.functional as F
from basicsr.archs.arch_util_fmnet import initialize_weights
import math
from basicsr.utils.registry import ARCH_REGISTRY

class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class FMBlock(nn.Module):
    def __init__(self, nf=64, opt=None):
        super(FMBlock, self).__init__()
        self.opt = opt

        self.down = nn.Conv2d(nf, 16, 1, 1, 0, bias=True)
        self.conv1_f = nn.Conv2d(16, 2 * 4, 3, 1, 1, bias=True)
        self.conv1_w = nn.Conv2d(16, 4, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(16, 16, 3, 1, 1, bias=True)
        self.up = nn.Conv2d(16, nf, 1, 1, 0, bias=True)
        
        self.pi = 3.1415927410125732
        p = torch.arange(0, 3, 1)
        q = torch.arange(0, 3, 1)
        p, q = torch.meshgrid(p, q)
        self.p = ((p + 0.5)* self.pi / 3).cuda().view(1, 1, 1, 3, 3)
        self.q = ((q + 0.5)* self.pi / 3).cuda().view(1, 1, 1, 3, 3)

        # initialization
        initialize_weights([self.down, self.conv1_f, self.conv1_w, self.conv2, self.up], 0.1)

    def forward(self, x):
        N, C, H, W = x.shape
        K = 4

        identity = x
        
        x = self.down(x)
        frequency = self.conv1_f(x) # N C H W -> N 2*K H W 
        weight = self.conv1_w(x) # N C H W -> N K H W 
        frequency = torch.sigmoid(frequency)
        weight = torch.nn.functional.softmax(weight, dim=1)
        frequency = frequency * (3 - 1)
        frequency = frequency.permute(0, 2, 3, 1).contiguous().view(N, H * W, 2, K) # N 2*K H W -> N H*W 2 K
        weight = weight.permute(0, 2, 3, 1).contiguous().view(N, H * W, 1, K) # N K H W -> N H*W 1 K
        hFrequency = frequency[:, :, 0, :].view(N , H * W, K, 1, 1).expand([-1, -1, -1, 3, 3])
        wFrequency = frequency[:, :, 1, :].view(N , H * W, K, 1, 1).expand([-1, -1, -1, 3, 3])
        p = self.p.expand([N, H * W, K, -1, -1])
        q = self.q.expand([N, H * W, K, -1, -1])
        
        kernel = torch.cos(hFrequency * p) * torch.cos(wFrequency * q)
        kernel = kernel.view(N, H * W, K, 3 ** 2)
        kernel = torch.matmul(weight, kernel)
        kernel = kernel.view(N, H * W, 3 ** 2, 1)
        # N H*W K**2 1

        v = torch.nn.functional.unfold(x, kernel_size=3, padding=int((3 - 1) / 2), stride=1) # N C H W -> N C*(K**2) H*W
        v = v.view(N, 16, 3 ** 2, H * W) # N C K**2 H*W
        v = v.permute(0, 3, 1, 2).contiguous() # N H*W C K**2

        z = torch.matmul(v, kernel) # N H*W C 1
        z = z.squeeze(-1).view(N, H, W, 16).permute(0, 3, 1, 2) # N H*W C -> N C H W
        z = self.conv2(z)
        out = self.up(z)
        
        return identity + out

    def build_filter(self, kernelSize):
        filters = torch.zeros((kernelSize, kernelSize, kernelSize, kernelSize))
        for i in range(kernelSize):
            for j in range(kernelSize):
                for h in range(kernelSize):
                    for w in range(kernelSize):
                        filters[i, j, h, w] = math.cos(math.pi * i * (h + 0.5) / kernelSize) * math.cos(math.pi * j * (w + 0.5) / kernelSize)
        return filters.view(kernelSize ** 2, kernelSize, kernelSize).cuda()

@ARCH_REGISTRY.register()
class FMNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, act_type='relu', FM_blockNumber=1, opt=None):
        super(FMNet, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 2, 1, bias=True)

        fm_block = functools.partial(FMBlock, nf=nf, opt=opt)
        if FM_blockNumber == 0:
            self.recon_trunk_fm = nn.Identity()
        else:
            self.recon_trunk_fm = arch_util.make_layer(fm_block, 1)

        res_block = functools.partial(ResidualBlock_noBN, nf=nf)
        if nb - FM_blockNumber == 0:
            self.recon_trunk_res = nn.Identity()
        else:
            self.recon_trunk_res = arch_util.make_layer(res_block, nb - 1)

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
        initialize_weights([self.conv_first, self.upconv, self.HRconv, self.conv_last], 0.1)

    def forward(self, x):
        fea = self.act(self.conv_first(x))
        out = self.recon_trunk_res(fea)
        out = self.recon_trunk_fm(out)
        out = self.act(self.upsampler(self.upconv(out)))
        out = self.conv_last(self.act(self.HRconv(out)))
        return out