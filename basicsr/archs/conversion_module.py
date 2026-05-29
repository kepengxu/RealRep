import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


eps = 1e-6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RGB2ITP(nn.Module):
    def __init__(self, src='bt2100_pq', relative=False):
        super().__init__()
        if src in ['bt2100_pq', 'hdr']:
            self.rgb_to_ictcp = BT2100PQ2ICtCp()
        elif src in ['bt1886', 'sdr']:
            self.rgb_to_ictcp = BT1886RGB2ICtCp()
        else:
            raise ValueError
        
        self.ictcp_to_itp = ICtCp2ITP(relative=relative)
    
    def forward(self, x):
        x1 = self.rgb_to_ictcp(x)
        y = self.ictcp_to_itp(x1)

        return y


class BT2100PQ2ICtCp(nn.Module):
    def __init__(self):
        super().__init__()
        self.pq_to_display_light = PQ2DisplayLight(normalize=False)
        self.rgb_to_lms = RGB2LMS('bt2020')
        self.display_light_to_pq = DisplayLight2PQ(normalized=False)
        self.lms_to_ictcp = LMS2ICtCp()

    def forward(self, x):
        x0 = torch.clamp(x, eps, 1.0)
        # if torch.isnan(x0).any() or torch.isinf(x0).any():
        #     print('x0 has nan or inf.')
        x1 = self.pq_to_display_light(x0)
        x2 = self.rgb_to_lms(x1)
        x3 = self.display_light_to_pq(x2)
        y = self.lms_to_ictcp(x3)

        return y


class BT1886RGB2ICtCp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma_to_display_light = Gamma2DisplayLight(normalize=False, gamma=2.4, L_w=100, L_b=0.1)
        self.rgb_bt709_to_bt2020 = GamutConversion('bt709', 'bt2020')
        self.rgb_to_lms = RGB2LMS('bt2020')
        self.display_light_to_pq = DisplayLight2PQ(normalized=False)
        self.lms_to_ictcp = LMS2ICtCp()

    def forward(self, x):
        x0 = torch.clamp(x, eps, 1.0)
        x1 = self.gamma_to_display_light(x0)
        x2 = self.rgb_bt709_to_bt2020(x1)
        x3 = self.rgb_to_lms(x2)
        x4 = self.display_light_to_pq(x3)
        y = self.lms_to_ictcp(x4)

        return y


class PQ2DisplayLight(nn.Module):
    def __init__(self, normalize=False, l_max=10000):
        super().__init__()
        self.normalize = normalize
        self.l_max = l_max

        self.m1 = 0.1593017578125  # 2610 / 16384
        self.m2 = 78.84375  # 2523 / 4096 * 128
        self.c1 = 0.8359375  # 3424 / 4096
        self.c2 = 18.8515625  # 2413 / 4096 * 32
        self.c3 = 18.6875  # 2392 / 128

    def forward(self, x):
        x1 = x ** (1 / self.m2)
        x2 = torch.clamp(x1 - self.c1, eps)
        x3 = self.c2 - self.c3 * x1
        y = 10000 * (x2 / x3) ** (1 / self.m1)

        if self.normalize:
            y = y / self.l_max

        return y


class DisplayLight2PQ(nn.Module):
    def __init__(self, normalized=False, l_max=10000):
        super().__init__()
        self.normalized = normalized
        self.l_max = l_max

        self.m1 = 0.1593017578125  # 2610 / 16384
        self.m2 = 78.84375  # 2523 / 4096 * 128
        self.c1 = 0.8359375  # 3424 / 4096
        self.c2 = 18.8515625  # 2413 / 4096 * 32
        self.c3 = 18.6875  # 2392 / 128

    def forward(self, x):
        if self.normalized:
            x = x * self.l_max
        
        x = x / 10000
        x1 = x ** self.m1
        x2 = self.c1 + self.c2 * x1
        x3 = 1 + self.c3 * x1
        y = (x2 / x3) ** self.m2

        return y


class Gamma2DisplayLight(nn.Module):
    def __init__(self, normalize=False, gamma=2.40, L_w=100, L_b=0.1):
        super().__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.L_w = L_w
        self.L_b = L_b

    def forward(self, x):
        if self.normalize:
            return x ** self.gamma

        a = (self.L_w ** (1 / self.gamma) - self.L_b ** (1 / self.gamma)) ** self.gamma
        b = self.L_b ** (1 / self.gamma) / (self.L_w ** (1 / self.gamma) - self.L_b ** (1 / self.gamma))

        return a * (torch.clamp(x + b, eps)) ** self.gamma


class GamutConversion(nn.Module):
    def __init__(self, src='bt709', dest='bt2020'):
        super().__init__()
        if src in ['bt709', 'rec709'] and dest in ['bt2020', 'rec2020']:
            self.register_buffer('trans_mat', torch.Tensor([
                [0.6274, 0.3293, 0.0433],
                [0.0691, 0.9195, 0.0114],
                [0.0164, 0.0880, 0.8956]
            ]))
        else:
            raise NotImplementedError
        
    def forward(self, x):
        b, c, h, w = x.shape
        y = torch.matmul(self.trans_mat, 
                         x.reshape([b, c, -1])).reshape([b, c, h, w])
        
        return y.clamp(eps)


class RGB2LMS(nn.Module):
    def __init__(self, gamut='bt2020'):
        super().__init__()
        if gamut in ['bt2020', 'rec2020']:
            self.register_buffer('trans_mat', torch.Tensor([
                [1688, 2146, 262],
                [683, 2951, 462],
                [99, 309, 3688]
            ]) / 4096)
        
    def forward(self, x):
        b, c, h, w = x.shape
        y = torch.matmul(self.trans_mat, 
                         x.reshape([b, c, -1])).reshape([b, c, h, w])

        return y.clamp(eps)


class LMS2ICtCp(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('trans_mat', torch.Tensor([
            [2048, 2048, 0],
            [6610, -13613, 7003],
            [17933, -17390, -543]
        ]) / 4096)

    def forward(self, x):
        b, c, h, w = x.shape
        y = torch.matmul(self.trans_mat, 
                         x.reshape([b, c, -1])).reshape([b, c, h, w])
        
        return y #.clamp(eps, 1.0)


class ICtCp2ITP(nn.Module):
    def __init__(self, relative=False):
        super().__init__()
        scale_coeff = torch.Tensor([1., 0.5 * 1.823698, 1.887755]) \
            if relative else torch.Tensor([1., 0.5, 1.])
        self.register_buffer('scale_coeff', scale_coeff)
        
    def forward(self, x):
        y = torch.mul(x, self.scale_coeff.view(1, -1, 1, 1))

        return y
