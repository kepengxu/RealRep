import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs import conversion_module as conversion
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class DeltaEITPLoss(nn.Module):
    def __init__(self, loss_weight=1.0, relative=False, mode='pixelwise', pred_domain='hdr', target_domain='hdr'):
        super().__init__()
        self.loss_weight = loss_weight
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.convert_pred = conversion.RGB2ITP(src=pred_domain, relative=relative)
        self.convert_target = conversion.RGB2ITP(src=target_domain, relative=relative)

        self.filter = SCIELAB_SpatialFilter(samp_per_deg=50).to(self.device) \
            if mode == 'spatial' else None

    def forward(self, pred, target):
        # convert to ITP
        pred_itp = self.convert_pred(pred)
        target_itp = self.convert_target(target)

        if self.filter:
            pred_itp = self.filter(pred_itp)
            target_itp = self.filter(target_itp)

        mse_map = F.mse_loss(pred_itp, target_itp, reduction='none')  # (n, c, h, w)
        # print('mse_map', mse_map.min(), mse_map.max())
        delta_e_map = torch.sqrt(torch.sum(mse_map, dim=1) + conversion.eps)  # (n, h, w)
        # print('delta_e_map', delta_e_map.min(), delta_e_map.max())

        return self.loss_weight * torch.mean(delta_e_map)


@LOSS_REGISTRY.register()
class ITPPixLoss(nn.Module):
    def __init__(self, loss_weight=1.0, pix_criterion='l1', pred_domain='hdr', target_domain='hdr'):
        super().__init__()
        self.loss_weight = loss_weight
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.convert_pred = conversion.RGB2ITP(src=pred_domain)
        self.convert_target = conversion.RGB2ITP(src=target_domain)
        if pix_criterion in ['l1', 'L1']:
            self.criterion = nn.L1Loss()
        elif pix_criterion in ['l2', 'L2', 'mse', 'MSE']:
            self.criterion = nn.MSELoss()

    def forward(self, pred, target):
        # convert to ITP
        pred_itp = self.convert_pred(pred)
        target_itp = self.convert_target(target)

        loss = self.criterion(pred_itp, target_itp)

        return self.loss_weight * loss


class SCIELAB_SpatialFilter(nn.Module):
    def __init__(self, samp_per_deg=50, mode='itp'):
        self.samp_per_deg = samp_per_deg
        self.kernel_size = math.ceil(samp_per_deg / 2) * 2 - 1
        self.mode = mode
        
        self.param_lum = [[0.921, 0.0283], [0.105, 0.133], [-0.108, 4.336]]
        self.param_rg = [[0.531, 0.0392], [0.330, 0.494]]
        self.param_by = [[0.488, 0.0536], [0.371, 0.386]]

        kernels = self._generate_kernels()  # (3, ks, ks)
        kernels = torch.FloatTensor(kernels).unsqueeze(0)  # (1, 3, ks, ks)
        self.weight = nn.Parameter(data=kernels, requires_grad=False)

    def forward(self, x):
        x = F.conv2d(x, self.weight, padding='same', groups=3)
        return x
        
    def _gaussian_kernel(self, sigma):
        kernel = np.zeros(shape=(self.kernel_size, self.kernel_size), dtype=np.float64)
        radius = self.kernel_size // 2
        for y in range(-radius, radius + 1):  # [-r, r]
            for x in range(-radius, radius + 1):
                val = np.exp(-1.0 / (sigma ** 2) * (x ** 2 + y ** 2))
                kernel[y + radius, x + radius] = val
        return kernel / np.sum(kernel)

    def _generate_kernel_for_one_channel(self, param):
        kernel = np.zeros(shape=(self.kernel_size, self.kernel_size), dtype=np.float64)
        for weight, sigma in param:
            kernel += weight * self._gaussian_kernel(sigma * self.samp_per_deg)
        return kernel / np.sum(kernel)
    
    def _generate_kernels(self):
        if self.mode == 'itp':
            kernel_1 = self._generate_kernel_for_one_channel(self.param_lum)  # I
            kernel_2 = self._generate_kernel_for_one_channel(self.param_by)   # T
            kernel_3 = self._generate_kernel_for_one_channel(self.param_rg)   # P
        
        return np.array([kernel_1, kernel_2, kernel_3])