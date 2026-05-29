import math
import torch
import torch.nn as nn

from basicsr.archs import histogram_module as histogram
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class HistogramLoss(nn.Module):
    def __init__(self, loss_weight=1.0, colorspace='rgb-uv', h_sz=64, img_sz=160,
                 resizing='sampling', method='inverse-quadratic', sigma=0.02,
                 intensity_scale=True, hist_boundary=None):
        super().__init__()
        self.loss_weight = loss_weight
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if colorspace == 'rgb-uv':
            self.cal_histogram = histogram.RGBuvHistBlock(h_sz, img_sz, resizing, method, sigma,
                                                          intensity_scale, hist_boundary, device=self.device)
        elif colorspace == 'lab':
            self.cal_histogram = histogram.LabHistBlock(h_sz, img_sz, resizing, method, sigma,
                                                        intensity_scale, hist_boundary, device=self.device)
        else:
            raise NotImplementedError

    def forward(self, pred, target):
        pred_hist = self.cal_histogram(pred)
        target_hist = self.cal_histogram(target)

        return self.loss_weight * (1 / math.sqrt(2.0) * (torch.sqrt(torch.sum(
            torch.pow(torch.sqrt(target_hist) - torch.sqrt(pred_hist), 2))))
                / pred_hist.shape[0])
