import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6

"""
##### Copyright 2021 Mahmoud Afifi.

 If you find this code useful, please cite our paper:

 Mahmoud Afifi, Marcus A. Brubaker, and Michael S. Brown. "HistoGAN: 
 Controlling Colors of GAN-Generated and Real Images via Color Histograms." 
 In CVPR, 2021.

 @inproceedings{afifi2021histogan,
  title={Histo{GAN}: Controlling Colors of {GAN}-Generated and Real Images via 
  Color Histograms},
  author={Afifi, Mahmoud and Brubaker, Marcus A. and Brown, Michael S.},
  booktitle={CVPR},
  year={2021}
}
####
"""


class RGBuvHistBlock(nn.Module):
    def __init__(self, h=64, insz=150, resizing='interpolation',
                 method='inverse-quadratic', sigma=0.02, intensity_scale=True,
                 hist_boundary=None, green_only=False, device='cuda'):
        """ Computes the RGB-uv histogram feature of a given image.
        Args:
          h: histogram dimension size (scalar). The default value is 64.
          insz: maximum size of the input image; if it is larger than this size, the
            image will be resized (scalar). Default value is 150 (i.e., 150 x 150
            pixels).
          resizing: resizing method if applicable. Options are: 'interpolation' or
            'sampling'. Default is 'interpolation'.
          method: the method used to count the number of pixels for each bin in the
            histogram feature. Options are: 'thresholding', 'RBF' (radial basis
            function), or 'inverse-quadratic'. Default value is 'inverse-quadratic'.
          sigma: if the method value is 'RBF' or 'inverse-quadratic', then this is
            the sigma parameter of the kernel function. The default value is 0.02.
          intensity_scale: boolean variable to use the intensity scale (I_y in
            Equation 2). Default value is True.
          hist_boundary: a list of histogram boundary values. Default is [-3, 3].
          green_only: boolean variable to use only the log(g/r), log(g/b) channels.
            Default is False.

        Methods:
          forward: accepts input image and returns its histogram feature. Note that
            unless the method is 'thresholding', this is a differentiable function
            and can be easily integrated with the loss function. As mentioned in the
             paper, the 'inverse-quadratic' was found more stable than 'RBF' in our
             training.
        """
        super(RGBuvHistBlock, self).__init__()
        self.h = h
        self.insz = insz
        self.device = device
        self.resizing = resizing
        self.method = method
        self.intensity_scale = intensity_scale
        self.green_only = green_only
        if hist_boundary is None:
            hist_boundary = [-3, 3]
        hist_boundary.sort()
        self.hist_boundary = hist_boundary
        if self.method == 'thresholding':
            self.eps = (abs(hist_boundary[0]) + abs(hist_boundary[1])) / h
        else:
            self.sigma = sigma

    def forward(self, x):
        # clamp and resize the input image
        x = torch.clamp(x, 0, 1)
        if x.shape[2] > self.insz or x.shape[3] > self.insz:
            if self.resizing == 'interpolation':
                x_sampled = F.interpolate(x, size=(self.insz, self.insz),
                                          mode='bilinear', align_corners=False)
            elif self.resizing == 'sampling':
                inds_1 = torch.LongTensor(
                    np.linspace(0, x.shape[2], self.h, endpoint=False)).to(
                    device=self.device)
                inds_2 = torch.LongTensor(
                    np.linspace(0, x.shape[3], self.h, endpoint=False)).to(
                    device=self.device)
                x_sampled = x.index_select(2, inds_1)
                x_sampled = x_sampled.index_select(3, inds_2)
            else:
                raise Exception(
                    f'Wrong resizing method. It should be: interpolation or sampling. '
                    f'But the given value is {self.resizing}.')
        else:
            x_sampled = x

        L = x_sampled.shape[0]  # size of mini-batch
        if x_sampled.shape[1] > 3:
            x_sampled = x_sampled[:, :3, :, :]
        X = torch.unbind(x_sampled, dim=0)
        hists = torch.zeros((x_sampled.shape[0], 1 + int(not self.green_only) * 2,
                             self.h, self.h)).to(device=self.device)
        for l in range(L):
            I = torch.t(torch.reshape(X[l], (3, -1)))   # (H * W, 3)
            II = torch.pow(I, 2)
            if self.intensity_scale:
                Iy = torch.unsqueeze(torch.sqrt(II[:, 0] + II[:, 1] + II[:, 2] + EPS),
                                     dim=1)     # (H * W, 1)
            else:
                Iy = 1

            # R
            if not self.green_only:
                Iu0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 1] +
                                                                           EPS), dim=1)  # R / G (H * W, 1)
                Iv0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 2] +
                                                                           EPS), dim=1)  # R / B (H * W, 1)
                diff_u0 = abs(
                    Iu0 - torch.unsqueeze(torch.tensor(np.linspace(
                        self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                        dim=0).to(self.device))     # (H * W, h)
                diff_v0 = abs(
                    Iv0 - torch.unsqueeze(torch.tensor(np.linspace(
                        self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                        dim=0).to(self.device))     # (H * W, h)
                if self.method == 'thresholding':
                    diff_u0 = torch.reshape(diff_u0, (-1, self.h)) <= self.eps / 2
                    diff_v0 = torch.reshape(diff_v0, (-1, self.h)) <= self.eps / 2
                elif self.method == 'RBF':
                    diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_u0 = torch.exp(-diff_u0)  # Radial basis function
                    diff_v0 = torch.exp(-diff_v0)
                elif self.method == 'inverse-quadratic':
                    diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_u0 = 1 / (1 + diff_u0)  # Inverse quadratic
                    diff_v0 = 1 / (1 + diff_v0)
                else:
                    raise Exception(
                        f'Wrong kernel method. It should be either thresholding, RBF,'
                        f' inverse-quadratic. But the given value is {self.method}.')
                diff_u0 = diff_u0.type(torch.float32)
                diff_v0 = diff_v0.type(torch.float32)
                a = torch.t(Iy * diff_u0)
                hists[l, 0, :, :] = torch.mm(a, diff_v0)

            # G
            Iu1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 0] + EPS),
                                  dim=1)        # G / R
            Iv1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 2] + EPS),
                                  dim=1)        # G / B
            diff_u1 = abs(
                Iu1 - torch.unsqueeze(torch.tensor(np.linspace(
                    self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                    dim=0).to(self.device))
            diff_v1 = abs(
                Iv1 - torch.unsqueeze(torch.tensor(np.linspace(
                    self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                    dim=0).to(self.device))

            if self.method == 'thresholding':
                diff_u1 = torch.reshape(diff_u1, (-1, self.h)) <= self.eps / 2
                diff_v1 = torch.reshape(diff_v1, (-1, self.h)) <= self.eps / 2
            elif self.method == 'RBF':
                diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u1 = torch.exp(-diff_u1)  # Gaussian
                diff_v1 = torch.exp(-diff_v1)
            elif self.method == 'inverse-quadratic':
                diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                                    2) / self.sigma ** 2
                diff_u1 = 1 / (1 + diff_u1)  # Inverse quadratic
                diff_v1 = 1 / (1 + diff_v1)

            diff_u1 = diff_u1.type(torch.float32)
            diff_v1 = diff_v1.type(torch.float32)
            a = torch.t(Iy * diff_u1)
            if not self.green_only:
                hists[l, 1, :, :] = torch.mm(a, diff_v1)
            else:
                hists[l, 0, :, :] = torch.mm(a, diff_v1)

            # B
            if not self.green_only:
                Iu2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 0] +
                                                                           EPS), dim=1)     # B / R
                Iv2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 1] +
                                                                           EPS), dim=1)     # B / G
                diff_u2 = abs(
                    Iu2 - torch.unsqueeze(torch.tensor(np.linspace(
                        self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                        dim=0).to(self.device))
                diff_v2 = abs(
                    Iv2 - torch.unsqueeze(torch.tensor(np.linspace(
                        self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                        dim=0).to(self.device))
                if self.method == 'thresholding':
                    diff_u2 = torch.reshape(diff_u2, (-1, self.h)) <= self.eps / 2
                    diff_v2 = torch.reshape(diff_v2, (-1, self.h)) <= self.eps / 2
                elif self.method == 'RBF':
                    diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_u2 = torch.exp(-diff_u2)  # Gaussian
                    diff_v2 = torch.exp(-diff_v2)
                elif self.method == 'inverse-quadratic':
                    diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                                        2) / self.sigma ** 2
                    diff_u2 = 1 / (1 + diff_u2)  # Inverse quadratic
                    diff_v2 = 1 / (1 + diff_v2)
                diff_u2 = diff_u2.type(torch.float32)
                diff_v2 = diff_v2.type(torch.float32)
                a = torch.t(Iy * diff_u2)
                hists[l, 2, :, :] = torch.mm(a, diff_v2)

        # normalization
        hists_normalized = hists / (
                ((hists.sum(dim=1)).sum(dim=1)).sum(dim=1).view(-1, 1, 1, 1) + EPS)

        return hists_normalized


class rgChromaHistBlock(nn.Module):
    def __init__(self, h=64, insz=150, resizing='interpolation',
                 method='inverse-quadratic', sigma=0.02, intensity_scale=False,
                 hist_boundary=None, device='cuda'):
        """ Computes the rg-chroma histogram feature of a given image.
        Args:
          h: histogram dimension size (scalar). The default value is 64.
          insz: maximum size of the input image; if it is larger than this size, the
            image will be resized (scalar). Default value is 150 (i.e., 150 x 150
            pixels).
          resizing: resizing method if applicable. Options are: 'interpolation' or
            'sampling'. Default is 'interpolation'.
          method: the method used to count the number of pixels for each bin in the
            histogram feature. Options are: 'thresholding', 'RBF' (radial basis
            function), or 'inverse-quadratic'. Default value is 'inverse-quadratic'.
          sigma: if the method value is 'RBF' or 'inverse-quadratic', then this is
            the sigma parameter of the kernel function. The default value is 0.02.
          intensity_scale: boolean variable to use the intensity scale (I_y in
            Equation 2). Default value is False.
          hist_boundary: a list of histogram boundary values. Default is [0, 1].

        Methods:
          forward: accepts input image and returns its histogram feature. Note that
            unless the method is 'thresholding', this is a differentiable function
            and can be easily integrated with the loss function. As mentioned in the
             paper, the 'inverse-quadratic' was found more stable than 'RBF' in our
             training.
        """
        super(rgChromaHistBlock, self).__init__()
        self.h = h
        self.insz = insz
        self.device = device
        self.resizing = resizing
        self.method = method
        self.intensity_scale = intensity_scale
        if hist_boundary is None:
            hist_boundary = [0, 1]
        hist_boundary.sort()
        self.hist_boundary = hist_boundary
        if self.method == 'thresholding':
            self.eps = (abs(hist_boundary[0]) + abs(hist_boundary[1])) / h
        else:
            self.sigma = sigma

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        if x.shape[2] > self.insz or x.shape[3] > self.insz:
            if self.resizing == 'interpolation':
                x_sampled = F.interpolate(x, size=(self.insz, self.insz),
                                          mode='bilinear', align_corners=False)
            elif self.resizing == 'sampling':
                inds_1 = torch.LongTensor(
                    np.linspace(0, x.shape[2], self.h, endpoint=False)).to(
                    device=self.device)
                inds_2 = torch.LongTensor(
                    np.linspace(0, x.shape[3], self.h, endpoint=False)).to(
                    device=self.device)
                x_sampled = x.index_select(2, inds_1)
                x_sampled = x_sampled.index_select(3, inds_2)
            else:
                raise Exception(
                    f'Wrong resizing method. It should be: interpolation or sampling. '
                    f'But the given value is {self.resizing}.')
        else:
            x_sampled = x

        L = x_sampled.shape[0]  # size of mini-batch
        if x_sampled.shape[1] > 3:
            x_sampled = x_sampled[:, :3, :, :]
        X = torch.unbind(x_sampled, dim=0)
        hists = torch.zeros((x_sampled.shape[0], 1, self.h, self.h)).to(
            device=self.device)
        for l in range(L):
            I = torch.t(torch.reshape(X[l], (3, -1)))
            II = torch.pow(I, 2)
            if self.intensity_scale:
                Iy = torch.unsqueeze(torch.sqrt(II[:, 0] + II[:, 1] + II[:, 2] + EPS),
                                     dim=1)
            else:
                Iy = 1

            Ir = torch.unsqueeze(I[:, 0] / (torch.sum(I, dim=-1) + EPS), dim=1)
            Ig = torch.unsqueeze(I[:, 1] / (torch.sum(I, dim=-1) + EPS), dim=1)

            diff_r = abs(Ir - torch.unsqueeze(torch.tensor(np.linspace(
                self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                dim=0).to(self.device))
            diff_g = abs(Ig - torch.unsqueeze(torch.tensor(np.linspace(
                self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                dim=0).to(self.device))

            if self.method == 'thresholding':
                diff_r = torch.reshape(diff_r, (-1, self.h)) <= self.eps / 2
                diff_g = torch.reshape(diff_g, (-1, self.h)) <= self.eps / 2
            elif self.method == 'RBF':
                diff_r = torch.pow(torch.reshape(diff_r, (-1, self.h)),
                                   2) / self.sigma ** 2
                diff_g = torch.pow(torch.reshape(diff_g, (-1, self.h)),
                                   2) / self.sigma ** 2
                diff_r = torch.exp(-diff_r)  # Gaussian
                diff_g = torch.exp(-diff_g)
            elif self.method == 'inverse-quadratic':
                diff_r = torch.pow(torch.reshape(diff_r, (-1, self.h)),
                                   2) / self.sigma ** 2
                diff_g = torch.pow(torch.reshape(diff_g, (-1, self.h)),
                                   2) / self.sigma ** 2
                diff_r = 1 / (1 + diff_r)  # Inverse quadratic
                diff_g = 1 / (1 + diff_g)

            diff_r = diff_r.type(torch.float32)
            diff_g = diff_g.type(torch.float32)
            a = torch.t(Iy * diff_r)

            hists[l, 0, :, :] = torch.mm(a, diff_g)

        # normalization
        hists_normalized = hists / (
                ((hists.sum(dim=1)).sum(dim=1)).sum(dim=1).view(-1, 1, 1, 1) + EPS)

        return hists_normalized


class LabHistBlock(nn.Module):
    def __init__(self, h=64, insz=150, resizing='interpolation',
                 method='inverse-quadratic', sigma=0.02, intensity_scale=False,
                 hist_boundary=None, device='cuda'):
        """ Computes the Lab (CIELAB) histogram feature of a given image.
        Args:
          h: histogram dimension size (scalar). The default value is 64.
          insz: maximum size of the input image; if it is larger than this size, the
            image will be resized (scalar). Default value is 150 (i.e., 150 x 150
            pixels).
          resizing: resizing method if applicable. Options are: 'interpolation' or
            'sampling'. Default is 'interpolation'.
          method: the method used to count the number of pixels for each bin in the
            histogram feature. Options are: 'thresholding', 'RBF' (radial basis
            function), or 'inverse-quadratic'. Default value is 'inverse-quadratic'.
          sigma: if the method value is 'RBF' or 'inverse-quadratic', then this is
            the sigma parameter of the kernel function. The default value is 0.02.
          intensity_scale: boolean variable to use the intensity scale (I_y in
            Equation 2). Default value is False.
          hist_boundary: a list of histogram boundary values. Default is [0, 1].

        Methods:
          forward: accepts input image and returns its histogram feature. Note that
            unless the method is 'thresholding', this is a differentiable function
            and can be easily integrated with the loss function. As mentioned in the
             paper, the 'inverse-quadratic' was found more stable than 'RBF' in our
             training.
        """
        super(LabHistBlock, self).__init__()
        self.h = h
        self.insz = insz
        self.device = device
        self.resizing = resizing
        self.method = method
        self.intensity_scale = intensity_scale
        if hist_boundary is None:
            hist_boundary = [0, 1]
        hist_boundary.sort()
        self.hist_boundary = hist_boundary
        if self.method == 'thresholding':
            self.eps = (abs(hist_boundary[0]) + abs(hist_boundary[1])) / h
        else:
            self.sigma = sigma

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        if x.shape[2] > self.insz or x.shape[3] > self.insz:
            if self.resizing == 'interpolation':
                x_sampled = F.interpolate(x, size=(self.insz, self.insz),
                                          mode='bilinear', align_corners=False)
            elif self.resizing == 'sampling':
                inds_1 = torch.LongTensor(
                    np.linspace(0, x.shape[2], self.h, endpoint=False)).to(
                    device=self.device)
                inds_2 = torch.LongTensor(
                    np.linspace(0, x.shape[3], self.h, endpoint=False)).to(
                    device=self.device)
                x_sampled = x.index_select(2, inds_1)
                x_sampled = x_sampled.index_select(3, inds_2)
            else:
                raise Exception(
                    f'Wrong resizing method. It should be: interpolation or sampling. '
                    f'But the given value is {self.resizing}.')
        else:
            x_sampled = x

        L = x_sampled.shape[0]  # size of mini-batch
        if x_sampled.shape[1] > 3:
            x_sampled = x_sampled[:, :3, :, :]
        X = torch.unbind(x_sampled, dim=0)
        hists = torch.zeros((x_sampled.shape[0], 1, self.h, self.h)).to(
            device=self.device)
        for l in range(L):
            I = torch.t(torch.reshape(X[l], (3, -1)))
            if self.intensity_scale:
                Il = torch.unsqueeze(I[:, 0], dim=1)
            else:
                Il = 1

            Ia = torch.unsqueeze(I[:, 1], dim=1)
            Ib = torch.unsqueeze(I[:, 2], dim=1)

            diff_a = abs(Ia - torch.unsqueeze(torch.tensor(np.linspace(
                self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                dim=0).to(self.device))
            diff_b = abs(Ib - torch.unsqueeze(torch.tensor(np.linspace(
                self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
                dim=0).to(self.device))

            if self.method == 'thresholding':
                diff_a = torch.reshape(diff_a, (-1, self.h)) <= self.eps / 2
                diff_b = torch.reshape(diff_b, (-1, self.h)) <= self.eps / 2
            elif self.method == 'RBF':
                diff_a = torch.pow(torch.reshape(diff_a, (-1, self.h)),
                                   2) / self.sigma ** 2
                diff_b = torch.pow(torch.reshape(diff_b, (-1, self.h)),
                                   2) / self.sigma ** 2
                diff_a = torch.exp(-diff_a)  # Gaussian
                diff_b = torch.exp(-diff_b)
            elif self.method == 'inverse-quadratic':
                diff_a = torch.pow(torch.reshape(diff_a, (-1, self.h)),
                                   2) / self.sigma ** 2
                diff_b = torch.pow(torch.reshape(diff_b, (-1, self.h)),
                                   2) / self.sigma ** 2
                diff_a = 1 / (1 + diff_a)  # Inverse quadratic
                diff_b = 1 / (1 + diff_b)

            diff_a = diff_a.type(torch.float32)
            diff_b = diff_b.type(torch.float32)
            a = torch.t(Il * diff_a)

            hists[l, 0, :, :] = torch.mm(a, diff_b)

        # normalization
        hists_normalized = hists / (
                ((hists.sum(dim=1)).sum(dim=1)).sum(dim=1).view(-1, 1, 1, 1) + EPS)

        return hists_normalized
