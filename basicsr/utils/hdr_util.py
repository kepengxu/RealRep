"""
Classes and methods for conversion among common image signal representations
(including transfer function, color gamut, color space, etc).

Written by Tong Shao
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

type_check_default = True
range_check_default = True
clip_default = True
eps = 1e-10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BT709RGB2BT2020_MAT = np.array([
    [0.6274, 0.3293, 0.0433],
    [0.0691, 0.9195, 0.0114],
    [0.0164, 0.0880, 0.8956]
])

BT2020RGB2LMS_MAT = np.array([
    [1688, 2146, 262],
    [683, 2951, 462],
    [99, 309, 3688]
]) / 4096

LMS2BT2020RGB_MAT = np.array([
    [14082, -10271, 288],
    [-3242, 8130, -786],
    [-104, -408, 4607]
]) / 4096

LMS2ICTCP_MAT = np.array([
    [2048, 2048, 0],
    [6610, -13613, 7003],
    [17933, -17390, -543]
]) / 4096

ICTCP2LMS_MAT = np.array([
    [1, 0.009, 0.111],
    [1, -0.009, -0.111],
    [1, 0.56, -0.321]
])


class SdrHdrPairRepresentationConversion(nn.Module):
    """
    Convert Representation (TF & colorspace) of SDR-HDR data pair.
    Supported representation:
        'rgb': Gamma, PQ
        'ycbcr': Gamma, PQ
        'xyz': Linear, Linear
        'ictcp': PQ, PQ
    """

    def __init__(self, src='rgb', dst='ictcp', zero_one_norm=True):
        super().__init__()
        if src == 'rgb':
            if dst == 'rgb':
                self.sdr_convert = None
                self.hdr_convert = None
                self.shift_range = False
            elif dst == 'ycbcr':
                self.sdr_convert = ConvertFromRGBToYCbCr('bt709')
                self.hdr_convert = ConvertFromRGBToYCbCr('bt2020')
                self.shift_range = zero_one_norm
            elif dst == 'xyz':
                self.sdr_convert = ConvertToDisplayLinearXYZ('bt1886')
                self.hdr_convert = ConvertToDisplayLinearXYZ('bt2100_pq')
                self.shift_range = False
            elif dst == 'ictcp':
                self.sdr_convert = ConvertToICtCp('bt1886')
                self.hdr_convert = ConvertToICtCp('bt2100_pq')
                self.shift_range = zero_one_norm
            else:
                raise NotImplementedError("Unsupported representation.")
        else:
            raise NotImplementedError

    def forward(self, sdr_image, hdr_image):
        if self.sdr_convert:
            sdr_image = remove_specials(
                self.sdr_convert(sdr_image)
            )
        if self.hdr_convert:
            hdr_image = remove_specials(
                self.hdr_convert(hdr_image)
            )
        if self.shift_range:
            sdr_image = chroma_shift_range(sdr_image)
            hdr_image = chroma_shift_range(hdr_image)

        return sdr_image, hdr_image


class HdrConvertBackToRGB(nn.Module):
    def __init__(self, src='ictcp', zero_one_normed=True):
        super().__init__()
        if src == 'rgb':
            self.shift_range = False
            self.hdr_convert = None
        elif src == 'ycbcr':
            self.shift_range = zero_one_normed
            self.hdr_convert = ConvertFromYCbCrToRGB('bt2020')
        elif src == 'xyz':
            self.shift_range = False
            self.hdr_convert = ConvertToBT2100PQRGB('linear_xyz')
        elif src == 'ictcp':
            self.shift_range = zero_one_normed
            self.hdr_convert =ConvertToBT2100PQRGB('ictcp')
        else:
            raise NotImplementedError("Unsupported representation.")

    def forward(self, hdr_image):
        if self.shift_range:
            hdr_image = chroma_shift_range(hdr_image, -0.5)
        if self.hdr_convert:
            hdr_image = remove_specials(
                self.hdr_convert(hdr_image)
            )

        return hdr_image


# deprecated
class LightConversion(nn.Module):
    def __init__(self, src_tf: str, dst_tf: str):
        super().__init__()

        supported_options = ['gamma', 'pq', 'display']
        if src_tf not in supported_options or dst_tf not in supported_options:
            raise ValueError

        if src_tf == dst_tf:
            self.convert = ()

        if src_tf == 'gamma' and self.dst_tf == 'display':
            self.convert = (gamma_to_display_light,)

        elif self.src_tf == 'display' and self.dst_tf == 'gamma':
            self.convert = (display_light_to_gamma,)

        elif self.src_tf == 'pq' and self.dst_tf == 'display':
            self.convert = (pq_to_display_light,)

        elif self.src_tf == 'display' and self.dst_tf == 'pq':
            self.convert = (display_light_to_pq,)

        elif self.src_tf == 'gamma' and self.dst_tf == 'pq':
            self.convert = (gamma_to_display_light, display_light_to_pq)

        else:
            raise NotImplementedError

    def forward(self, x: Tensor):
        for op in self.convert:
            x = op(x)

        return x


class ConvertFromRGBToYCbCr(nn.Module):
    """
    Convert gamma encoded RGB signal into YCbCr representation.
    """

    def __init__(self, primaries='bt709'):
        super().__init__()

        self.primaries = primaries

    def forward(self, x):
        return rgb_to_ycbcr(x, self.primaries)


class ConvertFromYCbCrToRGB(nn.Module):
    """
    Convert gamma encoded YCbCr signal into RGB representation.
    """

    def __init__(self, primaries='bt709'):
        super().__init__()

        self.primaries = primaries

    def forward(self, x):
        return ycbcr_to_rgb(x, self.primaries)


class ConvertToDisplayLinearXYZ(nn.Module):
    """
    Convert the input signal to display-referred linear XYZ representation.
    """

    def __init__(self, src_rep='bt2100_pq'):
        super().__init__()

        supported_options = ['bt2100_pq', 'bt2100_hlg', 'bt1886']
        if src_rep not in supported_options:
            raise ValueError

        if src_rep == 'bt2100_pq':
            self.convert = (pq_to_display_light, rgb_to_xyz)
            self.args = ({}, {'primaries': 'bt2020'})

        elif src_rep == 'bt1886':
            self.convert = (gamma_to_display_light, rgb_to_xyz)
            self.args = ({}, {'primaries': 'bt709'})

        else:
            raise NotImplementedError

    def forward(self, x):
        for i, op in enumerate(self.convert):
            x = op(x, **self.args[i])

        return x


class ConvertToDisplayLinearRGB(nn.Module):
    """
    Convert the input signal to BT.2020 display-referred linear RGB representation.
    """

    def __init__(self, src_rep='bt2100_pq'):
        super().__init__()

        supported_options = ['linear_xyz', 'bt2100_pq', 'bt2100_hlg', 'bt1886', 'ictcp']
        if src_rep not in supported_options:
            raise ValueError

        if src_rep == 'bt2100_pq':
            self.convert = (pq_to_display_light,)

        elif src_rep == 'bt1886':
            self.convert = (gamma_to_display_light, rgb_bt709_to_bt2020)

        elif src_rep == 'ictcp':
            self.convert = (ictcp_to_lms, pq_to_display_light, lms_to_rgb_bt2020)

        else:
            raise NotImplementedError

    def forward(self, x):
        for op in self.convert:
            x = op(x)

        return x


class ConvertToICtCp(nn.Module):
    """
    Convert the input signal to BT.2100 ICtCp representation.
    """

    def __init__(self, src_rep='linear_rgb'):
        super().__init__()

        supported_options = ['linear_rgb', 'linear_xyz', 'bt2100_pq', 'bt2100_hlg', 'bt1886']
        if src_rep not in supported_options:
            raise ValueError

        if src_rep == 'linear_rgb':
            self.convert = (rgb_bt2020_to_lms, display_light_to_pq, lms_to_ictcp)

        elif src_rep in ['linear_xyz', 'bt2100_pq', 'bt2100_hlg', 'bt1886']:
            self.convert = (ConvertToDisplayLinearRGB(src_rep), ConvertToICtCp('linear_rgb'))

        else:
            raise NotImplementedError

    def forward(self, x):
        for op in self.convert:
            x = op(x)

        return x


class ConvertToBT2100PQRGB(nn.Module):
    """
    Convert the input signal to BT.2100 PQ encoded RGB representation.
    """

    def __init__(self, src_rep='linear_rgb'):
        super().__init__()

        supported_options = ['linear_rgb', 'linear_xyz', 'bt2100_hlg', 'bt1886', 'ictcp']
        if src_rep not in supported_options:
            raise ValueError

        if src_rep == 'linear_rgb':
            self.convert = (display_light_to_pq,)

        elif src_rep in ['linear_xyz', 'bt2100_hlg', 'bt1886', 'ictcp']:
            self.convert = (ConvertToDisplayLinearRGB(src_rep), ConvertToBT2100PQRGB('linear_rgb'))

        else:
            raise NotImplementedError

    def forward(self, x):
        for op in self.convert:
            x = op(x)

        return x


def input_check(x, func_name,
                type_check=type_check_default, rep=None, warning=range_check_default, clip=clip_default):
    """
    Check the validity of input.

    :param x: input image
    :param func_name: function name
    :param type_check: whether to check the type of the input
    :param rep: data representation
    :param warning: whether to warn the data out-of-range of the input
    :param clip: whether to clip the data range
    :return: the input image after check and clip (if set True)
    """
    if type_check and not isinstance(x, (np.ndarray, Tensor)):
        raise TypeError('Expecting np.ndarray or torch.Tensor, but got {}'.format(type(x)))

    if rep is None or not (warning and clip):
        pass

    if clip:
        x = remove_specials(x)

    if rep in ['normalized_rgb', ] and (x.min() < 0 or x.max() > 1):
        if warning:
            print('Warning: The input of {} has values out of {} range, min: {}, max: {}'
                  .format(func_name, rep, x.min(), x.max()))
        if clip:
            x = torch.clamp(x, eps, 1) if isinstance(x, Tensor) else np.clip(x, eps, 1)

    elif rep in ['normalized_ycbcr', 'normalized_ictcp'] and (
            (isinstance(x, Tensor) and (x[:, 0, :, :].min() < 0 or x[:, 0, :, :].max() > 1
                                        or x[:, 1:, :, :].min() < -0.5 or x[:, 1:, :, :].max() > 0.5))
            or (isinstance(x, np.ndarray) and (x[:, :, 0].min() < 0 or x[:, :, 0].max() > 1
                                               or x[:, :, 1:].min() < -0.5 or x[:, :, 1:].max() > 0.5))):
        if warning:
            print('Warning: The input of {} has values out of {} range, min: {}, max: {}'
                  .format(func_name, rep, x.min(), x.max()))
        if clip:
            if isinstance(x, Tensor):
                x[:, 0, :, :] = torch.clamp(x[:, 0, :, :], eps, 1)
                x[:, 1:, :, :] = torch.clamp(x[:, 1:, :, :], -0.5, 0.5)
            else:
                x[:, :, 0] = np.clip(x[:, :, 0], eps, 1)
                x[:, :, 1:] = np.clip(x[:, :, 1:], -0.5, 0.5)

    elif rep in ['linear', 'non_negative'] and x.min() < 0:
        if warning:
            print('Warning: The input of {} has values out of {} range, min: {}, max: {}'
                  .format(func_name, rep, x.min(), x.max()))
        if clip:
            x = torch.clamp(x, eps) if isinstance(x, Tensor) else np.clip(x, eps, None)

    return x


def remove_specials(x, clamping_value=eps):
    """
    Clamp the special values (Inf and NaN) to the given value (clamping_value).

    :param x: input image
    :param clamping_value: clamping value
    :return: the input image with special values replaced
    """
    if isinstance(x, Tensor):
        x[torch.isnan(x) | torch.isinf(x)] = clamping_value
    else:
        x[np.isnan(x) | np.isinf(x)] = clamping_value

    return x


def chroma_shift_range(x, offset=0.5):
    """
    Shift the data range of luma / chroma represented (YCbCr, ICtCp, etc.) image.

    :param x: [0, 1] / [-0.5, 0.5] for luma / chroma channel respectively.
    :param offset: shift offset (typically 0.5 or -0.5).
    :return: [0, 1] normalized image for all three channels.
    """
    if isinstance(x, Tensor):
        if x.dim() == 3:
            x[1:, :, :] += offset
        else:
            x[:, 1:, :, :] += offset
    else:
        x[:, :, 1:] += offset

    return x


# deprecated, use input_check instead
def clip_to_range(x, range='normalized_rgb'):
    """
    Clip the data range to certain representation.

    :param x: input image
    :param range: data range of which representation
    :return:
    """
    if range is None:
        pass

    elif range in ['normalized_rgb']:
        x = torch.clamp(x, 0, 1) if isinstance(x, Tensor) else np.clip(x, 0, 1)

    elif range in ['normalized_ycbcr', 'normalized_ictcp']:
        if isinstance(x, Tensor):
            x[:, 0, :, :] = torch.clamp(x[:, 0, :, :], 0, 1)
            x[:, 1:, :, :] = torch.clamp(x[:, 1:, :, :], -0.5, 0.5)
        else:
            x[:, :, 0] = np.clip(x[:, :, 0], 0, 1)
            x[:, :, 1:] = np.clip(x[:, :, 1:], -0.5, 0.5)

    elif range == ['linear', 'non_negative']:
        x = torch.clamp(x, 0) if isinstance(x, Tensor) else np.clip(x, 0, None)

    else:
        raise NotImplementedError

    return x


def gamma_to_display_light(x, normalize=False, gamma=2.40, L_w=100, L_b=0.1):
    """
    Convert gamma encoded signal to linear display light signal,
    according to Rec. ITU-R BT.1886 Reference EOTF.

    :param x: [0, 1] normalized gamma encoded signal
    :param normalize: whether to normalize the output
    :param L_w: luminance of 'white'
    :param L_b: luminance of 'black'
    :return: [0, 1] if normalize else [0, L_w] linear display light signal
    """
    x = input_check(x, 'gamma_to_display_light', rep='normalized_rgb')

    if normalize:
        return x ** gamma

    a = (L_w ** (1 / gamma) - L_b ** (1 / gamma)) ** gamma
    b = L_b ** (1 / gamma) / (L_w ** (1 / gamma) - L_b ** (1 / gamma))

    if isinstance(x, Tensor):
        return a * (torch.clamp(x + b, 0)) ** gamma
    else:
        return a * (np.maximum(x + b, 0)) ** gamma


def display_light_to_gamma(x, normalized=False, gamma=2.40, L_w=100, L_b=0.1):
    """
    Convert linear display light signal to gamma encoded signal,
    according to the inverse of Rec. ITU-R BT.1886 Reference EOTF.

    :param x: [0, 1] if normalized else [0, L_w] linear display light signal
    :param normalized: whether the input is normalized
    :param L_w: luminance of 'white'
    :param L_b: luminance of 'black'
    :return: [0, 1] normalized gamma encoded signal
    """
    x = input_check(x, 'display_light_to_gamma', rep='linear')

    if normalized:
        return x ** (1 / gamma)

    else:
        raise NotImplementedError


def pq_to_display_light(x, normalize=False, L_max=10000):
    """
    Convert PQ encoded signal to linear display light signal,
    according to Rec. ITU-R BT.2100 PQ EOTF.

    :param x: [0, 1] normalized PQ encoded signal
    :param normalize: whether to normalize the output
    :param L_max: maximum luminance
    :return: [0, 10000 / L_max] if normalize else [0, 10000] linear display light signal
    """
    x = input_check(x, 'pq_to_display_light', rep='normalized_rgb')

    m1 = 0.1593017578125  # 2610 / 16384
    m2 = 78.84375  # 2523 / 4096 * 128
    c1 = 0.8359375  # 3424 / 4096
    c2 = 18.8515625  # 2413 / 4096 * 32
    c3 = 18.6875  # 2392 / 128

    if isinstance(x, Tensor):
        x1 = x ** (1 / m2)
        x2 = torch.clamp(x1 - c1, 0)
        x3 = (c2 - c3 * x1)
        y = 10000 * (x2 / x3) ** (1 / m1)
    else:
        y = 10000 * (np.maximum(x ** (1 / m2) - c1, 0) / (c2 - c3 * x ** (1 / m2))) ** (1 / m1)

    return y / L_max if normalize else y


def display_light_to_pq(x, normalized=False, L_max=10000):
    """
    Convert linear display light signal to PQ encoded signal,
    according to the inverse of Rec. ITU-R BT.2100 PQ EOTF.

    :param x: [0, 10000 / L_max] if normalized else [0, 10000] linear display light signal
    :param normalized: whether the input is normalized
    :param L_max: maximum luminance
    :return: [0, 1] normalized PQ encoded signal
    """
    x = input_check(x, 'display_light_to_pq', rep='linear')

    m1 = 0.1593017578125  # 2610 / 16384
    m2 = 78.84375  # 2523 / 4096 * 128
    c1 = 0.8359375  # 3424 / 4096
    c2 = 18.8515625  # 2413 / 4096 * 32
    c3 = 18.6875  # 2392 / 128

    if normalized:
        x = x * L_max / 10000
    else:
        x = x / 10000
    
    x1 = x ** m1
    x2 = (c1 + c2 * x1)
    x3 = (1 + c3 * x1)

    return (x2 / x3) ** m2


def rgb_bt709_to_bt2020(x):
    """
    Convert the color gamut of linearly represented normalized RGB signal from BT.709 to BT.2020,
    according to Rec. ITU-R BT.2087 conversion matrix.

    :param x: linearly represented, normalized RGB color signal
    :return:
    """
    x = input_check(x, 'rgb_bt709_to_bt2020', rep=None)

    if isinstance(x, Tensor):
        mat = torch.Tensor(BT709RGB2BT2020_MAT).to(device)
        b, c, h, w = x.shape
        y = torch.matmul(mat, x.reshape([b, c, -1])).reshape([b, c, h, w])
    else:
        mat = BT709RGB2BT2020_MAT
        y = np.matmul(x, mat.transpose())

    return y


# TODO: implement the matrix transform method of image
def rgb_to_ycbcr(x, primaries='bt709'):
    """
    Convert gamma encoded RGB signal into YCbCr representation.

    :param x: [0, 1] normalized gamma encoded RGB signal
    :param primaries: 'bt709' or 'bt2020'
    :return: [0, 1][-0.5, 0.5] normalized YCbCr signal
    """
    x = input_check(x, 'rgb_to_ycbcr', rep='normalized_rgb')

    if primaries == 'bt709':
        a1 = 0.2126
        a2 = 0.7152
        a3 = 0.0722
        b1 = 1.8556
        b2 = 1.5748
    elif primaries == 'bt2020':
        a1 = 0.2627
        a2 = 0.6780
        a3 = 0.0593
        b1 = 1.8814
        b2 = 1.4746
    else:
        raise NotImplementedError

    if isinstance(x, Tensor):
        y = torch.zeros_like(x)
        y[:, 0, :, :] = a1 * x[:, 0, :, :] + a2 * x[:, 1, :, :] + a3 * x[:, 2, :, :]
        y[:, 1, :, :] = - (a1 / b1) * x[:, 0, :, :] - (a2 / b1) * x[:, 1, :, :] + 0.5 * x[:, 2, :, :]
        y[:, 2, :, :] = 0.5 * x[:, 0, :, :] - (a2 / b2) * x[:, 1, :, :] - (a3 / b2) * x[:, 2, :, :]
    else:
        y = np.zeros_like(x)
        y[:, :, 0] = a1 * x[:, :, 0] + a2 * x[:, :, 1] + a3 * x[:, :, 2]
        y[:, :, 1] = - (a1 / b1) * x[:, :, 0] - (a2 / b1) * x[:, :, 1] + 0.5 * x[:, :, 2]
        y[:, :, 2] = 0.5 * x[:, :, 0] - (a2 / b2) * x[:, :, 1] - (a3 / b2) * x[:, :, 2]

    return y


def ycbcr_to_rgb(x, primaries='bt709'):
    """
    Convert gamma encoded YCbCr signal into RGB representation.

    :param x: [0, 1][-0.5, 0.5] normalized YCbCr signal
    :param primaries: 'bt709' or 'bt2020'
    :return: [0, 1] normalized gamma encoded RGB signal
    """
    x = input_check(x, 'ycbcr_to_rgb', rep='normalized_ycbcr')

    if primaries == 'bt709':
        a1 = 0.2126
        a2 = 0.7152
        a3 = 0.0722
        b1 = 1.8556
        b2 = 1.5748
    elif primaries == 'bt2020':
        a1 = 0.2627
        a2 = 0.6780
        a3 = 0.0593
        b1 = 1.8814
        b2 = 1.4746
    else:
        raise NotImplementedError

    if isinstance(x, Tensor):
        y = torch.zeros_like(x)
        y[:, 0, :, :] = x[:, 0, :, :] + b2 * x[:, 2, :, :]
        y[:, 1, :, :] = x[:, 0, :, :] - (a3 * b1 / a2) * x[:, 1, :, :] - (a1 * b2 / a2) * x[:, 2, :, :]
        y[:, 2, :, :] = x[:, 0, :, :] + b1 * x[:, 1, :, :]
    else:
        y = np.zeros_like(x)
        y[:, :, 0] = x[:, :, 0] + b2 * x[:, :, 2]
        y[:, :, 1] = x[:, :, 0] - (a3 * b1 / a2) * x[:, :, 1] - (a1 * b2 / a2) * x[:, :, 2]
        y[:, :, 2] = x[:, :, 0] + b1 * x[:, :, 1]

    return y


def rgb_to_xyz(x, primaries='bt709'):
    """
    Convert linear RGB signal into XYZ representation.

    :param x: linear RGB signal
    :param primaries: 'bt709' or 'bt2020'
    :return: linear XYZ signal
    """
    x = input_check(x, 'rgb_to_xyz', rep='linear')

    if primaries == 'bt709':
        a1 = [0.4124, 0.3576, 0.1805]
        a2 = [0.2126, 0.7152, 0.0722]
        a3 = [0.0193, 0.1192, 0.9505]
    elif primaries == 'bt2020':
        a1 = [0.6370, 0.1446, 0.1689]
        a2 = [0.2627, 0.6780, 0.0593]
        a3 = [0, 0.0281, 1.0610]
    else:
        raise NotImplementedError

    if isinstance(x, Tensor):
        y = torch.zeros_like(x)
        y[:, 0, :, :] = a1[0] * x[:, 0, :, :] + a1[1] * x[:, 1, :, :] + a1[2] * x[:, 2, :, :]
        y[:, 1, :, :] = a2[0] * x[:, 0, :, :] + a2[1] * x[:, 1, :, :] + a2[2] * x[:, 2, :, :]
        y[:, 2, :, :] = a3[0] * x[:, 0, :, :] + a3[1] * x[:, 1, :, :] + a3[2] * x[:, 2, :, :]
    else:
        y = np.zeros_like(x)
        y[:, :, 0] = a1[0] * x[:, :, 0] + a1[1] * x[:, :, 1] + a1[2] * x[:, :, 2]
        y[:, :, 1] = a2[0] * x[:, :, 0] + a2[1] * x[:, :, 1] + a2[2] * x[:, :, 2]
        y[:, :, 2] = a3[0] * x[:, :, 0] + a3[1] * x[:, :, 1] + a3[2] * x[:, :, 2]

    return y


def xyz_to_rgb(x, primaries='bt709'):
    """
    Convert linear XYZ signal into RGB representation.

    :param x: linear XYZ signal
    :param primaries: 'bt709' or 'bt2020'
    :return: linear RGB signal
    """
    x = input_check(x, 'xyz_to_rgb', rep='linear')

    if primaries == 'bt709':
        a1 = [3.2410, -1.5374, -0.4986]
        a2 = [-0.9692, 1.8760, 0.0416]
        a3 = [0.0556, -0.2040, 1.0570]
    elif primaries == 'bt2020':
        a1 = [1.7167, -0.3557, -0.2534]
        a2 = [-0.6667, 1.6165, 0.0158]
        a3 = [0.0176, -0.0428, 0.9421]
    else:
        raise NotImplementedError

    if isinstance(x, Tensor):
        y = torch.zeros_like(x)
        y[:, 0, :, :] = a1[0] * x[:, 0, :, :] + a1[1] * x[:, 1, :, :] + a1[2] * x[:, 2, :, :]
        y[:, 1, :, :] = a2[0] * x[:, 0, :, :] + a2[1] * x[:, 1, :, :] + a2[2] * x[:, 2, :, :]
        y[:, 2, :, :] = a3[0] * x[:, 0, :, :] + a3[1] * x[:, 1, :, :] + a3[2] * x[:, 2, :, :]
    else:
        y = np.zeros_like(x)
        y[:, :, 0] = a1[0] * x[:, :, 0] + a1[1] * x[:, :, 1] + a1[2] * x[:, :, 2]
        y[:, :, 1] = a2[0] * x[:, :, 0] + a2[1] * x[:, :, 1] + a2[2] * x[:, :, 2]
        y[:, :, 2] = a3[0] * x[:, :, 0] + a3[1] * x[:, :, 1] + a3[2] * x[:, :, 2]

    return y


def rgb_bt2020_to_lms(x):
    """
    Convert BT.2020 linear RGB signal into LMS representation,
    according to Dolby white paper on ICtCp.

    :param x: BT.2020 linear RGB signal
    :return: linear LMS signal
    """
    x = input_check(x, 'rgb_bt2020_to_lms', rep='linear')

    if isinstance(x, Tensor):
        mat = torch.Tensor(BT2020RGB2LMS_MAT).to(device)
        b, c, h, w = x.shape
        y = torch.matmul(mat, x.reshape([b, c, -1])).reshape([b, c, h, w])
    else:
        mat = BT2020RGB2LMS_MAT
        y = np.matmul(x, mat.transpose())

    return y


def lms_to_rgb_bt2020(x):
    """
    Convert linear LMS signal to BT.2020 linear RGB signal.

    :param x: linear LMS signal
    :return: BT.2020 linear RGB signal
    """
    x = input_check(x, 'lms_to_rgb_bt2020', rep='linear')

    if isinstance(x, Tensor):
        mat = torch.Tensor(LMS2BT2020RGB_MAT).to(device)
        b, c, h, w = x.shape
        y = torch.matmul(mat, x.reshape([b, c, -1])).reshape([b, c, h, w])
    else:
        mat = LMS2BT2020RGB_MAT
        y = np.matmul(x, mat.transpose())

    return y


def lms_to_ictcp(x):
    """
    Convert PQ encoded LMS signal into ICtCp representation,
    according to Dolby white paper on ICtCp.

    :param x: PQ encoded LMS signal
    :return: PQ encoded ICtCp signal
    """
    x = input_check(x, 'lms_to_ictcp', rep='normalized_rgb')

    if isinstance(x, Tensor):
        mat = torch.Tensor(LMS2ICTCP_MAT).to(device)
        b, c, h, w = x.shape
        y = torch.matmul(mat, x.reshape([b, c, -1])).reshape([b, c, h, w])
    else:
        mat = LMS2ICTCP_MAT
        y = np.matmul(x, mat.transpose())

    return y


def ictcp_to_lms(x):
    """
    Convert PQ encoded ICtCp signal into LMS representation.

    :param x: PQ encoded ICtCp signal
    :return: PQ encoded LMS signal
    """
    x = input_check(x, 'ictcp_to_lms', rep='normalized_ictcp')

    if isinstance(x, Tensor):
        mat = torch.Tensor(ICTCP2LMS_MAT).to(device)
        b, c, h, w = x.shape
        y = torch.matmul(mat, x.reshape([b, c, -1])).reshape([b, c, h, w])
    else:
        mat = ICTCP2LMS_MAT
        y = np.matmul(x, mat.transpose())

    return y
