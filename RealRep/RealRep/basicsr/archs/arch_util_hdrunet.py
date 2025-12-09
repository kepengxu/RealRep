import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

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

class SFTLayer(nn.Module):
    def __init__(self, in_nc=32, out_nc=64, nf=32):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_scale_conv1 = nn.Conv2d(nf, out_nc, 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_shift_conv1 = nn.Conv2d(nf, out_nc, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift


class ResBlock_with_SFT(nn.Module):
    def __init__(self, nf=64):
        super(ResBlock_with_SFT, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.sft1 = SFTLayer(in_nc=32, out_nc=64, nf=32)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.sft2 = SFTLayer(in_nc=32, out_nc=64, nf=32)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.sft1(x)
        fea = F.relu(self.conv1(fea), inplace=True)
        fea = self.sft2((fea, x[1]))
        fea = self.conv2(fea)
        return (x[0] + fea, x[1])


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output


# import torch
# import torch.nn as nn
# import torch.nn.init as init
# import torch.nn.functional as F
# from torch.nn.modules.batchnorm import _BatchNorm

# @torch.no_grad()
# def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
#     """Initialize network weights.

#     Args:
#         module_list (list[nn.Module] | nn.Module): Modules to be initialized.
#         scale (float): Scale initialized weights, especially for residual
#             blocks. Default: 1.
#         bias_fill (float): The value to fill bias. Default: 0
#         kwargs (dict): Other arguments for initialization function.
#     """
#     if not isinstance(module_list, list):
#         module_list = [module_list]
#     for module in module_list:
#         for m in module.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, **kwargs)
#                 m.weight.data *= scale
#                 if m.bias is not None:
#                     m.bias.data.fill_(bias_fill)
#             elif isinstance(m, nn.Linear):
#                 init.kaiming_normal_(m.weight, **kwargs)
#                 m.weight.data *= scale
#                 if m.bias is not None:
#                     m.bias.data.fill_(bias_fill)
#             elif isinstance(m, _BatchNorm):
#                 init.constant_(m.weight, 1)
#                 if m.bias is not None:
#                     m.bias.data.fill_(bias_fill)


# class ResidualBlockNoBN(nn.Module):
#     """Residual block without BN.

#     It has a style of:
#         ---Conv-ReLU-Conv-+-
#          |________________|

#     Args:
#         num_feat (int): Channel number of intermediate features.
#             Default: 64.
#         res_scale (float): Residual scale. Default: 1.
#         pytorch_init (bool): If set to True, use pytorch default init,
#             otherwise, use default_init_weights. Default: False.
#     """

#     def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
#         super(ResidualBlockNoBN, self).__init__()
#         self.res_scale = res_scale
#         self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
#         self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
#         self.relu = nn.ReLU(inplace=True)

#         if not pytorch_init:
#             default_init_weights([self.conv1, self.conv2], 0.1)

#     def forward(self, x):
#         identity = x
#         out = self.conv2(self.relu(self.conv1(x)))
#         return identity + out * self.res_scale

# def initialize_weights(net_l, scale=1):
#     if not isinstance(net_l, list):
#         net_l = [net_l]
#     for net in net_l:
#         for m in net.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, a=0, mode='fan_in')
#                 m.weight.data *= scale  # for residual block
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 init.kaiming_normal_(m.weight, a=0, mode='fan_in')
#                 m.weight.data *= scale
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias.data, 0.0)

# def make_layer_hycond(basic_block, num_basic_block, **kwarg):
#     """Make layers by stacking the same blocks.

#     Args:
#         basic_block (nn.module): nn.module class for basic block.
#         num_basic_block (int): number of blocks.

#     Returns:
#         nn.Sequential: Stacked blocks in nn.Sequential.
#     """
#     layers = []
#     for _ in range(num_basic_block):
#         layers.append(basic_block(**kwarg))
#     return nn.Sequential(*layers)

# def make_layer(block, n_layers):
#     layers = []
#     for _ in range(n_layers):
#         layers.append(block())
#     return nn.Sequential(*layers)


# class ResidualBlock_noBN(nn.Module):
#     '''Residual block w/o BN
#     ---Conv-ReLU-Conv-+-
#      |________________|
#     '''

#     def __init__(self, nf=64):
#         super(ResidualBlock_noBN, self).__init__()
#         self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

#         # initialization
#         initialize_weights([self.conv1, self.conv2], 0.1)

#     def forward(self, x):
#         identity = x
#         out = F.relu(self.conv1(x), inplace=True)
#         out = self.conv2(out)
#         return identity + out

# class SFTLayer(nn.Module):
#     def __init__(self, in_nc=32, out_nc=32, nf=32):
#         super(SFTLayer, self).__init__()
#         self.SFT_scale_conv0 = nn.Conv2d(in_nc, nf, 1)
#         self.SFT_scale_conv1 = nn.Conv2d(nf, out_nc, 1)
#         self.SFT_shift_conv0 = nn.Conv2d(in_nc, nf, 1)
#         self.SFT_shift_conv1 = nn.Conv2d(nf, out_nc, 1)

#     def forward(self, x, cond):
#         # x[0]: fea; x[1]: cond
#         scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(cond), 0.1, inplace=True))
#         shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(cond), 0.1, inplace=True))
#         return x * (scale + 1) + shift


# class ResBlock_with_SFT(nn.Module):
#     def __init__(self, nf=32, global_nf=64):
#         super(ResBlock_with_SFT, self).__init__()
#         self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
#         self.sft1g = SFTLayer(in_nc=global_nf, out_nc=32, nf=global_nf)
#         self.sft1 = SFTLayer(in_nc=32, out_nc=32, nf=32)
#         self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
#         self.sft2g = SFTLayer(in_nc=global_nf, out_nc=32, nf=global_nf)
#         self.sft2 = SFTLayer(in_nc=32, out_nc=32, nf=32)
#         self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

#         # initialization
#         initialize_weights([self.conv1, self.conv2], 0.1)

#     def forward(self, x):
#         # x[0]: fea; x[1]: cond
#         fea = self.sft1g(x[0], x[1])
#         fea = self.sft1(fea, x[2])
#         fea = F.relu(self.conv1(fea), inplace=True)
#         fea = self.sft2g(fea, x[1])
#         fea = self.sft2(fea, x[2])
#         fea = self.conv2(fea)
#         return (x[0] + fea, x[1], x[2])


# def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
#     """Warp an image or feature map with optical flow
#     Args:
#         x (Tensor): size (N, C, H, W)
#         flow (Tensor): size (N, H, W, 2), normal value
#         interp_mode (str): 'nearest' or 'bilinear'
#         padding_mode (str): 'zeros' or 'border' or 'reflection'

#     Returns:
#         Tensor: warped image or feature map
#     """
#     assert x.size()[-2:] == flow.size()[1:3]
#     B, C, H, W = x.size()
#     # mesh grid
#     grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
#     grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
#     grid.requires_grad = False
#     grid = grid.type_as(x)
#     vgrid = grid + flow
#     # scale grid to [-1,1]
#     vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
#     vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
#     vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
#     output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
#     return output
