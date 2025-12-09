import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

from .arch_util import ResidualBlockNoBN, make_layer

class ZeroConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, use_bf16=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        with torch.no_grad():
            self.weight.data.zero_()
            self.weight.data = self.weight.data.to(dtype)
            if self.bias is not None:
                self.bias.data.zero_()
                self.bias.data = self.bias.data.to(dtype)

class ZeroLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, use_bf16=False):
        super().__init__(in_features, out_features, bias=bias)

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        with torch.no_grad():
            self.weight.data.zero_()
            self.weight.data = self.weight.data.to(dtype)
            if self.bias is not None:
                self.bias.data.zero_()
                self.bias.data = self.bias.data.to(dtype)

class CNNMultiHeadAttention(nn.Module):

    def __init__(self, in_channels, num_heads=4, kernel_size=3, do_final_conv=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.do_final_conv = do_final_conv

        self.heads = nn.ModuleList([
            nn.Conv2d(in_channels*2, in_channels,
                      kernel_size=kernel_size,
                      padding=kernel_size//2,
                      bias=True)
            for _ in range(num_heads)
        ])

        if do_final_conv:
            self.final_conv = nn.Conv2d(
                in_channels * num_heads,
                in_channels,
                kernel_size=1, bias=True
            )
        else:
            self.final_conv = None

    def forward(self, cond1, cond2):
        B, C, H, W = cond1.shape
        x = torch.cat([cond1, cond2], dim=1)

        outs = []
        for conv in self.heads:
            gating = conv(x)                      # [B, C, H, W]
            gating = torch.sigmoid(gating)        # gating in [0,1]
            head_out = gating * cond1 + (1-gating)* cond2
            outs.append(head_out)
        out_cat = torch.cat(outs, dim=1)

        if self.final_conv is not None:
            fused = self.final_conv(out_cat)
            return fused
        else:
            return out_cat
        

@ARCH_REGISTRY.register()
class RealRep_fuse_norm_contra(nn.Module):
    def __init__(self, in_channels, transform_channels, global_cond_channels, spatial_cond_channels, refinement=True, num_heads=4):
        super().__init__()


        self.global_fusion_attention = CNNMultiHeadAttention(in_channels=global_cond_channels, num_heads=4,
                                              kernel_size=1, do_final_conv=True)

        self.spatial_fusion_attention = CNNMultiHeadAttention(in_channels=spatial_cond_channels, num_heads=4,
                                              kernel_size=3, do_final_conv=True)

        self.global_transform_1 = ConditionedTransform(
            in_channels, transform_channels, global_cond_channels, 'global')
        self.global_transform_2 = ConditionedTransform(
            transform_channels, transform_channels, global_cond_channels, 'global')
        self.global_transform_3 = ConditionedTransform(
            transform_channels, in_channels, global_cond_channels, 'global', activation=False)

        self.spatial_transform_1 = ConditionedTransform(
            in_channels, transform_channels, spatial_cond_channels, 'spatial', ada_method='cbam')
        self.spatial_transform_2 = ConditionedTransform(
            transform_channels, transform_channels, spatial_cond_channels, 'spatial', ada_method='cbam')
        self.spatial_transform_3 = ConditionedTransform(
            transform_channels, in_channels, spatial_cond_channels, 'spatial', ada_method='vanilla', activation=False)

        self.refinement = RefinementBlock(in_channels, in_channels, transform_channels) if refinement else None

    def switch_to_zero(self, use_bf16=False, device=torch.device('cpu')):
        self.global_transform_1.switch_to_zero(use_bf16=use_bf16, device=device)
        self.global_transform_2.switch_to_zero(use_bf16=use_bf16, device=device)
        self.global_transform_3.switch_to_zero(use_bf16=use_bf16, device=device)
        self.spatial_transform_1.switch_to_zero(use_bf16=use_bf16, device=device)
        self.spatial_transform_2.switch_to_zero(use_bf16=use_bf16, device=device)
        self.spatial_transform_3.switch_to_zero(use_bf16=use_bf16, device=device)
    def forward(self, x, global_cond_I, global_cond_TP, spatial_cond_I, spatial_cond_TP):
        B, C, H, W = x.shape

        fused_global_cond = self.global_fusion_attention(global_cond_I, global_cond_TP)  # [B, C_g]

        fused_spatial_cond = self.spatial_fusion_attention(spatial_cond_I, spatial_cond_TP)  # [B, C_s, H, W]

        coarsely_tuned_x = self.global_transform_1(x, fused_global_cond)
        coarsely_tuned_x = self.global_transform_2(coarsely_tuned_x, fused_global_cond)
        coarsely_tuned_x = self.global_transform_3(coarsely_tuned_x, fused_global_cond)

        spatially_modulated_x = self.spatial_transform_1(coarsely_tuned_x, fused_spatial_cond)
        spatially_modulated_x = self.spatial_transform_2(spatially_modulated_x, fused_spatial_cond)
        spatially_modulated_x = self.spatial_transform_3(spatially_modulated_x, fused_spatial_cond)

        if self.refinement:
            result = self.refinement(spatially_modulated_x)
        else:
            result = spatially_modulated_x

        self.mid_result = coarsely_tuned_x
        return result



class RefinementBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_features, num_blocks=3):
        super().__init__()

        self.conv_first = nn.Conv2d(in_channels, n_features, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_blocks, num_feat=n_features, pytorch_init=True)
        self.conv_last = nn.Conv2d(n_features, out_channels, 3, 1, 1)

    def forward(self, x):
        res = self.conv_last(self.body(self.conv_first(x)))

        return x + res


class HybridConditionModule(nn.Module):
    def __init__(self, in_channels, out_channels, global_cond_channels, init_mid_channels=16,
                 down_method='stride', up_method='bilinear'):
        super().__init__()

        self.in_conv = HyCondModConvBlock(in_channels, init_mid_channels)                           # in_channels -> 16
        self.enc_1 = HyCondModEncBlock(init_mid_channels, init_mid_channels * 2, down_method)       # 16 -> 32  1/2
        self.enc_2 = HyCondModEncBlock(init_mid_channels * 2, init_mid_channels * 4, down_method)   # 32 -> 64  1/4
        self.enc_3 = HyCondModEncBlock(init_mid_channels * 4, init_mid_channels * 8, down_method)   # 64 -> 128  1/8
        self.global_cond = HyCondModGlobalConditionBlock(init_mid_channels * 8, global_cond_channels)  # 128 -> 64
        self.dec_1 = HyCondModDecBlock(init_mid_channels * 8, init_mid_channels * 4, up_method)     # 128 -> 64  1/4
        self.dec_2 = HyCondModDecBlock(init_mid_channels * 4, init_mid_channels * 2, up_method)     # 64 -> 32  1/2
        self.dec_3 = HyCondModDecBlock(init_mid_channels * 2, init_mid_channels, up_method)         # 32 -> 16  1
        self.out_conv = HyCondModConvBlock(init_mid_channels, out_channels)                         # 16 -> out_channels

    def forward(self, x):
        x_1 = self.in_conv(x)       # 16
        x_2 = self.enc_1(x_1)       # 32
        x_3 = self.enc_2(x_2)       # 64
        x_4 = self.enc_3(x_3)       # 128
        z = self.global_cond(x_4)   # global_cond_channels
        y = self.dec_1(x_4, x_3)    # 64
        y = self.dec_2(y, x_2)      # 32
        y = self.dec_3(y, x_1)      # 16
        y = self.out_conv(y)        # out_channels

        return z, y


class HyCondModConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, act='relu'):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        if act == 'lrelu':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif act == 'prelu':
            self.act = nn.PReLU()
        else:
            self.act = nn.ReLU(inplace=True)
        

    def forward(self, x):
        return self.act(self.conv(x))


class HyCondModEncBlock(nn.Module):
    """
        input: (N, in_channels, H, W)
        output: (N, out_channels, H / 2, W / 2)
    """
    def __init__(self, in_channels, out_channels, downscale_method='stride'):
        super().__init__()

        if downscale_method == 'stride':
            self.down = HyCondModConvBlock(in_channels, out_channels, stride=2)
        elif downscale_method == 'pool':
            self.down = nn.Sequential(
                nn.MaxPool2d(2),
                HyCondModConvBlock(in_channels, out_channels)
            )
        else:
            raise NotImplementedError

        self.conv = HyCondModConvBlock(out_channels, out_channels)

    def forward(self, x):
        return self.conv(self.down(x))


class HyCondModDecBlock(nn.Module):
    """
        input: (N, in_channels, H, W)
        output: (N, out_channels, 2 * H, 2 * W)
    """
    def __init__(self, in_channels, out_channels, upscale_method='bilinear'):
        super().__init__()

        if upscale_method == 'bilinear':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                HyCondModConvBlock(in_channels, out_channels)
            )
        elif upscale_method == 'transpose':
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            raise NotImplementedError

        self.conv = HyCondModConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class HyCondModGlobalConditionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.cond = nn.Sequential(
            HyCondModConvBlock(in_channels, out_channels, kernel_size=1, padding=0),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        return self.cond(x)

class multiLinear(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_layers=3):
        super(multiLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        layers = []
        hidden_dim = in_channels  # 保持维度一致

        # 构造 num_layers 个 Linear 层，全部是 in_channels -> in_channels
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))  # 可选激活函数
        
        self.net = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_dim, out_channels)

    def forward(self, x):
        # x: [B, C, 1, 1] -> [B, C]
        x = x.view(x.size(0), -1)
        x = self.net(x)
        x = self.final(x)  # [B, C]
        return x.view(x.size(0), self.out_channels, 1, 1)


class CLayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(-1, C, 1, 1) * y + bias.view(-1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(-1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = (1. / torch.sqrt(var + eps)) * (g - y * mean_gy - mean_g)
        grad_weight = (grad_output * y).sum(dim=(2, 3), keepdim=True)  # [N, C, 1, 1]
        grad_bias = grad_output.sum(dim=(2, 3), keepdim=True)          # [N, C, 1, 1]

        return gx, grad_weight, grad_bias, None
        # return gx, (grad_output * y).sum(dim=3).sum(dim=2), grad_output.sum(dim=3).sum(dim=2), None

class CLayerNorm2d(nn.Module):

    def __init__(self, channels, rep_c,eps=1e-6):
        super(CLayerNorm2d, self).__init__()
        self.eps = eps
        self.weight = multiLinear(rep_c,channels)
        self.bias = multiLinear(rep_c,channels)
    def forward(self, x, label):
        weights = self.weight(label)
        bias = self.bias(label)
        return CLayerNormFunction.apply(x, weights, bias, self.eps)


class ConditionedTransform(nn.Module):
    """
        (in_channels) -> Conv -> (n_features) -> CLayerNorm2d -> Transform -> (n_features) -> Act -> (n_features)
                              (cond_channels) +|
    """

    def __init__(self,
                 in_channels,
                 n_features,
                 cond_channels,
                 transform='global',
                 ada_method='vanilla',
                 activation=True,
                 use_clayernorm=True):
        super().__init__()
        self.use_clayernorm = use_clayernorm
        self.transform_type = transform  # 保存 transform 类型
        self.conv1 = nn.Conv2d(in_channels, n_features, kernel_size=1)
        # self.conv2 = nn.Conv2d(n_features, n_features, kernel_size=1)

        if self.use_clayernorm:
            self.norm = CLayerNorm2d(n_features, cond_channels)

        if transform == 'global':
            self.transform = GlobalFeatureModulation(cond_channels, n_features)
        elif transform == 'spatial':
            self.transform = SpatialFeatureTransform(cond_channels, n_features, ada_method=ada_method)
        else:
            raise NotImplementedError

        self.act = nn.ReLU(inplace=True) if activation else None
    def switch_to_zero(self, use_bf16=False, device=torch.device('cpu')):
        # self.conv1 = ZeroConv2d(
        #     self.conv1.in_channels,
        #     self.conv1.out_channels,
        #     kernel_size=self.conv1.kernel_size,
        #     stride=self.conv1.stride,
        #     padding=self.conv1.padding,
        #     bias=(self.conv1.bias is not None),
        #     use_bf16=use_bf16
        # ).to(device)

        if hasattr(self.transform, "switch_to_zero"):
            self.transform.switch_to_zero(use_bf16=use_bf16, device=device)

    def _global_pool(self, cond):
        """
        输入: Tensor of shape [N, C, H, W]
        输出: Tensor of shape [N, C]
        """
        return F.adaptive_avg_pool2d(cond, 1).view(cond.size(0), -1)

    def forward(self, x, cond):
        out = self.conv1(x)

        if self.use_clayernorm:
            norm_cond = cond if self.transform_type == 'global' else self._global_pool(cond)
            out = self.norm(out, norm_cond) + out

        out = self.transform(out, cond)

        # out = self.conv2(out)

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
    def switch_to_zero(self, use_bf16=False, device=torch.device('cpu')):
        self.cond_scale = ZeroLinear(
            self.cond_scale.in_features,
            self.cond_scale.out_features,
            bias=(self.cond_scale.bias is not None),
            use_bf16=use_bf16
        ).to(device)

        self.cond_shift = ZeroLinear(
            self.cond_shift.in_features,
            self.cond_shift.out_features,
            bias=(self.cond_shift.bias is not None),
            use_bf16=use_bf16
        ).to(device)

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

    def switch_to_zero(self, use_bf16=False, device=torch.device('cpu')):
        def replace_last_conv_with_zero(seq):
            layers = list(seq)

            # 只在最后一个模块是 Conv2d 且不是 ZeroConv2d 时替换
            last_layer = layers[-1]
            if isinstance(last_layer, nn.Conv2d) and not isinstance(last_layer, ZeroConv2d):
                conv = last_layer
                layers[-1] = ZeroConv2d(
                    conv.in_channels,
                    conv.out_channels,
                    kernel_size=conv.kernel_size,
                    stride=conv.stride,
                    padding=conv.padding,
                    bias=(conv.bias is not None),
                    use_bf16=use_bf16
                ).to(device)

            return nn.Sequential(*layers)
        # def replace_last_conv_with_zero(seq):
        #     layers = list(seq)
        #     # Find last Conv2d layer that is not inside CBAM
        #     for i in reversed(range(len(layers))):
        #         layer = layers[i]
        #         if isinstance(layer, CBAM):
        #             continue  # Skip CBAM module
        #         if isinstance(layer, nn.Conv2d) and not isinstance(layer, ZeroConv2d):
        #             conv = layer
        #             layers[i] = ZeroConv2d(
        #                 conv.in_channels,
        #                 conv.out_channels,
        #                 kernel_size=conv.kernel_size,
        #                 stride=conv.stride,
        #                 padding=conv.padding,
        #                 bias=(conv.bias is not None),
        #                 use_bf16=use_bf16
        #             ).to(device)
        #             break
        #     return nn.Sequential(*layers)

        self.cond_scale = replace_last_conv_with_zero(self.cond_scale)
        self.cond_shift = replace_last_conv_with_zero(self.cond_shift)

    def forward(self, x, cond):
        scale = self.cond_scale(cond)  # (N, n_features, H, W)
        shift = self.cond_shift(cond)  # (N, n_features, H, W)
        out = x * scale + shift

        if self.residual:
            return out + x
        else:
            return out



class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, r):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // r, in_channels, 1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        y1 = self.avg_pool(x)
        y1 = self.mlp(y1)

        y2 = self.max_pool(x)
        y2 = self.mlp(y2)

        y = self.act(y1 + y2)

        return x * y


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_mask, max_mask], dim=1)

        mask = self.layer(mask)
        return x * mask


class CBAM(nn.Module):
    def __init__(self, in_channels, r=16, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)
        self.cam = ChannelAttentionModule(in_channels, r)
        self.sam = SpatialAttentionModule(kernel_size)

    def forward(self, x):
        y = self.conv(x)
        y = self.cam(y)
        y = self.sam(y)

        return x + y
    

# 定义共享基网络
class HybridConditionModuleBase(nn.Module):
    def __init__(self, in_channels=3, init_mid_channels=16, down_method='stride'):
        super().__init__()

        self.in_conv = HyCondModConvBlock(in_channels, init_mid_channels)                           # in_channels -> 16
        self.enc_1 = HyCondModEncBlock(init_mid_channels, init_mid_channels * 2, down_method)       # 16 -> 32  1/2
        self.enc_2 = HyCondModEncBlock(init_mid_channels * 2, init_mid_channels * 4, down_method)   # 32 -> 64  1/4
        self.enc_3 = HyCondModEncBlock(init_mid_channels * 4, init_mid_channels * 8, down_method)   # 64 -> 128  1/8

    def forward(self, x):
        x_1 = self.in_conv(x)       # 16
        x_2 = self.enc_1(x_1)       # 32
        x_3 = self.enc_2(x_2)       # 64
        x_4 = self.enc_3(x_3)       # 128
        return x_4  # 输出到高层特征提取器
    
# 定义高层特征提取器
class HybridConditionModuleHigh(nn.Module):
    def __init__(self, in_channels, global_cond_channels, spatial_cond_channels, init_mid_channels=16,
                 down_method='stride', up_method='bilinear'):
        super().__init__()

        self.global_cond = HyCondModGlobalConditionBlock(in_channels, global_cond_channels)  # in_channels -> 64
        self.dec_1 = HyCondModDecBlock(in_channels, in_channels // 2, up_method)     # in_channels -> 32
        self.dec_2 = HyCondModDecBlock(in_channels // 2, in_channels // 4, up_method) # 32 -> 16
        self.dec_3 = HyCondModDecBlock(in_channels // 4, in_channels // 8, up_method) # 16 -> 8
        self.out_conv = HyCondModConvBlock(in_channels // 8, spatial_cond_channels)   # 8 -> spatial_cond_channels

    def forward(self, x):
        z = self.global_cond(x)   # global_cond_channels
        y = self.dec_1(x)         # 32
        y = self.dec_2(y)         # 16
        y = self.dec_3(y)         # 8
        y = self.out_conv(y)      # spatial_cond_channels

        return z, y