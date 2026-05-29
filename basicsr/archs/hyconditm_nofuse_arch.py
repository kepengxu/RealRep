import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

from .arch_util import ResidualBlockNoBN, make_layer       

class CNNMultiHeadAttention(nn.Module):
    """
    使用CNN实现的多头“注意力”融合，将两个输入 [B, C, H, W] 特征图 cond1, cond2 融合为一个输出。
    - 不使用Transformer/全局Flatten，而是多头卷积门控，不会产生过多显存占用。
    - 每个“头”输出与输入C相同的多通道gating，再插值融合 cond1, cond2。
    - 最后把多头输出拼接在通道维度上，再用 1x1 conv 压回 [B, C, H, W]。
    """

    def __init__(self, in_channels, num_heads=4, kernel_size=3, do_final_conv=True):
        """
        参数:
        - in_channels: 输入特征图通道数 (C)。
        - num_heads: 多头数，表示有多少个并行的卷积分支来生成多通道gating。
        - kernel_size: 每个卷积分支的卷积核大小。
        - do_final_conv: 是否在多头拼接后再用1×1卷积将通道压回 C；若=False则输出 [B, C*num_heads, H, W]。
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.do_final_conv = do_final_conv

        # 多个卷积分支：将 [B, 2C, H, W] => [B, C, H, W] (多通道gating)
        # gating_i = sigmoid(conv_i(x)) => [B, C, H, W]
        # 然后 head_out_i = gating_i * cond1 + (1 - gating_i)* cond2
        self.heads = nn.ModuleList([
            nn.Conv2d(in_channels*2, in_channels,
                      kernel_size=kernel_size,
                      padding=kernel_size//2,
                      bias=True)
            for _ in range(num_heads)
        ])

        # 将 num_heads 个 [B,C,H,W] 拼接 => [B, C*num_heads, H, W]
        # 若需要压回 [B, C, H, W]，则加一层1×1 conv
        if do_final_conv:
            self.final_conv = nn.Conv2d(
                in_channels * num_heads,
                in_channels,
                kernel_size=1, bias=True
            )
        else:
            self.final_conv = None

    def forward(self, cond1, cond2):
        """
        cond1, cond2: [B, C, H, W]
        return: fused [B, C, H, W] (若 do_final_conv=True)
                或   [B, C*num_heads, H, W] (若 do_final_conv=False)
        """
        B, C, H, W = cond1.shape
        # 1) 在通道维度拼接 => [B, 2C, H, W]
        x = torch.cat([cond1, cond2], dim=1)

        # 2) 多头并行 => gating, 再和 cond1, cond2 融合
        outs = []
        for conv in self.heads:
            gating = conv(x)                      # [B, C, H, W]
            gating = torch.sigmoid(gating)        # gating in [0,1]
            head_out = gating * cond1 + (1-gating)* cond2
            outs.append(head_out)
        # 3) 拼接多头输出 => [B, C*num_heads, H, W]
        out_cat = torch.cat(outs, dim=1)

        # 4) 若需要，做 final 1×1 conv => [B, C, H, W]
        if self.final_conv is not None:
            fused = self.final_conv(out_cat)
            return fused
        else:
            return out_cat
        

@ARCH_REGISTRY.register()
class HyCondITMv1_fuse_contra_woglobal(nn.Module):
    def __init__(self, in_channels, transform_channels, global_cond_channels, spatial_cond_channels, refinement=True, num_heads=4):
        super().__init__()


        # self.global_fusion_attention = CNNMultiHeadAttention(in_channels=global_cond_channels, num_heads=4,
        #                                       kernel_size=1, do_final_conv=True)

        self.spatial_fusion_attention = CNNMultiHeadAttention(in_channels=spatial_cond_channels, num_heads=4,
                                              kernel_size=3, do_final_conv=True)

        # # 全局条件变换模块
        # self.global_transform_1 = ConditionedTransform(
        #     in_channels, transform_channels, global_cond_channels, 'global')
        # self.global_transform_2 = ConditionedTransform(
        #     transform_channels, transform_channels, global_cond_channels, 'global')
        # self.global_transform_3 = ConditionedTransform(
        #     transform_channels, in_channels, global_cond_channels, 'global', activation=False)

        # 空间条件变换模块
        self.spatial_transform_1 = ConditionedTransform(
            in_channels, transform_channels, spatial_cond_channels, 'spatial', ada_method='cbam')
        self.spatial_transform_2 = ConditionedTransform(
            transform_channels, transform_channels, spatial_cond_channels, 'spatial', ada_method='cbam')
        self.spatial_transform_3 = ConditionedTransform(
            transform_channels, in_channels, spatial_cond_channels, 'spatial', ada_method='vanilla', activation=False)

        # 精细化模块
        self.refinement = RefinementBlock(in_channels, in_channels, transform_channels) if refinement else None


    def forward(self, x, spatial_cond_I, spatial_cond_TP):
        B, C, H, W = x.shape

        # 融合全局条件
        # fused_global_cond = self.global_fusion_attention(global_cond_I, global_cond_TP)  # [B, C_g]

        # 融合空间条件
        fused_spatial_cond = self.spatial_fusion_attention(spatial_cond_I, spatial_cond_TP)  # [B, C_s, H, W]

        # # 全局变换
        # coarsely_tuned_x = self.global_transform_1(x, fused_global_cond)
        # coarsely_tuned_x = self.global_transform_2(coarsely_tuned_x, fused_global_cond)
        # coarsely_tuned_x = self.global_transform_3(coarsely_tuned_x, fused_global_cond)

        # 空间变换
        spatially_modulated_x = self.spatial_transform_1(x, fused_spatial_cond)
        spatially_modulated_x = self.spatial_transform_2(spatially_modulated_x, fused_spatial_cond)
        spatially_modulated_x = self.spatial_transform_3(spatially_modulated_x, fused_spatial_cond)

        # 精细化
        if self.refinement:
            result = self.refinement(spatially_modulated_x)
        else:
            result = spatially_modulated_x

        # self.mid_result = coarsely_tuned_x
        return result
    
@ARCH_REGISTRY.register()
class HyCondITMv1_fuse_contra_wolocal(nn.Module):
    def __init__(self, in_channels, transform_channels, global_cond_channels, spatial_cond_channels, refinement=True, num_heads=4):
        super().__init__()


        self.global_fusion_attention = CNNMultiHeadAttention(in_channels=global_cond_channels, num_heads=4,
                                              kernel_size=1, do_final_conv=True)

        # self.spatial_fusion_attention = CNNMultiHeadAttention(in_channels=spatial_cond_channels, num_heads=4,
        #                                       kernel_size=3, do_final_conv=True)

        # 全局条件变换模块
        self.global_transform_1 = ConditionedTransform(
            in_channels, transform_channels, global_cond_channels, 'global')
        self.global_transform_2 = ConditionedTransform(
            transform_channels, transform_channels, global_cond_channels, 'global')
        self.global_transform_3 = ConditionedTransform(
            transform_channels, in_channels, global_cond_channels, 'global', activation=False)

        # # 空间条件变换模块
        # self.spatial_transform_1 = ConditionedTransform(
        #     in_channels, transform_channels, spatial_cond_channels, 'spatial', ada_method='cbam')
        # self.spatial_transform_2 = ConditionedTransform(
        #     transform_channels, transform_channels, spatial_cond_channels, 'spatial', ada_method='cbam')
        # self.spatial_transform_3 = ConditionedTransform(
        #     transform_channels, in_channels, spatial_cond_channels, 'spatial', ada_method='vanilla', activation=False)

        # 精细化模块
        self.refinement = RefinementBlock(in_channels, in_channels, transform_channels) if refinement else None


    def forward(self, x, global_cond_I, global_cond_TP):
        B, C, H, W = x.shape

        # 融合全局条件
        fused_global_cond = self.global_fusion_attention(global_cond_I, global_cond_TP)  # [B, C_g]

        # # 融合空间条件
        # fused_spatial_cond = self.spatial_fusion_attention(spatial_cond_I, spatial_cond_TP)  # [B, C_s, H, W]

        # 全局变换
        coarsely_tuned_x = self.global_transform_1(x, fused_global_cond)
        coarsely_tuned_x = self.global_transform_2(coarsely_tuned_x, fused_global_cond)
        coarsely_tuned_x = self.global_transform_3(coarsely_tuned_x, fused_global_cond)

        # # 空间变换
        # spatially_modulated_x = self.spatial_transform_1(x, fused_spatial_cond)
        # spatially_modulated_x = self.spatial_transform_2(spatially_modulated_x, fused_spatial_cond)
        # spatially_modulated_x = self.spatial_transform_3(spatially_modulated_x, fused_spatial_cond)

        # 精细化
        if self.refinement:
            result = self.refinement(coarsely_tuned_x)
        else:
            result = coarsely_tuned_x

        # self.mid_result = coarsely_tuned_x
        return result

@ARCH_REGISTRY.register()
class HyCondITMv1_nofuse_contra(nn.Module):
    def __init__(self, in_channels, transform_channels, global_cond_channels, spatial_cond_channels, refinement=True):
        super().__init__()

        # 全局条件变换模块
        self.global_transform_1 = ConditionedTransform(
            in_channels, transform_channels, global_cond_channels, 'global')
        self.global_transform_2 = ConditionedTransform(
            transform_channels, transform_channels, global_cond_channels, 'global')
        self.global_transform_3 = ConditionedTransform(
            transform_channels, in_channels, global_cond_channels, 'global', activation=False)

        # 空间条件变换模块
        self.spatial_transform_1 = ConditionedTransform(
            in_channels, transform_channels, spatial_cond_channels, 'spatial', ada_method='cbam')
        self.spatial_transform_2 = ConditionedTransform(
            transform_channels, transform_channels, spatial_cond_channels, 'spatial', ada_method='cbam')
        self.spatial_transform_3 = ConditionedTransform(
            transform_channels, in_channels, spatial_cond_channels, 'spatial', ada_method='vanilla', activation=False)

        # 精细化模块
        self.refinement = RefinementBlock(in_channels, in_channels, transform_channels) if refinement else None


    def forward(self, x, global_cond_I, global_cond_TP, spatial_cond_I, spatial_cond_TP):
        B, C, H, W = x.shape

        # 融合全局条件
        fused_global_cond = global_cond_I + global_cond_TP

        # 融合空间条件
        fused_spatial_cond = spatial_cond_I+spatial_cond_TP

        # 全局变换
        coarsely_tuned_x = self.global_transform_1(x, fused_global_cond)
        coarsely_tuned_x = self.global_transform_2(coarsely_tuned_x, fused_global_cond)
        coarsely_tuned_x = self.global_transform_3(coarsely_tuned_x, fused_global_cond)

        # 空间变换
        spatially_modulated_x = self.spatial_transform_1(coarsely_tuned_x, fused_spatial_cond)
        spatially_modulated_x = self.spatial_transform_2(spatially_modulated_x, fused_spatial_cond)
        spatially_modulated_x = self.spatial_transform_3(spatially_modulated_x, fused_spatial_cond)

        # 精细化
        if self.refinement:
            result = self.refinement(spatially_modulated_x)
        else:
            result = spatially_modulated_x

        self.mid_result = coarsely_tuned_x
        return result 

@ARCH_REGISTRY.register()
class HyCondITMv1_oneview_contra(nn.Module):
    def __init__(self, in_channels, transform_channels, global_cond_channels, spatial_cond_channels, refinement=True):
        super().__init__()

        # 全局条件变换模块
        self.global_transform_1 = ConditionedTransform(
            in_channels, transform_channels, global_cond_channels, 'global')
        self.global_transform_2 = ConditionedTransform(
            transform_channels, transform_channels, global_cond_channels, 'global')
        self.global_transform_3 = ConditionedTransform(
            transform_channels, in_channels, global_cond_channels, 'global', activation=False)

        # 空间条件变换模块
        self.spatial_transform_1 = ConditionedTransform(
            in_channels, transform_channels, spatial_cond_channels, 'spatial', ada_method='cbam')
        self.spatial_transform_2 = ConditionedTransform(
            transform_channels, transform_channels, spatial_cond_channels, 'spatial', ada_method='cbam')
        self.spatial_transform_3 = ConditionedTransform(
            transform_channels, in_channels, spatial_cond_channels, 'spatial', ada_method='vanilla', activation=False)

        # 精细化模块
        self.refinement = RefinementBlock(in_channels, in_channels, transform_channels) if refinement else None


    def forward(self, x, fused_global_cond, fused_spatial_cond):
        B, C, H, W = x.shape

        # 全局变换
        coarsely_tuned_x = self.global_transform_1(x, fused_global_cond)
        coarsely_tuned_x = self.global_transform_2(coarsely_tuned_x, fused_global_cond)
        coarsely_tuned_x = self.global_transform_3(coarsely_tuned_x, fused_global_cond)

        # 空间变换
        spatially_modulated_x = self.spatial_transform_1(coarsely_tuned_x, fused_spatial_cond)
        spatially_modulated_x = self.spatial_transform_2(spatially_modulated_x, fused_spatial_cond)
        spatially_modulated_x = self.spatial_transform_3(spatially_modulated_x, fused_spatial_cond)

        # 精细化
        if self.refinement:
            result = self.refinement(spatially_modulated_x)
        else:
            result = spatially_modulated_x

        self.mid_result = coarsely_tuned_x
        return result

@ARCH_REGISTRY.register()
class HyCondITMv1_woz(nn.Module):
    def __init__(self, in_channels, transform_channels, global_cond_channels, spatial_cond_channels, refinement=True):
        super().__init__()

        # 全局条件变换模块
        self.global_transform_1 = ConditionedTransform(
            in_channels, transform_channels, global_cond_channels, 'global')
        self.global_transform_2 = ConditionedTransform(
            transform_channels, transform_channels, global_cond_channels, 'global')
        self.global_transform_3 = ConditionedTransform(
            transform_channels, in_channels, global_cond_channels, 'global', activation=False)

        # 空间条件变换模块
        self.spatial_transform_1 = ConditionedTransform(
            in_channels, transform_channels, spatial_cond_channels, 'spatial', ada_method='cbam')
        self.spatial_transform_2 = ConditionedTransform(
            transform_channels, transform_channels, spatial_cond_channels, 'spatial', ada_method='cbam')
        self.spatial_transform_3 = ConditionedTransform(
            transform_channels, in_channels, spatial_cond_channels, 'spatial', ada_method='vanilla', activation=False)

        # 精细化模块
        self.refinement = RefinementBlock(in_channels, in_channels, transform_channels) if refinement else None

        # 生成全局和空间条件的模块
        self.global_condition_gen = nn.Sequential(
            nn.Conv2d(in_channels, global_cond_channels, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling
        )

        self.spatial_condition_gen = nn.Conv2d(in_channels, spatial_cond_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # 生成全局条件和空间条件
        fused_global_cond = self.global_condition_gen(x).view(B, -1)  # 将结果flatten
        fused_spatial_cond = self.spatial_condition_gen(x)  # 生成空间条件

        # 全局变换
        coarsely_tuned_x = self.global_transform_1(x, fused_global_cond)
        coarsely_tuned_x = self.global_transform_2(coarsely_tuned_x, fused_global_cond)
        coarsely_tuned_x = self.global_transform_3(coarsely_tuned_x, fused_global_cond)

        # 空间变换
        spatially_modulated_x = self.spatial_transform_1(coarsely_tuned_x, fused_spatial_cond)
        spatially_modulated_x = self.spatial_transform_2(spatially_modulated_x, fused_spatial_cond)
        spatially_modulated_x = self.spatial_transform_3(spatially_modulated_x, fused_spatial_cond)

        # 精细化
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
    input:  (N, in_channels, H, W)
    output: (N, out_channels, 2*H, 2*W)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        upscale_method='bilinear',
        align_corners=True,
        upsample_compute_dtype='auto',  # 'auto' | 'fp32' | 'keep'
    ):
        super().__init__()
        self.upscale_method = upscale_method
        self.align_corners = align_corners
        self.upsample_compute_dtype = upsample_compute_dtype

        if upscale_method == 'bilinear':
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=align_corners)
            self.up_conv = HyCondModConvBlock(in_channels, out_channels)
        elif upscale_method == 'transpose':
            self.up_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            raise NotImplementedError

        # cat 后通道数是 in_channels（x2） + out_channels（x1_up）
        self.conv = HyCondModConvBlock(in_channels, out_channels)

    def _cast_for_upsample(self, x: torch.Tensor):
        """让 upsample 在不支持的 dtype 上也能跑（重点是 bf16 / fp16 on CPU）。"""
        orig_dtype = x.dtype

        if self.upsample_compute_dtype == 'keep':
            return x, orig_dtype

        # 强制 fp32
        if self.upsample_compute_dtype == 'fp32':
            return x.float(), orig_dtype

        # auto：尽量只在“容易炸”的情况下 cast
        if x.device.type == 'cpu' and x.dtype in (torch.float16, torch.bfloat16):
            return x.float(), orig_dtype

        # 你遇到的典型：bf16 + bilinear interpolate
        if x.dtype == torch.bfloat16 and self.upscale_method == 'bilinear':
            return x.float(), orig_dtype

        return x, orig_dtype

    def forward(self, x1, x2):
        if self.upscale_method == 'bilinear':
            x1_cast, orig_dtype = self._cast_for_upsample(x1)
            x1_up = self.upsample(x1_cast)
            if x1_up.dtype != orig_dtype:
                x1_up = x1_up.to(orig_dtype)
            x1_up = self.up_conv(x1_up)
        else:  # transpose
            # transpose 一般在 GPU 上支持 bf16/fp16；CPU 不支持时也可以用同样的 auto cast
            x1_cast, orig_dtype = self._cast_for_upsample(x1)
            x1_up = self.up_transpose(x1_cast)
            if x1_up.dtype != orig_dtype:
                x1_up = x1_up.to(orig_dtype)

        diffY = x2.size(2) - x1_up.size(2)
        diffX = x2.size(3) - x1_up.size(3)

        x1_up = F.pad(x1_up, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1_up], dim=1)
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