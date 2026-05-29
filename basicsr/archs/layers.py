import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers, **kwargs):
    layers = []
    for _ in range(n_layers):
        layers.append(block(**kwargs))
    return nn.Sequential(*layers)


# global cond 
def csrnet_condition_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
    )

class Conditioncompression(nn.Module): 
    def __init__(self, in_channels=3, n_features=32):
        super().__init__()

        self.condition = nn.Sequential(
            *csrnet_condition_block(in_channels, n_features, 7, stride=2, padding=1),
            *csrnet_condition_block(n_features, n_features, 3, stride=2, padding=1),
            *csrnet_condition_block(n_features, n_features, 3, stride=2, padding=1),
        )
    def forward(self, x):
        return self.condition(x)  # (N, n_features, 1, 1)

# hdrtvnet c128 condition c64 
class GlobalConditionBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, n_features=32):
        super().__init__()

        self.cond = nn.Sequential(
            csrnet_condition_block(in_channels, n_features, 7, stride=2, padding=1),
            nn.Dropout(p=0.5),
            csrnet_condition_block(n_features, out_channels, 1, stride=1, padding=0),
            nn.AdaptiveMaxPool2d(1)
        )
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.cond(x)
        return x

class GlobalConditionBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, n_features=32):
        super().__init__()

        self.cond = nn.Sequential(
            csrnet_condition_block(in_channels, n_features, 7, stride=2, padding=1),
            csrnet_condition_block(n_features, n_features, 5, stride=2, padding=1),
            nn.Dropout(p=0.5),
            csrnet_condition_block(n_features, out_channels, 1, stride=1, padding=0),
            nn.AdaptiveMaxPool2d(1)
        )
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.cond(x)
        return x

class GlobalConditionBlock3(nn.Module):
    def __init__(self, in_channels, out_channels, n_features=32):
        super().__init__()

        self.cond = nn.Sequential(
            csrnet_condition_block(in_channels, n_features, 7, stride=2, padding=1),
            csrnet_condition_block(n_features, n_features, 5, stride=2, padding=1),
            csrnet_condition_block(n_features, n_features, 3, stride=2, padding=1),
            nn.Dropout(p=0.5),
            csrnet_condition_block(n_features, out_channels, 1, stride=1, padding=0),
            nn.AdaptiveMaxPool2d(1)
        )
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.cond(x)
        return x

# class GlobalFeatureModulation(nn.Module): # 全局调制
#     def __init__(self, cond_channels, n_features, residual=True):
#         super().__init__()
#         self.cond_scale1 = GlobalConditionBlock(in_channels, out_channels, n_features)
#         self.cond_shift1 = GlobalConditionBlock(cond_channels, n_features, 1, stride=1)
#         self.cond_scale2 = GlobalConditionBlock(cond_channels, n_features, 1, stride=1)
#         self.n_features = n_features
#         self.residual = residual

#     def forward(self, cond):
#         b,c,h,w = cond.shape
#         cond_vec = cond#.squeeze(-1).squeeze(-1)  # (N, cond_channels)
#         scale1 = self.cond_scale1(cond_vec).view(b, self.n_features, h, w)  # (N, n_features, 1, 1)
#         shift1 = self.cond_shift1(cond_vec).view(b, self.n_features, h, w)  # (N, n_features, 1, 1)
        
#         scale2 = self.cond_scale2(cond_vec).view(b, self.n_features, h, w)
#         # print(x.shape, scale.shape, shift.shape)
#         # out = x * scale + shift

#         return scale1, shift1, scale2

class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        

    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1) # 待定
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear") # 有问题
        prompt = self.conv3x3(prompt)

        return prompt
    

class FeatureTransform(nn.Module):
    def __init__(self, dim, cond_channels):
        super(FeatureTransform, self).__init__()
        self.modulation_conv0 = nn.Conv2d(cond_channels, cond_channels, 1, 1, 0)
        self.modulation_conv1 = nn.Conv2d(cond_channels, 3*dim, 1, 1, 0)
        # self.num_heads = num_heads
        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        # self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        # self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x,cond):
        b, c, h, w = x.shape
        B, C, H, W = cond.shape
        cond = self.modulation_conv1(F.relu(self.modulation_conv0(cond)))  # (B, 3*C, 1, 1)

        # 分离 shift, scale, gate
        shift, scale, gate = cond.chunk(3, dim=1)  # 每个部分的形状 (B*D, C, 1, 1)
      
        x = x + gate*(scale+1)+shift  

        # qkv = self.qkv_dwconv(self.qkv(x))
        # q,k,v = qkv.chunk(3, dim=1)   
        
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # q = torch.nn.functional.normalize(q, dim=-1)
        # k = torch.nn.functional.normalize(k, dim=-1)

        # attn = (q @ k.transpose(-2, -1)) * self.temperature
        # attn = attn.softmax(dim=-1)

        # out = (attn @ v)
        
        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # out = self.project_out(out)
        return x

    

class ResBlock_with_GSFT(nn.Module):
    def __init__(self, dim, global_channels, spatial_channels, nf=64):
        super(ResBlock_with_GSFT, self).__init__()

        self.gft = FeatureTransform(dim, global_channels)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.sft = FeatureTransform(dim, spatial_channels)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        cond1, cond2 = x[1], x[2]
        x = x[0]
        b,c,h,w = x.shape
        # x = x.view(-1, c, h, w)
        fea = self.gft(x, cond1)
        fea = F.relu(self.conv1(fea), inplace=True)
        fea = self.sft(fea, cond2)
        fea = self.conv2(fea)
        return (x + fea, cond1, cond2)

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

class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h*w)

        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        output = output.contiguous().view(b, c, h, w)

        return output
    
class InputProj(nn.Module):
    """Video input projection

    Args:
        in_channels (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of output channels. Default: 32.
        kernel_size (int): Size of the convolution kernel. Default: 3
        stride (int): Stride of the convolution. Default: 1
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        act_layer (nn.Module): Activation layer. Default: nn.LeakyReLU.
    """
    def __init__(self, in_channels=3, embed_dim=32, kernel_size=3, stride=1, 
                 act_layer=nn.LeakyReLU):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, 
                      stride=stride, padding=kernel_size//2),
            act_layer
        )

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): (B, D, C, H, W)

        Returns:
            torch.Tensor: (B, D, C, H, W)
        """
        B, C, H, W = x.shape
        x = self.proj(x) # B, C, H, W
        return x
    
    
class CondProj(nn.Module):
    """Video input projection

    Args:
        in_channels (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of output channels. Default: 32.
        kernel_size (int): Size of the convolution kernel. Default: 3
        stride (int): Stride of the convolution. Default: 1
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        act_layer (nn.Module): Activation layer. Default: nn.LeakyReLU.
    """
    def __init__(self, cond_in_nc=3, cond_nf=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(cond_in_nc, cond_nf, 3, 1, 1), nn.LeakyReLU(0.1, True), 
            nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True), 
            nn.Conv2d(cond_nf, cond_nf, 1), nn.LeakyReLU(0.1, True)
        )

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): (B, D, C, H, W)

        Returns:
            torch.Tensor: (B, D, C, H, W)
        """
        B, C, H, W = x.shape
        x = self.proj(x) # B, C, H, W
        return x

class Downsample(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): (B, D, C, H, W)

        Returns:
            torch.Tensor: (B, D, C, H, W)
        """
        B, C, H, W = x.shape
        out = self.conv(x).view(B, -1, H // 2, W // 2)  # B, D, C, H, W
        return out

class Upsample(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans*4, 3, 1, 1), nn.PixelShuffle(2)
        )

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): (B, D, C, H, W)

        Returns:
            torch.Tensor: (B, D, C, H, W)
        """
        B, C, H, W = x. shape
        out = self.deconv(x).view(B, -1, H * 2, W * 2) # B, D, C, H, W
        return out
