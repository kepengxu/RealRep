"""
encoder refers to AirNet
repository: https://github.com/XLearning-SCU/2022-CVPR-AirNet/blob/main/net/encoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.img_util import dilator, thresholder
import kornia
import numpy as np
from torchvision import models
from .RealRep_fuse_zero_norm_arch import HybridConditionModule, HybridConditionModuleBase, HybridConditionModuleHigh, HybridConditionModuleSharedGlobal
from basicsr.utils.registry import ARCH_REGISTRY


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
        )
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.short_cut(x))

class FeatureExtract(nn.Module):
    def __init__(self, in_channels, out_channels, dim_in=128):
        super(FeatureExtract, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.resblock = nn.Sequential(
            ResBlock(out_channels, out_channels*2),  # 128 -> 64
            ResBlock(out_channels*2, out_channels*4),  # 64 -> 32
            ResBlock(out_channels*4, out_channels*8),   # 32 -> 16
        )
        self.mlp = nn.Sequential(
            nn.Linear(out_channels*8, dim_in),
            nn.LeakyReLU(0.1, True),
            nn.Linear(dim_in, dim_in)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock(out)
        out_feature = out
        out = self.avg_pool(out).squeeze(-1).squeeze(-1)  # 256-dimensional vector
        out_z = self.mlp(out)
        out_z = F.normalize(out_z, dim=-1)
        out_feature = F.normalize(out_feature, dim=1)  # [B, 128, 16, 16] (contrastive learning)
        return out_z, out_feature  # [B, D] shape, unit vectors of dimension D

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, in_channels, out_channels, dim, m=0.999, temperature=1.0):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.m = m
        self.temperature = temperature
        self.dim = dim

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = FeatureExtract(in_channels, out_channels, dim)
        self.encoder_k = FeatureExtract(in_channels, out_channels, dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
    def forward(self, im_q, im_k, im_negs):
        """
        输入:
            im_q: 查询图像的批次
            im_k: 关键图像的批次
            im_negs: 负样本图像的批次
        输出:
            logits, targets
        """
        # 提取负样本图像的形状信息，用于后续处理
        b, n_neg, c, h, w = im_negs.shape
        # 编码查询图像以获取其特征和归一化向量
        q, q_feat = self.encoder_q(im_q)
        # 获取查询特征的形状，以计算特征的分辨率
        _, fc, fh, fw = q_feat.shape  
        # 归一化查询向量
        q = nn.functional.normalize(q, dim=1)
        
        # 使用动量更新方法更新关键编码器参数，并处理负样本图像
        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_negs = im_negs.view(b * n_neg, c, h, w).contiguous()
            # 将关键图像和负样本图像拼接后通过关键编码器获取特征
            _, kn_feat = self.encoder_k(torch.cat([im_k, im_negs], dim=0))  # [b + b*n_neg, c, h, w]
            kn_feat = nn.functional.normalize(kn_feat, dim=1)
            k_feat = kn_feat[:b]
            # 重塑负样本特征的形状
            neg_feat = kn_feat[b:].view(b, n_neg, fc, fh, fw).contiguous()  # [b, n_neg, c, h, w]

        # 重塑查询特征、关键特征和负样本特征的形状
        q_feat = q_feat.view(b, fc * fh * fw).contiguous()
        k_feat = k_feat.view(b, fc * fh * fw).contiguous()
        neg_feat = neg_feat.view(b, n_neg, fc * fh * fw).contiguous()
        
        # 计算正样本和负样本的相似度得分
        l_pos = (q_feat * k_feat).sum(dim=-1, keepdims=True) / (fh * fw)
        l_neg = (q_feat.unsqueeze(1) * neg_feat).sum(dim=-1) / (fh * fw)
        
        # 合并正样本和负样本的相似度得分
        logits = torch.cat([l_pos, l_neg], dim=1)
        # 应用温度参数
        logits /= self.temperature
        # 生成标签，指示正样本的位置
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        return logits, labels, q
    # def forward(self, im_q, im_k, im_negs):
    #     """
    #     Input:
    #         im_q: a batch of query images
    #         im_k: a batch of key images
    #     Output:
    #         logits, targets
    #     """
    #     b, n_neg, c, h, w = im_negs.shape
    #     q, q_feat = self.encoder_q(im_q)
    #     _, fc, fh, fw = q_feat.shape  # resolution of features
    #     q = nn.functional.normalize(q, dim=1)
    #     with torch.no_grad():
    #         self._momentum_update_key_encoder()
    #         im_negs = im_negs.view(b*n_neg, c, h, w).contiguous()
    #         _, kn_feat = self.encoder_k(torch.cat([im_k, im_negs], dim=0))  # [b + b*n_neg, c, h, w]
    #         kn_feat = nn.functional.normalize(kn_feat, dim=1)
    #         k_feat = kn_feat[:b]
    #         neg_feat = kn_feat[b:].view(b, n_neg, fc, fh, fw).contiguous()  # [b, n_neg, c, h, w]

    #     q_feat = q_feat.view(b, fc*fh*fw).contiguous()
    #     k_feat = k_feat.view(b, fc*fh*fw).contiguous()
    #     neg_feat = neg_feat.view(b, n_neg, fc*fh*fw).contiguous()
    #     l_pos = (q_feat * k_feat).sum(dim=-1, keepdims=True) / (fh*fw)
    #     l_neg = (q_feat.unsqueeze(1) * neg_feat).sum(dim=-1) / (fh*fw)
    #     # logits: Nx(1+K)
    #     logits = torch.cat([l_pos, l_neg], dim=1)
    #     # apply temperature
    #     logits /= self.temperature
    #     # labels: positive key indicators
    #     labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    #     return logits, labels, q
    
class projection(nn.Module):
    """MLP projection head."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(projection, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)


class FeatureExtract_v2(nn.Module):
    def __init__(self, in_channels, out_channels, dim_in=128, dim_proj=128):
        super(FeatureExtract_v2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.resblock = nn.Sequential(
            ResBlock(out_channels, out_channels * 2),  # 128 -> 64
            ResBlock(out_channels * 2, out_channels * 4),  # 64 -> 32
            ResBlock(out_channels * 4, out_channels * 8),  # 32 -> 16
        )
        self.mlp = nn.Sequential(
            nn.Linear(out_channels * 8, dim_in),
            nn.LeakyReLU(0.1, True),
            nn.Linear(dim_in, dim_in)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # MLP projection head for contrastive learning
        self.proj_pool = nn.AdaptiveAvgPool2d(1)  # Pooling for `out_feature`
        self.projection_head = projection(out_channels * 8, out_channels * 8, dim_proj)

    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock(out)
        out_feature = out  # Raw feature map for contrastive learning
        out_feature_pooled = self.proj_pool(out_feature).squeeze(-1).squeeze(-1)  # [B, C]

        # Projection head for contrastive learning
        proj_feature = self.projection_head(out_feature_pooled)  # Apply MLP
        proj_feature = F.normalize(proj_feature, dim=-1)

        # Main task feature
        out = self.avg_pool(out).squeeze(-1).squeeze(-1)  # [B, C]
        out_z = self.mlp(out)
        out_z = F.normalize(out_z, dim=-1)

        return out_z, proj_feature  # Main task and contrastive features

class FeatureExtract_Post(nn.Module):
    def __init__(self, in_channels, out_channels, dim_in=128, dim_proj=128):
        super(FeatureExtract_Post, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.resblock = nn.Sequential(
            ResBlock(out_channels, out_channels * 2),  # 128 -> 64
            ResBlock(out_channels * 2, out_channels * 4),  # 64 -> 32
            ResBlock(out_channels * 4, out_channels * 8),  # 32 -> 16
        )

        # MLP projection head for contrastive learning
        self.proj_pool = nn.AdaptiveAvgPool2d(1)  # Pooling for `out_feature`
        self.projection_head = projection(out_channels * 8, out_channels * 8, dim_proj)

    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock(out)
        out_feature = out  # Raw feature map for contrastive learning
        out_feature_pooled = self.proj_pool(out_feature).squeeze(-1).squeeze(-1)  # [B, C]

        # Projection head for contrastive learning
        proj_feature = self.projection_head(out_feature_pooled)  # Apply MLP
        proj_feature = F.normalize(proj_feature, dim=-1)


        return proj_feature  # Main task and contrastive features


class MoCoV2(nn.Module):
    def __init__(self, in_channels, out_channels, dim, hidden_dim=512, m=0.999, temperature=1.0):
        """
        dim: feature dimension (default: 128)
        hidden_dim: hidden dimension of MLP projection head
        m: momentum for key encoder update
        temperature: softmax temperature for contrastive loss
        """
        super(MoCoV2, self).__init__()

        self.m = m
        self.temperature = temperature

        # Encoders
        self.encoder_q = FeatureExtract_v2(in_channels, out_channels, dim_in=dim, dim_proj=dim)
        self.encoder_k = FeatureExtract_v2(in_channels, out_channels, dim_in=dim, dim_proj=dim)

        # Initialize the key encoder with the same parameters as the query encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # Initialize
            param_k.requires_grad = False  # Do not update via gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder and its MLP head.
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(),
            self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def forward(self, im_q, im_k, im_negs):
        """
        输入:
            im_q: 查询图像的批次
            im_k: 关键图像的批次
            im_negs: 负样本图像的批次
        输出:
            logits, targets
        """
        b, n_neg, c, h, w = im_negs.shape
        q, q_proj = self.encoder_q(im_q)  # Main task feature and contrastive projection
        _, fc = q_proj.shape  # Projection dimension

        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_negs = im_negs.view(b * n_neg, c, h, w).contiguous()
            _, kn_proj = self.encoder_k(torch.cat([im_k, im_negs], dim=0))
            k_proj = kn_proj[:b]  # Positive key projections
            neg_proj = kn_proj[b:].view(b, n_neg, fc).contiguous()  # Negative projections

        # Compute logits
        l_pos = torch.einsum('bd,bd->b', [q_proj, k_proj]).unsqueeze(-1)
        l_neg = torch.einsum('bd,bnd->bn', [q_proj, neg_proj])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        # Labels
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return logits, labels, q

class FeatureExtract_Post(nn.Module):
    def __init__(self, in_channels, out_channels, dim=128, dim_proj=128):
        super(FeatureExtract_Post, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.resblock = nn.Sequential(
            ResBlock(out_channels, out_channels * 2),  # 128 -> 64
            ResBlock(out_channels * 2, out_channels * 4),  # 64 -> 32
            ResBlock(out_channels * 4, out_channels * 8),  # 32 -> 16
        )
        self.local_proj_q = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
        )

        # MLP projection head for contrastive learning
        self.proj_pool = nn.AdaptiveAvgPool2d(1)  # Pooling for `out_feature`
        self.projection_head = projection(out_channels * 8, out_channels * 8, dim_proj)

    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock(out)
        out_feature = out  # Raw feature map for contrastive learning
        out_feature_pooled = self.proj_pool(out_feature).squeeze(-1).squeeze(-1)  # [B, C]

        # Projection head for contrastive learning
        proj_feature = self.projection_head(out_feature_pooled)  # Apply MLP
        proj_feature = F.normalize(proj_feature, dim=-1)


        return proj_feature  # Main task and contrastive features


class FeatureExtract_Post(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureExtract_Post, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.resblock = nn.Sequential(
            ResBlock(out_channels, out_channels * 2),  # 128 -> 64
            ResBlock(out_channels * 2, out_channels * 4),  # 64 -> 32
            ResBlock(out_channels * 4, out_channels * 4),  # 32 -> 16
        )
        self.projection_head = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels * 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 4, out_channels * 4, 1),
        )


    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock(out)
        out_feature = out  # Raw feature map for contrastive learning

        # Project features for contrastive learning without pooling
        proj_feature = self.projection_head(out_feature)  # Apply MLP
        proj_feature = F.normalize(proj_feature, dim=1)  # Normalize features

        return proj_feature  # Return normalized features for similarity calculation
    

class MoCoV2_Post(nn.Module):
    def __init__(self, in_channels, out_channels, dim, hidden_dim=512, m=0.999, temperature=1.0):
        """
        dim: feature dimension (default: 128)
        hidden_dim: hidden dimension of MLP projection head
        m: momentum for key encoder update
        temperature: softmax temperature for contrastive loss
        """
        super(MoCoV2_Post, self).__init__()

        self.m = m
        self.temperature = temperature
        
        self.kernel = torch.ones(11,11).cuda()  

        # Encoders
        self.encoder_q = FeatureExtract_Post(in_channels, out_channels)
        self.encoder_k = FeatureExtract_Post(in_channels, out_channels)

        # Initialize the key encoder with the same parameters as the query encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # Initialize
            param_k.requires_grad = False  # Do not update via gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder and its MLP head.
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(),
            self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def forward(self, im_q, im_k, im_negs):
        """
        输入:
            im_q: 查询图像的批次
            im_k: 关键图像的批次
            im_negs: 负样本图像的批次
        输出:
            logits, targets
        """
        mask = thresholder(im_negs, 0.92) # M_sat
        gt_edges = kornia.filters.sobel(im_k, normalized=False) # M_edge    
        dilated_gt_edges = dilator(gt_edges, self.kernel)
        
        
        b, c, h, w = im_negs.shape
        q_proj = self.encoder_q(mask*dilated_gt_edges*im_q)  # Main task feature and contrastive projection
        q_proj = q_proj.view(b, -1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            # im_negs = im_negs.view(b * n_neg, c, h, w).contiguous()
            kn_proj = self.encoder_k(torch.cat([mask*dilated_gt_edges*im_k, mask*dilated_gt_edges*im_negs], dim=0))
            k_proj = kn_proj[:b].view(b, -1).contiguous()  # Positive key projections
            neg_proj = kn_proj[b:].view(b, 1, -1).contiguous()  # Negative projections

        
        # Compute logits
        l_pos = (q_proj * k_proj).sum(dim=-1, keepdim=True)  / (h * w) # Compute positive similarity
        l_neg = (q_proj.unsqueeze(1) * neg_proj).sum(dim=-1) / (h * w)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        # Labels
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return logits, labels

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1) 
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4) 
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class HighlightContrastLoss(nn.Module):
    def __init__(self, ablation=False, device=None):

        super(HighlightContrastLoss, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.vgg = Vgg19().to(self.device)
        self.l1 = nn.L1Loss().to(self.device)
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation
        self.kernel = torch.ones(11,11).cuda() 

    def forward(self, a, p, n):
        # a, p, n = a.to(self.device), p.to(self.device), n.to(self.device)
        a, p, n = [F.interpolate(x, size=(384, 384)) for x in [a, p, n]]
        
        # mask = thresholder(n, 0.92) # M_sat
        # gt_edges = kornia.filters.sobel(p, normalized=False) # M_edge    
        # dilated_gt_edges = dilator(gt_edges, self.kernel)
        # mask_dilated = mask * dilated_gt_edges
        
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss

class MoCoWithUNet(nn.Module):
    def __init__(self, m=0.999, temperature=1.0, in_channels=3, global_cond_channels=64, spatial_cond_channels=16):
        super(MoCoWithUNet, self).__init__()
        
        self.unet_q = HybridConditionModule(in_channels, spatial_cond_channels, global_cond_channels)
        self.unet_k = HybridConditionModule(in_channels, spatial_cond_channels, global_cond_channels)
        self.m = m
        self.temperature = temperature

        # 初始化动量更新
        self._initialize_momentum()

        # 投影头（查询编码器）
        self.global_proj_q = nn.Sequential(
            nn.Linear(global_cond_channels, global_cond_channels),
            nn.ReLU(inplace=True),
            nn.Linear(global_cond_channels, global_cond_channels),
        )
        self.local_proj_q = nn.Sequential(
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
        )

        # 投影头（关键编码器）
        self.global_proj_k = nn.Sequential(
            nn.Linear(global_cond_channels, global_cond_channels),
            nn.ReLU(inplace=True),
            nn.Linear(global_cond_channels, global_cond_channels),
        )
        self.local_proj_k = nn.Sequential(
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
        )
        
        # 初始化关键编码器的投影头
        self._initialize_momentum_projs()

    def _initialize_momentum(self):
        """
        初始化关键编码器（unet_k）的参数，使其与查询编码器（unet_q）相同，并且不参与梯度更新
        """
        for param_q, param_k in zip(self.unet_q.parameters(), self.unet_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def _initialize_momentum_projs(self):
        """
        初始化关键编码器的投影头（proj_k）的参数，使其与查询编码器的投影头（proj_q）相同，并且不参与梯度更新
        """
        for param_q, param_k in zip(self.global_proj_q.parameters(), self.global_proj_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.local_proj_q.parameters(), self.local_proj_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        动量更新关键编码器及其投影头
        """
        # 更新 unet_k
        for param_q, param_k in zip(self.unet_q.parameters(), self.unet_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data
        
        # 更新 global_proj_k
        for param_q, param_k in zip(self.global_proj_q.parameters(), self.global_proj_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data
        
        # 更新 local_proj_k
        for param_q, param_k in zip(self.local_proj_q.parameters(), self.local_proj_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

    @torch.no_grad()
    def reset_unet_k(self):
        """
        完全重新初始化关键编码器（unet_k），使其回到初始的随机初始化状态。
        不涉及投影头的重新初始化。
        """
        # 重新初始化 unet_k
        self.unet_k.apply(self._init_module)
        
        # 确保关键编码器参数不需要梯度
        for param_k in self.unet_k.parameters():
            param_k.requires_grad = False
            
    def _init_module(self, module):
        """
        初始化模块的参数。根据需要调整初始化方法。
        """
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
     # ===== Stage2 init helpers =====
    @torch.no_grad()
    def copy_q_to_k(self):
        """Force k <- q (unet_k/global_proj_k/local_proj_k all copy from q)."""
        # unet
        for pq, pk in zip(self.unet_q.parameters(), self.unet_k.parameters()):
            pk.data.copy_(pq.data)
            pk.requires_grad = False

        # global proj
        for pq, pk in zip(self.global_proj_q.parameters(), self.global_proj_k.parameters()):
            pk.data.copy_(pq.data)
            pk.requires_grad = False

        # local proj
        for pq, pk in zip(self.local_proj_q.parameters(), self.local_proj_k.parameters()):
            pk.data.copy_(pq.data)
            pk.requires_grad = False

    @torch.no_grad()
    def reinit_unet_k_only(self):
        """
        Re-initialize only unet_k (kaiming), keep proj_k unchanged.
        This matches your current reset_unet_k() behavior.
        """
        self.reset_unet_k()

    @torch.no_grad()
    def reinit_key_encoder_all(self):
        """
        Re-initialize unet_k + global_proj_k + local_proj_k.
        Useful to decouple 'backbone reinit' vs 'head mismatch' effects.
        """
        # reinit unet_k
        self.unet_k.apply(self._init_module)
        for pk in self.unet_k.parameters():
            pk.requires_grad = False

        # reinit global_proj_k
        self.global_proj_k.apply(self._init_module)
        for pk in self.global_proj_k.parameters():
            pk.requires_grad = False

        # reinit local_proj_k
        self.local_proj_k.apply(self._init_module)
        for pk in self.local_proj_k.parameters():
            pk.requires_grad = False

    @torch.no_grad()
    def stage2_init_key_encoder(self, mode: str = "keep"):
        """
        Stage2 initialization policy for key encoder.
        mode:
          - "keep": keep current k (EMA from stage1)
          - "copy_q": copy q -> k at stage2 start
          - "reinit_unet": reinit only unet_k (proj_k kept)
          - "reinit_all": reinit unet_k + global/local proj_k
        """
        mode = (mode or "keep").lower()
        if mode == "keep":
            return
        elif mode == "copy_q":
            self.copy_q_to_k()
        elif mode in ["reinit_unet", "reinit_unet_only", "reinit_backbone"]:
            self.reinit_unet_k_only()
        elif mode in ["reinit_all", "reinit_full"]:
            self.reinit_key_encoder_all()
        else:
            raise ValueError(f"Unknown stage2_init_k mode: {mode}")
    
    def forward(self, im_q, im_k, im_negs, mode='I'):
        b, n_neg, c, h, w = im_negs.shape

        # 查询编码器
        z_q_0, y_q_0 = self.unet_q(im_q)
        z_q = self.global_proj_q(z_q_0.squeeze(-1).squeeze(-1))
        z_q = F.normalize(z_q, dim=1) # b,c
        y_q = self.local_proj_q(y_q_0)
        y_q = F.normalize(y_q, dim=1) # b,c,h,w

        with torch.no_grad():
            self._momentum_update_key_encoder()

            # 关键编码器
            z_k, y_k = self.unet_k(im_k)
            z_k = self.global_proj_k(z_k.squeeze(-1).squeeze(-1))
            z_k = F.normalize(z_k, dim=1) # b,c
            y_k = self.local_proj_k(y_k)
            y_k = F.normalize(y_k, dim=1) # b,c,h,w

            # 负样本
            im_negs = im_negs.view(b * n_neg, c, h, w)
            z_negs, y_negs = self.unet_k(im_negs)
            z_negs = self.global_proj_k(z_negs.squeeze(-1).squeeze(-1))  # 全局负样本
            z_negs = F.normalize(z_negs, dim=1)
            z_negs = z_negs.view(b, n_neg, -1)
            y_negs = self.local_proj_k(y_negs)
            y_negs = F.normalize(y_negs, dim=1)
            y_negs = y_negs.view(b, n_neg, *y_negs.shape[1:])

        # 全局对比损失
        l_pos_global = (z_q * z_k).sum(dim=-1, keepdim=True)
        l_neg_global = torch.einsum('bd,bnd->bn', z_q, z_negs)
        logits_global = torch.cat([l_pos_global, l_neg_global], dim=1) / self.temperature
        labels_global = torch.zeros(logits_global.size(0), dtype=torch.long).cuda()

        # 局部对比损失
        l_pos_local = (y_q * y_k).mean(dim=(1, 2, 3)).unsqueeze(-1)
        l_neg_local = torch.einsum('bchw,bnchw->bn', y_q, y_negs) / (h * w)
        logits_local = torch.cat([l_pos_local, l_neg_local], dim=1) / self.temperature
        labels_local = torch.zeros(logits_local.size(0), dtype=torch.long).cuda()

        return logits_global, labels_global, logits_local, labels_local, z_q_0, y_q_0

@ARCH_REGISTRY.register()
class MoCoWithUNetDualHead(MoCoWithUNet):
    def __init__(self, m=0.999, temperature=1.0, in_channels=3, global_cond_channels=64, spatial_cond_channels=16):
        super().__init__(m=m, temperature=temperature, in_channels=in_channels, global_cond_channels=global_cond_channels, spatial_cond_channels=spatial_cond_channels)

        self.global_proj_q_TP = nn.Sequential(
            nn.Linear(global_cond_channels, global_cond_channels),
            nn.ReLU(inplace=True),
            nn.Linear(global_cond_channels, global_cond_channels),
        )
        self.local_proj_q_TP = nn.Sequential(
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
        )
        self.global_proj_k_TP = nn.Sequential(
            nn.Linear(global_cond_channels, global_cond_channels),
            nn.ReLU(inplace=True),
            nn.Linear(global_cond_channels, global_cond_channels),
        )
        self.local_proj_k_TP = nn.Sequential(
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
        )
        self._initialize_tp_momentum_projs()

    def _initialize_tp_momentum_projs(self):
        for param_q, param_k in zip(self.global_proj_q_TP.parameters(), self.global_proj_k_TP.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.local_proj_q_TP.parameters(), self.local_proj_k_TP.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        super()._momentum_update_key_encoder()
        for param_q, param_k in zip(self.global_proj_q_TP.parameters(), self.global_proj_k_TP.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data
        for param_q, param_k in zip(self.local_proj_q_TP.parameters(), self.local_proj_k_TP.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

    @torch.no_grad()
    def copy_q_to_k(self):
        super().copy_q_to_k()
        for param_q, param_k in zip(self.global_proj_q_TP.parameters(), self.global_proj_k_TP.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.local_proj_q_TP.parameters(), self.local_proj_k_TP.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def reinit_key_encoder_all(self):
        super().reinit_key_encoder_all()
        self.global_proj_k_TP.apply(self._init_module)
        for param_k in self.global_proj_k_TP.parameters():
            param_k.requires_grad = False
        self.local_proj_k_TP.apply(self._init_module)
        for param_k in self.local_proj_k_TP.parameters():
            param_k.requires_grad = False

    def _project_query_dual(self, z_0, y_0):
        z_I = F.normalize(self.global_proj_q(z_0.squeeze(-1).squeeze(-1)), dim=1)
        y_I = F.normalize(self.local_proj_q(y_0), dim=1)
        z_TP = F.normalize(self.global_proj_q_TP(z_0.squeeze(-1).squeeze(-1)), dim=1)
        y_TP = F.normalize(self.local_proj_q_TP(y_0), dim=1)
        return z_I, y_I, z_TP, y_TP

    def _project_key_dual(self, z_0, y_0):
        z_I = F.normalize(self.global_proj_k(z_0.squeeze(-1).squeeze(-1)), dim=1)
        y_I = F.normalize(self.local_proj_k(y_0), dim=1)
        z_TP = F.normalize(self.global_proj_k_TP(z_0.squeeze(-1).squeeze(-1)), dim=1)
        y_TP = F.normalize(self.local_proj_k_TP(y_0), dim=1)
        return z_I, y_I, z_TP, y_TP

    def _split_negative_outputs(self, z_negs, y_negs, num_neg_I):
        z_negs_I = z_negs[:, :num_neg_I, :]
        y_negs_I = y_negs[:, :num_neg_I, ...]
        z_negs_TP = z_negs[:, num_neg_I:, :]
        y_negs_TP = y_negs[:, num_neg_I:, ...]
        return z_negs_I, y_negs_I, z_negs_TP, y_negs_TP

    def _compute_logits(self, z_q, y_q, z_k, y_k, z_negs, y_negs):
        h, w = y_q.shape[-2:]
        logits_global = torch.cat([
            (z_q * z_k).sum(dim=-1, keepdim=True),
            torch.einsum('bd,bnd->bn', z_q, z_negs),
        ], dim=1) / self.temperature
        labels_global = torch.zeros(logits_global.size(0), dtype=torch.long, device=z_q.device)
        logits_local = torch.cat([
            (y_q * y_k).mean(dim=(1, 2, 3)).unsqueeze(-1),
            torch.einsum('bchw,bnchw->bn', y_q, y_negs) / (h * w),
        ], dim=1) / self.temperature
        labels_local = torch.zeros(logits_local.size(0), dtype=torch.long, device=z_q.device)
        return logits_global, labels_global, logits_local, labels_local

    def encode_q(self, im_q):
        return self.unet_q(im_q)

    def encode_k(self, im_k):
        return self.unet_k(im_k)

    def forward(self, im_q, im_k, im_negs_I=None, im_negs_TP=None):
        z_q_0, y_q_0 = self.unet_q(im_q)
        z_q_I, y_q_I, z_q_TP, y_q_TP = self._project_query_dual(z_q_0, y_q_0)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            z_k_0, y_k_0 = self.unet_k(im_k)
            z_k_I, y_k_I, z_k_TP, y_k_TP = self._project_key_dual(z_k_0, y_k_0)

            num_neg_I = 0 if im_negs_I is None else im_negs_I.shape[1]
            num_neg_TP = 0 if im_negs_TP is None else im_negs_TP.shape[1]
            if num_neg_I + num_neg_TP > 0:
                if num_neg_I == 0:
                    im_negs_all = im_negs_TP
                elif num_neg_TP == 0:
                    im_negs_all = im_negs_I
                else:
                    im_negs_all = torch.cat([im_negs_I, im_negs_TP], dim=1)

                b, n_neg, c, h, w = im_negs_all.shape
                z_negs_0, y_negs_0 = self.unet_k(im_negs_all.view(b * n_neg, c, h, w))
                z_negs_I_all, y_negs_I_all, z_negs_TP_all, y_negs_TP_all = self._project_key_dual(z_negs_0, y_negs_0)
                z_negs_I_all = z_negs_I_all.view(b, n_neg, -1)
                y_negs_I_all = y_negs_I_all.view(b, n_neg, *y_negs_I_all.shape[1:])
                z_negs_TP_all = z_negs_TP_all.view(b, n_neg, -1)
                y_negs_TP_all = y_negs_TP_all.view(b, n_neg, *y_negs_TP_all.shape[1:])

                z_negs_I, y_negs_I, _, _ = self._split_negative_outputs(z_negs_I_all, y_negs_I_all, num_neg_I)
                _, _, z_negs_TP, y_negs_TP = self._split_negative_outputs(z_negs_TP_all, y_negs_TP_all, num_neg_I)
            else:
                z_negs_I = z_k_I.unsqueeze(1)
                y_negs_I = y_k_I.unsqueeze(1)
                z_negs_TP = z_k_TP.unsqueeze(1)
                y_negs_TP = y_k_TP.unsqueeze(1)

            if num_neg_I == 0:
                z_negs_I = z_k_I.unsqueeze(1)
                y_negs_I = y_k_I.unsqueeze(1)
            if num_neg_TP == 0:
                z_negs_TP = z_k_TP.unsqueeze(1)
                y_negs_TP = y_k_TP.unsqueeze(1)

        logits_global_I, labels_global_I, logits_local_I, labels_local_I = self._compute_logits(z_q_I, y_q_I, z_k_I, y_k_I, z_negs_I, y_negs_I)
        logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP = self._compute_logits(z_q_TP, y_q_TP, z_k_TP, y_k_TP, z_negs_TP, y_negs_TP)
        return logits_global_I, labels_global_I, logits_local_I, labels_local_I, logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP, z_q_0, y_q_0


@ARCH_REGISTRY.register()
class MoCoWithUNetDualNeg(MoCoWithUNet):
    def _project_shared_query(self, z_0, y_0):
        z_q = F.normalize(self.global_proj_q(z_0.squeeze(-1).squeeze(-1)), dim=1)
        y_q = F.normalize(self.local_proj_q(y_0), dim=1)
        return z_q, y_q

    def _project_shared_key(self, z_0, y_0):
        z_k = F.normalize(self.global_proj_k(z_0.squeeze(-1).squeeze(-1)), dim=1)
        y_k = F.normalize(self.local_proj_k(y_0), dim=1)
        return z_k, y_k

    def _compute_logits(self, z_q, y_q, z_k, y_k, z_negs, y_negs):
        h, w = y_q.shape[-2:]
        logits_global = torch.cat([
            (z_q * z_k).sum(dim=-1, keepdim=True),
            torch.einsum('bd,bnd->bn', z_q, z_negs),
        ], dim=1) / self.temperature
        labels_global = torch.zeros(logits_global.size(0), dtype=torch.long, device=z_q.device)
        logits_local = torch.cat([
            (y_q * y_k).mean(dim=(1, 2, 3)).unsqueeze(-1),
            torch.einsum('bchw,bnchw->bn', y_q, y_negs) / (h * w),
        ], dim=1) / self.temperature
        labels_local = torch.zeros(logits_local.size(0), dtype=torch.long, device=z_q.device)
        return logits_global, labels_global, logits_local, labels_local

    def encode_q(self, im_q):
        return self.unet_q(im_q)

    def encode_k(self, im_k):
        return self.unet_k(im_k)

    def forward(self, im_q, im_k, im_negs_I=None, im_negs_TP=None):
        z_q_0, y_q_0 = self.unet_q(im_q)
        z_q, y_q = self._project_shared_query(z_q_0, y_q_0)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            z_k_0, y_k_0 = self.unet_k(im_k)
            z_k, y_k = self._project_shared_key(z_k_0, y_k_0)

            if im_negs_I is not None:
                b, n_neg_I, c, h, w = im_negs_I.shape
                z_negs_I_0, y_negs_I_0 = self.unet_k(im_negs_I.view(b * n_neg_I, c, h, w))
                z_negs_I, y_negs_I = self._project_shared_key(z_negs_I_0, y_negs_I_0)
                z_negs_I = z_negs_I.view(b, n_neg_I, -1)
                y_negs_I = y_negs_I.view(b, n_neg_I, *y_negs_I.shape[1:])
            else:
                z_negs_I = z_k.unsqueeze(1)
                y_negs_I = y_k.unsqueeze(1)

            if im_negs_TP is not None:
                b, n_neg_TP, c, h, w = im_negs_TP.shape
                z_negs_TP_0, y_negs_TP_0 = self.unet_k(im_negs_TP.view(b * n_neg_TP, c, h, w))
                z_negs_TP, y_negs_TP = self._project_shared_key(z_negs_TP_0, y_negs_TP_0)
                z_negs_TP = z_negs_TP.view(b, n_neg_TP, -1)
                y_negs_TP = y_negs_TP.view(b, n_neg_TP, *y_negs_TP.shape[1:])
            else:
                z_negs_TP = z_k.unsqueeze(1)
                y_negs_TP = y_k.unsqueeze(1)

        logits_global_I, labels_global_I, logits_local_I, labels_local_I = self._compute_logits(z_q, y_q, z_k, y_k, z_negs_I, y_negs_I)
        logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP = self._compute_logits(z_q, y_q, z_k, y_k, z_negs_TP, y_negs_TP)
        return logits_global_I, labels_global_I, logits_local_I, labels_local_I, logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP, z_q_0, y_q_0

class MoCoWithUNetlatetmix(nn.Module):
    def __init__(self, m=0.999, temperature=1.0, in_channels=3, global_cond_channels=64, spatial_cond_channels=16):
        super(MoCoWithUNetlatetmix, self).__init__()
        
        self.unet_q = HybridConditionModule(in_channels, spatial_cond_channels, global_cond_channels)
        self.unet_k = HybridConditionModule(in_channels, spatial_cond_channels, global_cond_channels)
        self.m = m
        self.temperature = temperature

        self.use_alpha_mix = True
        self.alpha_min = 0.2
        self.alpha_max = 0.8
        self.alpha_mode = "beta"
        self.alpha_beta = 0.5
        self.alpha_dominance = "rank"   # "none" or "rank"


        # 初始化动量更新
        self._initialize_momentum()

        # 投影头（查询编码器）
        self.global_proj_q = nn.Sequential(
            nn.Linear(global_cond_channels, global_cond_channels),
            nn.ReLU(inplace=True),
            nn.Linear(global_cond_channels, global_cond_channels),
        )
        self.local_proj_q = nn.Sequential(
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
        )

        # 投影头（关键编码器）
        self.global_proj_k = nn.Sequential(
            nn.Linear(global_cond_channels, global_cond_channels),
            nn.ReLU(inplace=True),
            nn.Linear(global_cond_channels, global_cond_channels),
        )
        self.local_proj_k = nn.Sequential(
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
        )
        
        # 初始化关键编码器的投影头
        self._initialize_momentum_projs()

    def _initialize_momentum(self):
        """
        初始化关键编码器（unet_k）的参数，使其与查询编码器（unet_q）相同，并且不参与梯度更新
        """
        for param_q, param_k in zip(self.unet_q.parameters(), self.unet_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def _initialize_momentum_projs(self):
        """
        初始化关键编码器的投影头（proj_k）的参数，使其与查询编码器的投影头（proj_q）相同，并且不参与梯度更新
        """
        for param_q, param_k in zip(self.global_proj_q.parameters(), self.global_proj_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.local_proj_q.parameters(), self.local_proj_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        动量更新关键编码器及其投影头
        """
        # 更新 unet_k
        for param_q, param_k in zip(self.unet_q.parameters(), self.unet_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data
        
        # 更新 global_proj_k
        for param_q, param_k in zip(self.global_proj_q.parameters(), self.global_proj_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data
        
        # 更新 local_proj_k
        for param_q, param_k in zip(self.local_proj_q.parameters(), self.local_proj_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

    @torch.no_grad()
    def reset_unet_k(self):
        """
        完全重新初始化关键编码器（unet_k），使其回到初始的随机初始化状态。
        不涉及投影头的重新初始化。
        """
        # 重新初始化 unet_k
        self.unet_k.apply(self._init_module)
        
        # 确保关键编码器参数不需要梯度
        for param_k in self.unet_k.parameters():
            param_k.requires_grad = False
            
    def _init_module(self, module):
        """
        初始化模块的参数。根据需要调整初始化方法。
        """
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    # ===== Stage2 init helpers =====
    @torch.no_grad()
    def copy_q_to_k(self):
        """Force k <- q (unet_k/global_proj_k/local_proj_k all copy from q)."""
        # unet
        for pq, pk in zip(self.unet_q.parameters(), self.unet_k.parameters()):
            pk.data.copy_(pq.data)
            pk.requires_grad = False

        # global proj
        for pq, pk in zip(self.global_proj_q.parameters(), self.global_proj_k.parameters()):
            pk.data.copy_(pq.data)
            pk.requires_grad = False

        # local proj
        for pq, pk in zip(self.local_proj_q.parameters(), self.local_proj_k.parameters()):
            pk.data.copy_(pq.data)
            pk.requires_grad = False

    @torch.no_grad()
    def reinit_unet_k_only(self):
        """
        Re-initialize only unet_k (kaiming), keep proj_k unchanged.
        This matches your current reset_unet_k() behavior.
        """
        self.reset_unet_k()

    @torch.no_grad()
    def reinit_key_encoder_all(self):
        """
        Re-initialize unet_k + global_proj_k + local_proj_k.
        Useful to decouple 'backbone reinit' vs 'head mismatch' effects.
        """
        # reinit unet_k
        self.unet_k.apply(self._init_module)
        for pk in self.unet_k.parameters():
            pk.requires_grad = False

        # reinit global_proj_k
        self.global_proj_k.apply(self._init_module)
        for pk in self.global_proj_k.parameters():
            pk.requires_grad = False

        # reinit local_proj_k
        self.local_proj_k.apply(self._init_module)
        for pk in self.local_proj_k.parameters():
            pk.requires_grad = False

    @torch.no_grad()
    def stage2_init_key_encoder(self, mode: str = "keep"):
        """
        Stage2 initialization policy for key encoder.
        mode:
          - "keep": keep current k (EMA from stage1)
          - "copy_q": copy q -> k at stage2 start
          - "reinit_unet": reinit only unet_k (proj_k kept)
          - "reinit_all": reinit unet_k + global/local proj_k
        """
        mode = (mode or "keep").lower()
        if mode == "keep":
            return
        elif mode == "copy_q":
            self.copy_q_to_k()
        elif mode in ["reinit_unet", "reinit_unet_only", "reinit_backbone"]:
            self.reinit_unet_k_only()
        elif mode in ["reinit_all", "reinit_full"]:
            self.reinit_key_encoder_all()
        else:
            raise ValueError(f"Unknown stage2_init_k mode: {mode}")
    
    def _alpha_mix(self, z_k, y_k, z_negs, y_negs,
                alpha_min=0.2, alpha_max=0.8,
                alpha_mode="beta", alpha_beta=0.5,
                neg_ranks=None, dominance="none"):
        """
        z_k:    [B,Cg]
        z_negs: [B,N,Cg]
        y_k:    [B,Cs,H,W]
        y_negs: [B,N,Cs,H,W]
        neg_ranks: [B,N] optional, ranks (0=hardest) from dataset/mining
        dominance: "none" | "rank"
        return: z_mix [B,N,Cg], y_mix [B,N,Cs,H,W]
        """
        B, N, Cg = z_negs.shape

        # ---- sample alpha in float32 to avoid bf16 corner ops ----
        if alpha_mode == "beta":
            # symmetric Beta(alpha_beta, alpha_beta), smaller -> more mass near 0/1
            a = float(alpha_beta)
            alpha = np.random.beta(a, a, size=(B, N, 1)).astype(np.float32)
            alpha = torch.from_numpy(alpha).to(z_negs.device)
            alpha = alpha * (alpha_max - alpha_min) + alpha_min
        elif alpha_mode == "uniform":
            alpha = torch.rand((B, N, 1), device=z_negs.device, dtype=torch.float32)
            alpha = alpha * (alpha_max - alpha_min) + alpha_min
        else:
            alpha = torch.full((B, N, 1), (alpha_min + alpha_max) * 0.5,
                            device=z_negs.device, dtype=torch.float32)

        # ---- dominance-aware alpha (推荐用 rank 做，0成本) ----
        # rank 越靠前(更hard)，alpha 越小 -> mix 更保留原始 hard negative
        if dominance == "rank" and neg_ranks is not None:
            # neg_ranks: [B,N] int/float
            r = neg_ranks.float().clamp_min(0.0)
            # normalize to [0,1], 0=hardest, 1=easiest
            r_norm = r / (r.max(dim=1, keepdim=True).values + 1e-6)  # [B,N]
            # map to [alpha_min, alpha_max]: hardest -> alpha_min, easiest -> alpha_max
            alpha = alpha_min + (alpha_max - alpha_min) * r_norm.unsqueeze(-1)

        # ---- mix (broadcast) ----
        z_k_exp = z_k.unsqueeze(1)  # [B,1,Cg]
        z_mix = alpha * z_k_exp + (1.0 - alpha) * z_negs
        z_mix = F.normalize(z_mix, dim=-1)

        # local alpha broadcast to [B,N,1,1,1]
        alpha_y = alpha.view(B, N, 1, 1, 1)
        y_k_exp = y_k.unsqueeze(1)  # [B,1,Cs,H,W]
        y_mix = alpha_y * y_k_exp + (1.0 - alpha_y) * y_negs
        y_mix = F.normalize(y_mix, dim=2)

        return z_mix, y_mix
    
    def forward(self, im_q, im_k, im_negs, neg_ranks=None):
        """
        im_q:    [B,3,H,W]
        im_k:    [B,3,H,W]
        im_negs: [B,N,3,H,W] or [B,3,H,W] or None
        neg_ranks: [B,N] optional, int ranks in the fullrank list (0=hardest)

        return:
        logits_global, labels_global, logits_local, labels_local, z_q_0, y_q_0
        """
        device = im_q.device

        # ---- parse negatives shape robustly ----
        if im_negs is None:
            # no negatives -> cannot compute InfoNCE properly
            # fallback: create dummy negatives by using im_k as 1 negative (won't be meaningful but keeps shape)
            # Better: ensure dataset always supplies negatives.
            im_negs = im_k.unsqueeze(1)  # [B,1,3,H,W]

        if im_negs.dim() == 4:
            # treat as one negative
            im_negs = im_negs.unsqueeze(1)  # [B,1,3,H,W]

        assert im_negs.dim() == 5, f"im_negs must be 5D [B,N,C,H,W], got {im_negs.shape}"
        b, n_neg, c, h, w = im_negs.shape

        # ---- query encoder ----
        z_q_0, y_q_0 = self.unet_q(im_q)  # z_q_0: [B,Cg,1,1] or [B,Cg,?,?] depending on module
        z_q = self.global_proj_q(z_q_0.squeeze(-1).squeeze(-1))  # [B,Cg]
        z_q = F.normalize(z_q, dim=1)

        y_q = self.local_proj_q(y_q_0)  # [B,Cs,H,W]
        y_q = F.normalize(y_q, dim=1)

        with torch.no_grad():
            # ---- momentum update key encoder & proj heads ----
            self._momentum_update_key_encoder()

            # ---- key encoder ----
            z_k_0, y_k_0 = self.unet_k(im_k)
            z_k = self.global_proj_k(z_k_0.squeeze(-1).squeeze(-1))
            z_k = F.normalize(z_k, dim=1)  # [B,Cg]
            y_k = self.local_proj_k(y_k_0)
            y_k = F.normalize(y_k, dim=1)  # [B,Cs,H,W]

            # ---- negative encoder ----
            im_negs_flat = im_negs.view(b * n_neg, c, h, w)
            z_negs_0, y_negs_0 = self.unet_k(im_negs_flat)
            z_negs = self.global_proj_k(z_negs_0.squeeze(-1).squeeze(-1))
            z_negs = F.normalize(z_negs, dim=1)
            z_negs = z_negs.view(b, n_neg, -1)  # [B,N,Cg]

            y_negs = self.local_proj_k(y_negs_0)
            y_negs = F.normalize(y_negs, dim=1)
            y_negs = y_negs.view(b, n_neg, *y_negs.shape[1:])  # [B,N,Cs,H,W]

            # ------------------------------------------------------------
            # Rank-aware α-mix negatives (latent mixing)
            # ------------------------------------------------------------
            use_alpha_mix = getattr(self, "use_alpha_mix", True)
            alpha_mode = getattr(self, "alpha_mode", "beta")            # "beta" or "uniform" or "fixed"
            alpha_beta = float(getattr(self, "alpha_beta", 0.5))        # Beta(a,a)
            alpha_min = float(getattr(self, "alpha_min", 0.2))
            alpha_max = float(getattr(self, "alpha_max", 0.8))
            alpha_dominance = getattr(self, "alpha_dominance", "rank")  # "none" or "rank"
            alpha_mix_policy = getattr(self, "alpha_mix_policy", "replace")  # "concat" or "replace"

            if use_alpha_mix and n_neg > 0:
                # sample alpha in float32 to avoid bf16 issues
                if alpha_mode == "beta":
                    a = max(alpha_beta, 1e-6)
                    alpha_np = np.random.beta(a, a, size=(b, n_neg, 1)).astype(np.float32)
                    alpha = torch.from_numpy(alpha_np).to(device=device, dtype=torch.float32)
                    alpha = alpha * (alpha_max - alpha_min) + alpha_min
                elif alpha_mode == "uniform":
                    alpha = torch.rand((b, n_neg, 1), device=device, dtype=torch.float32)
                    alpha = alpha * (alpha_max - alpha_min) + alpha_min
                else:
                    alpha = torch.full((b, n_neg, 1), (alpha_min + alpha_max) * 0.5,
                                    device=device, dtype=torch.float32)

                # ---- rank-aware: harder neg (smaller rank) -> smaller alpha -> preserve hard negatives ----
                if alpha_dominance == "rank" and (neg_ranks is not None):
                    # neg_ranks expected [B,N], 0=hardest
                    if neg_ranks.dim() == 1:
                        neg_ranks = neg_ranks.view(b, n_neg)
                    neg_ranks_f = neg_ranks.to(device=device, dtype=torch.float32).clamp_min(0.0)

                    # normalize ranks to [0,1] per sample, avoid division by zero
                    rmax = neg_ranks_f.max(dim=1, keepdim=True).values + 1e-6
                    r_norm = (neg_ranks_f / rmax).unsqueeze(-1)  # [B,N,1], 0 hard -> small, 1 easy -> large

                    alpha = alpha_min + (alpha_max - alpha_min) * r_norm

                # mix in projected space
                z_k_exp = z_k.unsqueeze(1)                         # [B,1,Cg]
                z_mix = alpha * z_k_exp + (1.0 - alpha) * z_negs   # [B,N,Cg]
                z_mix = F.normalize(z_mix, dim=-1)

                alpha_y = alpha.view(b, n_neg, 1, 1, 1)            # [B,N,1,1,1]
                y_k_exp = y_k.unsqueeze(1)                         # [B,1,Cs,H,W]
                y_mix = alpha_y * y_k_exp + (1.0 - alpha_y) * y_negs
                y_mix = F.normalize(y_mix, dim=2)

                if alpha_mix_policy == "replace":
                    z_negs_all, y_negs_all = z_mix, y_mix
                else:
                    # concat -> 2N negatives
                    z_negs_all = torch.cat([z_negs, z_mix], dim=1)  # [B,2N,Cg]
                    y_negs_all = torch.cat([y_negs, y_mix], dim=1)  # [B,2N,Cs,H,W]
            else:
                z_negs_all, y_negs_all = z_negs, y_negs

        # ---- Global InfoNCE ----
        l_pos_global = (z_q * z_k).sum(dim=-1, keepdim=True)  # [B,1]
        l_neg_global = torch.einsum('bd,bnd->bn', z_q, z_negs_all)  # [B,Nneg]
        logits_global = torch.cat([l_pos_global, l_neg_global], dim=1) / self.temperature
        labels_global = torch.zeros(logits_global.size(0), dtype=torch.long, device=device)

        # ---- Local InfoNCE ----
        # use mean similarity as in your original
        l_pos_local = (y_q * y_k).mean(dim=(1, 2, 3)).unsqueeze(-1)  # [B,1]
        # y_negs_all: [B,N,Cs,H,W]
        l_neg_local = torch.einsum('bchw,bnchw->bn', y_q, y_negs_all) / (h * w)
        logits_local = torch.cat([l_pos_local, l_neg_local], dim=1) / self.temperature
        labels_local = torch.zeros(logits_local.size(0), dtype=torch.long, device=device)

        return logits_global, labels_global, logits_local, labels_local, z_q_0, y_q_0






@ARCH_REGISTRY.register()
class MoCoWithUNetDualNegRank(MoCoWithUNetDualNeg):
    def forward(self, im_q, im_k, im_negs_I=None, im_negs_TP=None, neg_ranks_I=None, neg_ranks_TP=None):
        _ = neg_ranks_I, neg_ranks_TP
        z_q_0, y_q_0 = self.unet_q(im_q)
        z_q, y_q = self._project_shared_query(z_q_0, y_q_0)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            z_k_0, y_k_0 = self.unet_k(im_k)
            z_k, y_k = self._project_shared_key(z_k_0, y_k_0)

            if im_negs_I is not None:
                b, n_neg_I, c, h, w = im_negs_I.shape
                z_negs_I_0, y_negs_I_0 = self.unet_k(im_negs_I.view(b * n_neg_I, c, h, w))
                z_negs_I, y_negs_I = self._project_shared_key(z_negs_I_0, y_negs_I_0)
                z_negs_I = z_negs_I.view(b, n_neg_I, -1)
                y_negs_I = y_negs_I.view(b, n_neg_I, *y_negs_I.shape[1:])
            else:
                z_negs_I = z_k.unsqueeze(1)
                y_negs_I = y_k.unsqueeze(1)

            if im_negs_TP is not None:
                b, n_neg_TP, c, h, w = im_negs_TP.shape
                z_negs_TP_0, y_negs_TP_0 = self.unet_k(im_negs_TP.view(b * n_neg_TP, c, h, w))
                z_negs_TP, y_negs_TP = self._project_shared_key(z_negs_TP_0, y_negs_TP_0)
                z_negs_TP = z_negs_TP.view(b, n_neg_TP, -1)
                y_negs_TP = y_negs_TP.view(b, n_neg_TP, *y_negs_TP.shape[1:])
            else:
                z_negs_TP = z_k.unsqueeze(1)
                y_negs_TP = y_k.unsqueeze(1)

        logits_global_I, labels_global_I, logits_local_I, labels_local_I = self._compute_logits(z_q, y_q, z_k, y_k, z_negs_I, y_negs_I)
        logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP = self._compute_logits(z_q, y_q, z_k, y_k, z_negs_TP, y_negs_TP)
        return logits_global_I, labels_global_I, logits_local_I, labels_local_I, logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP, z_q_0, y_q_0


@ARCH_REGISTRY.register()
class MoCoWithUNetDualNegLatentMix(MoCoWithUNetDualNegRank):
    def __init__(
        self,
        m=0.999,
        temperature=1.0,
        in_channels=3,
        global_cond_channels=64,
        spatial_cond_channels=16,
        alpha_min=0.2,
        alpha_max=0.8,
        alpha_mode='beta',
        alpha_beta=0.5,
        alpha_dominance='rank',
        alpha_mix_policy='replace',
        alpha_rank_reverse=False,
        preserve_topk_luma=0,
        preserve_topk_chroma=0,
        use_alpha_mix=True,
    ):
        super().__init__(
            m=m,
            temperature=temperature,
            in_channels=in_channels,
            global_cond_channels=global_cond_channels,
            spatial_cond_channels=spatial_cond_channels,
        )
        self.use_alpha_mix = use_alpha_mix
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_mode = alpha_mode
        self.alpha_beta = alpha_beta
        self.alpha_dominance = alpha_dominance
        self.alpha_mix_policy = alpha_mix_policy
        self.alpha_rank_reverse = alpha_rank_reverse
        self.preserve_topk_luma = int(preserve_topk_luma)
        self.preserve_topk_chroma = int(preserve_topk_chroma)

    def _mix_ranked_negs(self, z_k, y_k, z_negs, y_negs, neg_ranks=None, preserve_topk=0):
        b, n_neg, _ = z_negs.shape
        device = z_negs.device
        if (not self.use_alpha_mix) or n_neg == 0:
            return z_negs, y_negs

        if self.alpha_mode == 'beta':
            a = max(float(self.alpha_beta), 1e-6)
            alpha_np = np.random.beta(a, a, size=(b, n_neg, 1)).astype(np.float32)
            alpha = torch.tensor(alpha_np, device=device, dtype=torch.float32)
            alpha = alpha * (self.alpha_max - self.alpha_min) + self.alpha_min
        elif self.alpha_mode == 'uniform':
            alpha = torch.rand((b, n_neg, 1), device=device, dtype=torch.float32)
            alpha = alpha * (self.alpha_max - self.alpha_min) + self.alpha_min
        else:
            alpha = torch.full(
                (b, n_neg, 1),
                (self.alpha_min + self.alpha_max) * 0.5,
                device=device,
                dtype=torch.float32,
            )

        if self.alpha_dominance == 'rank' and neg_ranks is not None:
            if neg_ranks.dim() == 1:
                neg_ranks = neg_ranks.view(b, n_neg)
            neg_ranks_f = neg_ranks.to(device=device, dtype=torch.float32).clamp_min(0.0)
            rmax = neg_ranks_f.max(dim=1, keepdim=True).values + 1e-6
            r_norm = (neg_ranks_f / rmax).unsqueeze(-1)
            if self.alpha_rank_reverse:
                # Reversed direction: larger rank -> smaller alpha.
                alpha = self.alpha_max - (self.alpha_max - self.alpha_min) * r_norm
            else:
                # Default direction: larger rank -> larger alpha.
                alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * r_norm

        z_mix = alpha * z_k.unsqueeze(1) + (1.0 - alpha) * z_negs
        z_mix = F.normalize(z_mix, dim=-1)

        alpha_y = alpha.view(b, n_neg, 1, 1, 1)
        y_mix = alpha_y * y_k.unsqueeze(1) + (1.0 - alpha_y) * y_negs
        y_mix = F.normalize(y_mix, dim=2)

        if self.alpha_mix_policy == 'replace':
            return z_mix, y_mix
        if self.alpha_mix_policy == 'partialconcat':
            keep_from = max(int(preserve_topk), 0)
            z_mix_keep = z_mix[:, keep_from:, :]
            y_mix_keep = y_mix[:, keep_from:, ...]
            return torch.cat([z_negs, z_mix_keep], dim=1), torch.cat([y_negs, y_mix_keep], dim=1)
        if self.alpha_mix_policy == 'concat_plus_negneg':
            z_negs_pair = torch.roll(z_negs, shifts=-2, dims=1)
            y_negs_pair = torch.roll(y_negs, shifts=-2, dims=1)
            z_negneg_mix = alpha * z_negs + (1.0 - alpha) * z_negs_pair
            z_negneg_mix = F.normalize(z_negneg_mix, dim=-1)
            y_negneg_mix = alpha_y * y_negs + (1.0 - alpha_y) * y_negs_pair
            y_negneg_mix = F.normalize(y_negneg_mix, dim=2)
            return torch.cat([z_negs, z_mix, z_negneg_mix], dim=1), torch.cat([y_negs, y_mix, y_negneg_mix], dim=1)
        return torch.cat([z_negs, z_mix], dim=1), torch.cat([y_negs, y_mix], dim=1)

    def forward(self, im_q, im_k, im_negs_I=None, im_negs_TP=None, neg_ranks_I=None, neg_ranks_TP=None):
        z_q_0, y_q_0 = self.unet_q(im_q)
        z_q, y_q = self._project_shared_query(z_q_0, y_q_0)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            z_k_0, y_k_0 = self.unet_k(im_k)
            z_k, y_k = self._project_shared_key(z_k_0, y_k_0)

            if im_negs_I is not None:
                b, n_neg_I, c, h, w = im_negs_I.shape
                z_negs_I_0, y_negs_I_0 = self.unet_k(im_negs_I.view(b * n_neg_I, c, h, w))
                z_negs_I, y_negs_I = self._project_shared_key(z_negs_I_0, y_negs_I_0)
                z_negs_I = z_negs_I.view(b, n_neg_I, -1)
                y_negs_I = y_negs_I.view(b, n_neg_I, *y_negs_I.shape[1:])
                z_negs_I, y_negs_I = self._mix_ranked_negs(
                    z_k, y_k, z_negs_I, y_negs_I,
                    neg_ranks=neg_ranks_I,
                    preserve_topk=self.preserve_topk_luma,
                )
            else:
                z_negs_I = z_k.unsqueeze(1)
                y_negs_I = y_k.unsqueeze(1)

            if im_negs_TP is not None:
                b, n_neg_TP, c, h, w = im_negs_TP.shape
                z_negs_TP_0, y_negs_TP_0 = self.unet_k(im_negs_TP.view(b * n_neg_TP, c, h, w))
                z_negs_TP, y_negs_TP = self._project_shared_key(z_negs_TP_0, y_negs_TP_0)
                z_negs_TP = z_negs_TP.view(b, n_neg_TP, -1)
                y_negs_TP = y_negs_TP.view(b, n_neg_TP, *y_negs_TP.shape[1:])
                z_negs_TP, y_negs_TP = self._mix_ranked_negs(
                    z_k, y_k, z_negs_TP, y_negs_TP,
                    neg_ranks=neg_ranks_TP,
                    preserve_topk=self.preserve_topk_chroma,
                )
            else:
                z_negs_TP = z_k.unsqueeze(1)
                y_negs_TP = y_k.unsqueeze(1)

        logits_global_I, labels_global_I, logits_local_I, labels_local_I = self._compute_logits(z_q, y_q, z_k, y_k, z_negs_I, y_negs_I)
        logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP = self._compute_logits(z_q, y_q, z_k, y_k, z_negs_TP, y_negs_TP)
        return logits_global_I, labels_global_I, logits_local_I, labels_local_I, logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP, z_q_0, y_q_0


@ARCH_REGISTRY.register()
class MoCoWithUNetDualNegLatentMixQ(MoCoWithUNetDualNegLatentMix):
    def forward(self, im_q, im_k, im_negs_I=None, im_negs_TP=None, neg_ranks_I=None, neg_ranks_TP=None):
        z_q_0, y_q_0 = self.unet_q(im_q)
        z_q, y_q = self._project_shared_query(z_q_0, y_q_0)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            z_k_0, y_k_0 = self.unet_k(im_k)
            z_k, y_k = self._project_shared_key(z_k_0, y_k_0)

            if im_negs_I is not None:
                b, n_neg_I, c, h, w = im_negs_I.shape
                z_negs_I_0, y_negs_I_0 = self.unet_k(im_negs_I.view(b * n_neg_I, c, h, w))
                z_negs_I, y_negs_I = self._project_shared_key(z_negs_I_0, y_negs_I_0)
                z_negs_I = z_negs_I.view(b, n_neg_I, -1)
                y_negs_I = y_negs_I.view(b, n_neg_I, *y_negs_I.shape[1:])
                z_negs_I, y_negs_I = self._mix_ranked_negs(
                    z_q.detach(), y_q.detach(), z_negs_I, y_negs_I,
                    neg_ranks=neg_ranks_I,
                    preserve_topk=self.preserve_topk_luma,
                )
            else:
                z_negs_I = z_k.unsqueeze(1)
                y_negs_I = y_k.unsqueeze(1)

            if im_negs_TP is not None:
                b, n_neg_TP, c, h, w = im_negs_TP.shape
                z_negs_TP_0, y_negs_TP_0 = self.unet_k(im_negs_TP.view(b * n_neg_TP, c, h, w))
                z_negs_TP, y_negs_TP = self._project_shared_key(z_negs_TP_0, y_negs_TP_0)
                z_negs_TP = z_negs_TP.view(b, n_neg_TP, -1)
                y_negs_TP = y_negs_TP.view(b, n_neg_TP, *y_negs_TP.shape[1:])
                z_negs_TP, y_negs_TP = self._mix_ranked_negs(
                    z_q.detach(), y_q.detach(), z_negs_TP, y_negs_TP,
                    neg_ranks=neg_ranks_TP,
                    preserve_topk=self.preserve_topk_chroma,
                )
            else:
                z_negs_TP = z_k.unsqueeze(1)
                y_negs_TP = y_k.unsqueeze(1)

        logits_global_I, labels_global_I, logits_local_I, labels_local_I = self._compute_logits(z_q, y_q, z_k, y_k, z_negs_I, y_negs_I)
        logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP = self._compute_logits(z_q, y_q, z_k, y_k, z_negs_TP, y_negs_TP)
        return logits_global_I, labels_global_I, logits_local_I, labels_local_I, logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP, z_q_0, y_q_0


class MoCoWithUNetSharedGlobal(MoCoWithUNet):
    pass

# class MoCoWithUNetSharedGlobal(nn.Module):
#     def __init__(self, m=0.999, temperature=1.0, in_channels=3, global_cond_channels=64, spatial_cond_channels=16):
#         super(MoCoWithUNetSharedGlobal, self).__init__()

#         self.unet_q = HybridConditionModuleSharedGlobal(in_channels, spatial_cond_channels, global_cond_channels)
#         self.unet_k = HybridConditionModuleSharedGlobal(in_channels, spatial_cond_channels, global_cond_channels)
#         self.m = m
#         self.temperature = temperature

#         self.use_alpha_mix = True
#         self.alpha_min = 0.2
#         self.alpha_max = 0.8
#         self.alpha_mode = "beta"
#         self.alpha_beta = 0.5
#         self.alpha_dominance = "rank"

#         self._initialize_momentum()

#         self.global_proj_q_I = nn.Sequential(
#             nn.Linear(global_cond_channels, global_cond_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(global_cond_channels, global_cond_channels),
#         )
#         self.global_proj_q_TP = nn.Sequential(
#             nn.Linear(global_cond_channels, global_cond_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(global_cond_channels, global_cond_channels),
#         )
#         self.global_proj_k_I = nn.Sequential(
#             nn.Linear(global_cond_channels, global_cond_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(global_cond_channels, global_cond_channels),
#         )
#         self.global_proj_k_TP = nn.Sequential(
#             nn.Linear(global_cond_channels, global_cond_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(global_cond_channels, global_cond_channels),
#         )

#         self.local_proj_q_I = nn.Sequential(
#             nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
#         )
#         self.local_proj_q_TP = nn.Sequential(
#             nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
#         )
#         self.local_proj_k_I = nn.Sequential(
#             nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
#         )
#         self.local_proj_k_TP = nn.Sequential(
#             nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
#         )

#         self._initialize_momentum_projs()

#     def _global_proj_q(self, branch):
#         if branch == 'I':
#             return self.global_proj_q_I
#         if branch == 'TP':
#             return self.global_proj_q_TP
#         raise ValueError(f"Unknown branch: {branch}")

#     def _global_proj_k(self, branch):
#         if branch == 'I':
#             return self.global_proj_k_I
#         if branch == 'TP':
#             return self.global_proj_k_TP
#         raise ValueError(f"Unknown branch: {branch}")

#     def _local_proj_q(self, branch):
#         if branch == 'I':
#             return self.local_proj_q_I
#         if branch == 'TP':
#             return self.local_proj_q_TP
#         raise ValueError(f"Unknown branch: {branch}")

#     def _local_proj_k(self, branch):
#         if branch == 'I':
#             return self.local_proj_k_I
#         if branch == 'TP':
#             return self.local_proj_k_TP
#         raise ValueError(f"Unknown branch: {branch}")

#     def _initialize_momentum(self):
#         for param_q, param_k in zip(self.unet_q.parameters(), self.unet_k.parameters()):
#             param_k.data.copy_(param_q.data)
#             param_k.requires_grad = False

#     def _initialize_momentum_projs(self):
#         for param_q, param_k in zip(self.global_proj_q_I.parameters(), self.global_proj_k_I.parameters()):
#             param_k.data.copy_(param_q.data)
#             param_k.requires_grad = False

#         for param_q, param_k in zip(self.global_proj_q_TP.parameters(), self.global_proj_k_TP.parameters()):
#             param_k.data.copy_(param_q.data)
#             param_k.requires_grad = False

#         for param_q, param_k in zip(self.local_proj_q_I.parameters(), self.local_proj_k_I.parameters()):
#             param_k.data.copy_(param_q.data)
#             param_k.requires_grad = False

#         for param_q, param_k in zip(self.local_proj_q_TP.parameters(), self.local_proj_k_TP.parameters()):
#             param_k.data.copy_(param_q.data)
#             param_k.requires_grad = False

#     @torch.no_grad()
#     def _momentum_update_key_encoder(self):
#         for param_q, param_k in zip(self.unet_q.parameters(), self.unet_k.parameters()):
#             param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

#         for param_q, param_k in zip(self.global_proj_q_I.parameters(), self.global_proj_k_I.parameters()):
#             param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

#         for param_q, param_k in zip(self.global_proj_q_TP.parameters(), self.global_proj_k_TP.parameters()):
#             param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

#         for param_q, param_k in zip(self.local_proj_q_I.parameters(), self.local_proj_k_I.parameters()):
#             param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

#         for param_q, param_k in zip(self.local_proj_q_TP.parameters(), self.local_proj_k_TP.parameters()):
#             param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

#     @torch.no_grad()
#     def reset_unet_k(self):
#         self.unet_k.apply(self._init_module)
#         for param_k in self.unet_k.parameters():
#             param_k.requires_grad = False

#     def _init_module(self, module):
#         if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#             nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
#             if module.bias is not None:
#                 nn.init.constant_(module.bias, 0)
#         elif isinstance(module, nn.BatchNorm2d):
#             nn.init.constant_(module.weight, 1)
#             nn.init.constant_(module.bias, 0)

#     @torch.no_grad()
#     def copy_q_to_k(self):
#         for pq, pk in zip(self.unet_q.parameters(), self.unet_k.parameters()):
#             pk.data.copy_(pq.data)
#             pk.requires_grad = False

#         for pq, pk in zip(self.global_proj_q_I.parameters(), self.global_proj_k_I.parameters()):
#             pk.data.copy_(pq.data)
#             pk.requires_grad = False

#         for pq, pk in zip(self.global_proj_q_TP.parameters(), self.global_proj_k_TP.parameters()):
#             pk.data.copy_(pq.data)
#             pk.requires_grad = False

#         for pq, pk in zip(self.local_proj_q_I.parameters(), self.local_proj_k_I.parameters()):
#             pk.data.copy_(pq.data)
#             pk.requires_grad = False

#         for pq, pk in zip(self.local_proj_q_TP.parameters(), self.local_proj_k_TP.parameters()):
#             pk.data.copy_(pq.data)
#             pk.requires_grad = False

#     @torch.no_grad()
#     def reinit_unet_k_only(self):
#         self.reset_unet_k()

#     @torch.no_grad()
#     def reinit_key_encoder_all(self):
#         self.unet_k.apply(self._init_module)
#         for pk in self.unet_k.parameters():
#             pk.requires_grad = False

#         self.global_proj_k_I.apply(self._init_module)
#         for pk in self.global_proj_k_I.parameters():
#             pk.requires_grad = False

#         self.global_proj_k_TP.apply(self._init_module)
#         for pk in self.global_proj_k_TP.parameters():
#             pk.requires_grad = False

#         self.local_proj_k_I.apply(self._init_module)
#         for pk in self.local_proj_k_I.parameters():
#             pk.requires_grad = False

#         self.local_proj_k_TP.apply(self._init_module)
#         for pk in self.local_proj_k_TP.parameters():
#             pk.requires_grad = False

#     @torch.no_grad()
#     def stage2_init_key_encoder(self, mode: str = "keep"):
#         mode = (mode or "keep").lower()
#         if mode == "keep":
#             return
#         elif mode == "copy_q":
#             self.copy_q_to_k()
#         elif mode in ["reinit_unet", "reinit_unet_only", "reinit_backbone"]:
#             self.reinit_unet_k_only()
#         elif mode in ["reinit_all", "reinit_full"]:
#             self.reinit_key_encoder_all()
#         else:
#             raise ValueError(f"Unknown stage2_init_k mode: {mode}")

#     def forward(self, im_q, im_k, im_negs, branch='I', neg_ranks=None):
#         device = im_q.device

#         if im_negs is None:
#             im_negs = im_k.unsqueeze(1)
#         if im_negs.dim() == 4:
#             im_negs = im_negs.unsqueeze(1)

#         assert im_negs.dim() == 5, f"im_negs must be 5D [B,N,C,H,W], got {im_negs.shape}"
#         b, n_neg, c, h, w = im_negs.shape

#         global_proj_q = self._global_proj_q(branch)
#         global_proj_k = self._global_proj_k(branch)
#         local_proj_q = self._local_proj_q(branch)
#         local_proj_k = self._local_proj_k(branch)

#         z_q_0, y_q_0 = self.unet_q(im_q, branch=branch)
#         z_q = global_proj_q(z_q_0.squeeze(-1).squeeze(-1))
#         z_q = F.normalize(z_q, dim=1)
#         y_q = local_proj_q(y_q_0)
#         y_q = F.normalize(y_q, dim=1)

#         with torch.no_grad():
#             self._momentum_update_key_encoder()

#             z_k_0, y_k_0 = self.unet_k(im_k, branch=branch)
#             z_k = global_proj_k(z_k_0.squeeze(-1).squeeze(-1))
#             z_k = F.normalize(z_k, dim=1)
#             y_k = local_proj_k(y_k_0)
#             y_k = F.normalize(y_k, dim=1)

#             im_negs_flat = im_negs.view(b * n_neg, c, h, w)
#             z_negs_0, y_negs_0 = self.unet_k(im_negs_flat, branch=branch)
#             z_negs = global_proj_k(z_negs_0.squeeze(-1).squeeze(-1))
#             z_negs = F.normalize(z_negs, dim=1)
#             z_negs = z_negs.view(b, n_neg, -1)

#             y_negs = local_proj_k(y_negs_0)
#             y_negs = F.normalize(y_negs, dim=1)
#             y_negs = y_negs.view(b, n_neg, *y_negs.shape[1:])

#             use_alpha_mix = getattr(self, "use_alpha_mix", True)
#             alpha_mode = getattr(self, "alpha_mode", "beta")
#             alpha_beta = float(getattr(self, "alpha_beta", 0.5))
#             alpha_min = float(getattr(self, "alpha_min", 0.2))
#             alpha_max = float(getattr(self, "alpha_max", 0.8))
#             alpha_dominance = getattr(self, "alpha_dominance", "rank")
#             alpha_mix_policy = getattr(self, "alpha_mix_policy", "replace")

#             if use_alpha_mix and n_neg > 0:
#                 if alpha_mode == "beta":
#                     a = max(alpha_beta, 1e-6)
#                     alpha_np = np.random.beta(a, a, size=(b, n_neg, 1)).astype(np.float32)
#                     alpha = torch.from_numpy(alpha_np).to(device=device, dtype=torch.float32)
#                     alpha = alpha * (alpha_max - alpha_min) + alpha_min
#                 elif alpha_mode == "uniform":
#                     alpha = torch.rand((b, n_neg, 1), device=device, dtype=torch.float32)
#                     alpha = alpha * (alpha_max - alpha_min) + alpha_min
#                 else:
#                     alpha = torch.full((b, n_neg, 1), (alpha_min + alpha_max) * 0.5, device=device, dtype=torch.float32)

#                 if alpha_dominance == "rank" and (neg_ranks is not None):
#                     if neg_ranks.dim() == 1:
#                         neg_ranks = neg_ranks.view(b, n_neg)
#                     neg_ranks_f = neg_ranks.to(device=device, dtype=torch.float32).clamp_min(0.0)
#                     rmax = neg_ranks_f.max(dim=1, keepdim=True).values + 1e-6
#                     r_norm = (neg_ranks_f / rmax).unsqueeze(-1)
#                     w_hard = 1.0 - r_norm
#                     alpha = alpha_min + (alpha_max - alpha_min) * w_hard

#                 z_k_exp = z_k.unsqueeze(1)
#                 z_mix = alpha * z_k_exp + (1.0 - alpha) * z_negs
#                 z_mix = F.normalize(z_mix, dim=-1)

#                 alpha_y = alpha.view(b, n_neg, 1, 1, 1)
#                 y_k_exp = y_k.unsqueeze(1)
#                 y_mix = alpha_y * y_k_exp + (1.0 - alpha_y) * y_negs
#                 y_mix = F.normalize(y_mix, dim=2)

#                 if alpha_mix_policy == "replace":
#                     z_negs_all, y_negs_all = z_mix, y_mix
#                 else:
#                     z_negs_all = torch.cat([z_negs, z_mix], dim=1)
#                     y_negs_all = torch.cat([y_negs, y_mix], dim=1)
#             else:
#                 z_negs_all, y_negs_all = z_negs, y_negs

#         l_pos_global = (z_q * z_k).sum(dim=-1, keepdim=True)
#         l_neg_global = torch.einsum('bd,bnd->bn', z_q, z_negs_all)
#         logits_global = torch.cat([l_pos_global, l_neg_global], dim=1) / self.temperature
#         labels_global = torch.zeros(logits_global.size(0), dtype=torch.long, device=device)

#         l_pos_local = (y_q * y_k).mean(dim=(1, 2, 3)).unsqueeze(-1)
#         l_neg_local = torch.einsum('bchw,bnchw->bn', y_q, y_negs_all) / (h * w)
#         logits_local = torch.cat([l_pos_local, l_neg_local], dim=1) / self.temperature
#         labels_local = torch.zeros(logits_local.size(0), dtype=torch.long, device=device)

#         return logits_global, labels_global, logits_local, labels_local, z_q_0, y_q_0

# class MoCoWithUNetHigh(nn.Module):
#     def __init__(self, shared_base, m=0.999, temperature=1.0, global_cond_channels=64, spatial_cond_channels=16):
#         super(MoCoWithUNetHigh, self).__init__()

#         self.shared_base = shared_base  # 共享的基网络
        
#         # 定义独立的高层网络 for query encoder
#         self.unet_q = HybridConditionModuleHigh(shared_base.out_channels, global_cond_channels, spatial_cond_channels)
        
#         # 定义独立的高层网络 for key encoder
#         self.unet_k = HybridConditionModuleHigh(shared_base.out_channels, global_cond_channels, spatial_cond_channels)
        
#         self.m = m
#         self.temperature = temperature

#         # 初始化动量更新
#         for param_q, param_k in zip(self.unet_q.parameters(), self.unet_k.parameters()):
#             param_k.data.copy_(param_q.data)
#             param_k.requires_grad = False

#         # 投影头（查询编码器）
#         self.global_proj_q = nn.Sequential(
#             nn.Linear(global_cond_channels, global_cond_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(global_cond_channels, global_cond_channels),
#         )
#         self.local_proj_q = nn.Sequential(
#             nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
#         )

#         # 投影头（关键编码器）
#         self.global_proj_k = nn.Sequential(
#             nn.Linear(global_cond_channels, global_cond_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(global_cond_channels, global_cond_channels),
#         )
#         self.local_proj_k = nn.Sequential(
#             nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
#         )

#     @torch.no_grad()
#     def _momentum_update_key_encoder(self):
#         """
#         动量更新关键编码器
#         """
#         for param_q, param_k in zip(self.unet_q.parameters(), self.unet_k.parameters()):
#             param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

#     def forward(self, im_q, im_k, im_negs):
#         b, n_neg, c, h, w = im_negs.shape

#         # 查询编码器
#         z_q_0, y_q_0 = self.unet_q(im_q)
#         z_q = self.global_proj_q(z_q_0.squeeze(-1).squeeze(-1))
#         z_q = F.normalize(z_q, dim=1) # b,c
#         y_q = self.local_proj_q(y_q_0)
#         y_q = F.normalize(y_q, dim=1) # b,c,h,w

#         with torch.no_grad():
#             self._momentum_update_key_encoder()

#             # 关键编码器
#             z_k, y_k = self.unet_k(im_k)
#             z_k = self.global_proj_k(z_k.squeeze(-1).squeeze(-1))
#             z_k = F.normalize(z_k, dim=1) # b,c
#             y_k = self.local_proj_k(y_k)
#             y_k = F.normalize(y_k, dim=1) # b,c,h,w

#             # 负样本
#             im_negs = im_negs.view(b * n_neg, c, h, w)
#             z_negs, y_negs = self.unet_k(im_negs)
#             z_negs = self.global_proj_k(z_negs.squeeze(-1).squeeze(-1))  # 全局负样本
#             z_negs = F.normalize(z_negs, dim=1)
#             z_negs = z_negs.view(b, n_neg, -1)
#             y_negs = self.local_proj_k(y_negs)
#             y_negs = F.normalize(y_negs, dim=1)
#             y_negs = y_negs.view(b, n_neg, *y_negs.shape[1:])

#         # 全局对比损失
#         l_pos_global = (z_q * z_k).sum(dim=-1, keepdim=True)
#         l_neg_global = torch.einsum('bd,bnd->bn', z_q, z_negs)
#         logits_global = torch.cat([l_pos_global, l_neg_global], dim=1) / self.temperature
#         labels_global = torch.zeros(logits_global.size(0), dtype=torch.long).cuda()

#         # 局部对比损失
#         l_pos_local = (y_q * y_k).mean(dim=(1, 2, 3)).unsqueeze(-1)
#         l_neg_local = torch.einsum('bchw,bnchw->bn', y_q, y_negs) / (h * w)
#         logits_local = torch.cat([l_pos_local, l_neg_local], dim=1) / self.temperature
#         labels_local = torch.zeros(logits_local.size(0), dtype=torch.long).cuda()

#         return logits_global, labels_global, logits_local, labels_local, z_q_0, y_q_0