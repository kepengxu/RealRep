"""
encoder refers to AirNet
repository: https://github.com/XLearning-SCU/2022-CVPR-AirNet/blob/main/net/encoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.img_util import dilator, thresholder
import kornia
from torchvision import models
from .RealRep_fuse_zero_norm_arch import HybridConditionModule, HybridConditionModuleBase, HybridConditionModuleHigh


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

        b, n_neg, c, h, w = im_negs.shape
        q, q_feat = self.encoder_q(im_q)
        _, fc, fh, fw = q_feat.shape  
        q = nn.functional.normalize(q, dim=1)
        
        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_negs = im_negs.view(b * n_neg, c, h, w).contiguous()
            _, kn_feat = self.encoder_k(torch.cat([im_k, im_negs], dim=0))  # [b + b*n_neg, c, h, w]
            kn_feat = nn.functional.normalize(kn_feat, dim=1)
            k_feat = kn_feat[:b]
            neg_feat = kn_feat[b:].view(b, n_neg, fc, fh, fw).contiguous()  # [b, n_neg, c, h, w]

        q_feat = q_feat.view(b, fc * fh * fw).contiguous()
        k_feat = k_feat.view(b, fc * fh * fw).contiguous()
        neg_feat = neg_feat.view(b, n_neg, fc * fh * fw).contiguous()
        
        l_pos = (q_feat * k_feat).sum(dim=-1, keepdims=True) / (fh * fw)
        l_neg = (q_feat.unsqueeze(1) * neg_feat).sum(dim=-1) / (fh * fw)
        
        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        return logits, labels, q
    
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

        self._initialize_momentum()

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
        
        self._initialize_momentum_projs()

    def _initialize_momentum(self):
        for param_q, param_k in zip(self.unet_q.parameters(), self.unet_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def _initialize_momentum_projs(self):
        for param_q, param_k in zip(self.global_proj_q.parameters(), self.global_proj_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.local_proj_q.parameters(), self.local_proj_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.unet_q.parameters(), self.unet_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data
        
        for param_q, param_k in zip(self.global_proj_q.parameters(), self.global_proj_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data
        
        for param_q, param_k in zip(self.local_proj_q.parameters(), self.local_proj_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

    @torch.no_grad()
    def reset_unet_k(self):
        self.unet_k.apply(self._init_module)
        
        for param_k in self.unet_k.parameters():
            param_k.requires_grad = False
            
    def _init_module(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, im_q, im_k, im_negs, mode='I'):
        b, n_neg, c, h, w = im_negs.shape

        z_q_0, y_q_0 = self.unet_q(im_q)
        z_q = self.global_proj_q(z_q_0.squeeze(-1).squeeze(-1))
        z_q = F.normalize(z_q, dim=1) # b,c
        y_q = self.local_proj_q(y_q_0)
        y_q = F.normalize(y_q, dim=1) # b,c,h,w

        with torch.no_grad():
            self._momentum_update_key_encoder()

            z_k, y_k = self.unet_k(im_k)
            z_k = self.global_proj_k(z_k.squeeze(-1).squeeze(-1))
            z_k = F.normalize(z_k, dim=1) # b,c
            y_k = self.local_proj_k(y_k)
            y_k = F.normalize(y_k, dim=1) # b,c,h,w

            im_negs = im_negs.view(b * n_neg, c, h, w)
            z_negs, y_negs = self.unet_k(im_negs)
            z_negs = self.global_proj_k(z_negs.squeeze(-1).squeeze(-1))  # 全局负样本
            z_negs = F.normalize(z_negs, dim=1)
            z_negs = z_negs.view(b, n_neg, -1)
            y_negs = self.local_proj_k(y_negs)
            y_negs = F.normalize(y_negs, dim=1)
            y_negs = y_negs.view(b, n_neg, *y_negs.shape[1:])

        l_pos_global = (z_q * z_k).sum(dim=-1, keepdim=True)
        l_neg_global = torch.einsum('bd,bnd->bn', z_q, z_negs)
        logits_global = torch.cat([l_pos_global, l_neg_global], dim=1) / self.temperature
        labels_global = torch.zeros(logits_global.size(0), dtype=torch.long).cuda()

        l_pos_local = (y_q * y_k).mean(dim=(1, 2, 3)).unsqueeze(-1)
        l_neg_local = torch.einsum('bchw,bnchw->bn', y_q, y_negs) / (h * w)
        logits_local = torch.cat([l_pos_local, l_neg_local], dim=1) / self.temperature
        labels_local = torch.zeros(logits_local.size(0), dtype=torch.long).cuda()

        return logits_global, labels_global, logits_local, labels_local, z_q_0, y_q_0
    

class MoCoWithUNet_woglobal(nn.Module):
    def __init__(self, m=0.999, temperature=1.0, in_channels=3, global_cond_channels=64, spatial_cond_channels=16):
        super(MoCoWithUNet_woglobal, self).__init__()
        
        self.unet_q = HybridConditionModule_woglobal(in_channels, spatial_cond_channels, global_cond_channels)
        self.unet_k = HybridConditionModule_woglobal(in_channels, spatial_cond_channels, global_cond_channels)
        self.m = m
        self.temperature = temperature

        self._initialize_momentum()

        # self.global_proj_q = nn.Sequential(
        #     nn.Linear(global_cond_channels, global_cond_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(global_cond_channels, global_cond_channels),
        # )
        self.local_proj_q = nn.Sequential(
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
        )

        # self.global_proj_k = nn.Sequential(
        #     nn.Linear(global_cond_channels, global_cond_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(global_cond_channels, global_cond_channels),
        # )
        self.local_proj_k = nn.Sequential(
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
        )
        
        self._initialize_momentum_projs()

    def _initialize_momentum(self):
        for param_q, param_k in zip(self.unet_q.parameters(), self.unet_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def _initialize_momentum_projs(self):
        # for param_q, param_k in zip(self.global_proj_q.parameters(), self.global_proj_k.parameters()):
        #     param_k.data.copy_(param_q.data)
        #     param_k.requires_grad = False

        for param_q, param_k in zip(self.local_proj_q.parameters(), self.local_proj_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.unet_q.parameters(), self.unet_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data
        
        # for param_q, param_k in zip(self.global_proj_q.parameters(), self.global_proj_k.parameters()):
        #     param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data
        
        for param_q, param_k in zip(self.local_proj_q.parameters(), self.local_proj_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

    @torch.no_grad()
    def reset_unet_k(self):
        self.unet_k.apply(self._init_module)
        
        for param_k in self.unet_k.parameters():
            param_k.requires_grad = False
            
    def _init_module(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, im_q, im_k, im_negs):
        b, n_neg, c, h, w = im_negs.shape

        y_q_0 = self.unet_q(im_q)
        # z_q = self.global_proj_q(z_q_0.squeeze(-1).squeeze(-1))
        # z_q = F.normalize(z_q, dim=1) # b,c
        y_q = self.local_proj_q(y_q_0)
        y_q = F.normalize(y_q, dim=1) # b,c,h,w

        with torch.no_grad():
            self._momentum_update_key_encoder()

            y_k = self.unet_k(im_k)
            # z_k = self.global_proj_k(z_k.squeeze(-1).squeeze(-1))
            # z_k = F.normalize(z_k, dim=1) # b,c
            y_k = self.local_proj_k(y_k)
            y_k = F.normalize(y_k, dim=1) # b,c,h,w

            im_negs = im_negs.view(b * n_neg, c, h, w)
            y_negs = self.unet_k(im_negs)
            # z_negs = self.global_proj_k(z_negs.squeeze(-1).squeeze(-1))  # 全局负样本
            # z_negs = F.normalize(z_negs, dim=1)
            # z_negs = z_negs.view(b, n_neg, -1)
            y_negs = self.local_proj_k(y_negs)
            y_negs = F.normalize(y_negs, dim=1)
            y_negs = y_negs.view(b, n_neg, *y_negs.shape[1:])

        # l_pos_global = (z_q * z_k).sum(dim=-1, keepdim=True)
        # l_neg_global = torch.einsum('bd,bnd->bn', z_q, z_negs)
        # logits_global = torch.cat([l_pos_global, l_neg_global], dim=1) / self.temperature
        # labels_global = torch.zeros(logits_global.size(0), dtype=torch.long).cuda()

        l_pos_local = (y_q * y_k).mean(dim=(1, 2, 3)).unsqueeze(-1)
        l_neg_local = torch.einsum('bchw,bnchw->bn', y_q, y_negs) / (h * w)
        logits_local = torch.cat([l_pos_local, l_neg_local], dim=1) / self.temperature
        labels_local = torch.zeros(logits_local.size(0), dtype=torch.long).cuda()

        return logits_local, labels_local, y_q_0
    

class MoCoWithUNet_wolocal(nn.Module):
    def __init__(self, m=0.999, temperature=1.0, in_channels=3, global_cond_channels=64, spatial_cond_channels=16):
        super(MoCoWithUNet_wolocal, self).__init__()
        
        self.unet_q = HybridConditionModule_wolocal(in_channels, spatial_cond_channels, global_cond_channels)
        self.unet_k = HybridConditionModule_wolocal(in_channels, spatial_cond_channels, global_cond_channels)
        self.m = m
        self.temperature = temperature

        self._initialize_momentum()

        self.global_proj_q = nn.Sequential(
            nn.Linear(global_cond_channels, global_cond_channels),
            nn.ReLU(inplace=True),
            nn.Linear(global_cond_channels, global_cond_channels),
        )
        # self.local_proj_q = nn.Sequential(
        #     nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
        # )

        self.global_proj_k = nn.Sequential(
            nn.Linear(global_cond_channels, global_cond_channels),
            nn.ReLU(inplace=True),
            nn.Linear(global_cond_channels, global_cond_channels),
        )
        # self.local_proj_k = nn.Sequential(
        #     nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
        # )
        
        self._initialize_momentum_projs()

    def _initialize_momentum(self):
        for param_q, param_k in zip(self.unet_q.parameters(), self.unet_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def _initialize_momentum_projs(self):
        for param_q, param_k in zip(self.global_proj_q.parameters(), self.global_proj_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # for param_q, param_k in zip(self.local_proj_q.parameters(), self.local_proj_k.parameters()):
        #     param_k.data.copy_(param_q.data)
        #     param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.unet_q.parameters(), self.unet_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data
        
        for param_q, param_k in zip(self.global_proj_q.parameters(), self.global_proj_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data
        
        # for param_q, param_k in zip(self.local_proj_q.parameters(), self.local_proj_k.parameters()):
        #     param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

    @torch.no_grad()
    def reset_unet_k(self):
        self.unet_k.apply(self._init_module)
        
        for param_k in self.unet_k.parameters():
            param_k.requires_grad = False
            
    def _init_module(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, im_q, im_k, im_negs):
        b, n_neg, c, h, w = im_negs.shape

        z_q_0 = self.unet_q(im_q)
        z_q = self.global_proj_q(z_q_0.squeeze(-1).squeeze(-1))
        z_q = F.normalize(z_q, dim=1) # b,c
        # y_q = self.local_proj_q(y_q_0)
        # y_q = F.normalize(y_q, dim=1) # b,c,h,w

        with torch.no_grad():
            self._momentum_update_key_encoder()

            z_k = self.unet_k(im_k)
            z_k = self.global_proj_k(z_k.squeeze(-1).squeeze(-1))
            z_k = F.normalize(z_k, dim=1) # b,c
            # y_k = self.local_proj_k(y_k)
            # y_k = F.normalize(y_k, dim=1) # b,c,h,w

            im_negs = im_negs.view(b * n_neg, c, h, w)
            z_negs = self.unet_k(im_negs)
            z_negs = self.global_proj_k(z_negs.squeeze(-1).squeeze(-1)) 
            z_negs = F.normalize(z_negs, dim=1)
            z_negs = z_negs.view(b, n_neg, -1)
            # y_negs = self.local_proj_k(y_negs)
            # y_negs = F.normalize(y_negs, dim=1)
            # y_negs = y_negs.view(b, n_neg, *y_negs.shape[1:])

        l_pos_global = (z_q * z_k).sum(dim=-1, keepdim=True)
        l_neg_global = torch.einsum('bd,bnd->bn', z_q, z_negs)
        logits_global = torch.cat([l_pos_global, l_neg_global], dim=1) / self.temperature
        labels_global = torch.zeros(logits_global.size(0), dtype=torch.long).cuda()

        # l_pos_local = (y_q * y_k).mean(dim=(1, 2, 3)).unsqueeze(-1)
        # l_neg_local = torch.einsum('bchw,bnchw->bn', y_q, y_negs) / (h * w)
        # logits_local = torch.cat([l_pos_local, l_neg_local], dim=1) / self.temperature
        # labels_local = torch.zeros(logits_local.size(0), dtype=torch.long).cuda()

        return logits_global, labels_global, z_q_0
    
    
# class MoCoWithUNet(nn.Module):
#     def __init__(self, m=0.999, temperature=1.0, in_channels=3, global_cond_channels=64, spatial_cond_channels=16):
#         super(MoCoWithUNet, self).__init__()
        
#         self.unet_q = HybridConditionModule(in_channels, spatial_cond_channels, global_cond_channels)
#         self.unet_k = HybridConditionModule(in_channels, spatial_cond_channels, global_cond_channels)
#         self.m = m
#         self.temperature = temperature

#         for param_q, param_k in zip(self.unet_q.parameters(), self.unet_k.parameters()):
#             param_k.data.copy_(param_q.data)
#             param_k.requires_grad = False

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
#         for param_q, param_k in zip(self.unet_q.parameters(), self.unet_k.parameters()):
#             param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

#     def forward(self, im_q, im_k, im_negs):
#         b, n_neg, c, h, w = im_negs.shape

#         z_q_0, y_q_0 = self.unet_q(im_q)
#         z_q = self.global_proj_q(z_q_0.squeeze(-1).squeeze(-1))
#         z_q = F.normalize(z_q, dim=1) # b,c
#         y_q = self.local_proj_q(y_q_0)
#         y_q = F.normalize(y_q, dim=1) # b,c,h,w

#         with torch.no_grad():
#             self._momentum_update_key_encoder()

#             z_k, y_k = self.unet_k(im_k)
#             z_k = self.global_proj_k(z_k.squeeze(-1).squeeze(-1))
#             z_k = F.normalize(z_k, dim=1) # b,c
#             y_k = self.local_proj_k(y_k)
#             y_k = F.normalize(y_k, dim=1) # b,c,h,w

#             im_negs = im_negs.view(b * n_neg, c, h, w)
#             z_negs, y_negs = self.unet_k(im_negs)
#             z_negs = self.global_proj_k(z_negs.squeeze(-1).squeeze(-1))  # 全局负样本
#             z_negs = F.normalize(z_negs, dim=1)
#             z_negs = z_negs.view(b, n_neg, -1)
#             y_negs = self.local_proj_k(y_negs)
#             y_negs = F.normalize(y_negs, dim=1)
#             y_negs = y_negs.view(b, n_neg, *y_negs.shape[1:])

#         l_pos_global = (z_q * z_k).sum(dim=-1, keepdim=True)
#         l_neg_global = torch.einsum('bd,bnd->bn', z_q, z_negs)
#         logits_global = torch.cat([l_pos_global, l_neg_global], dim=1) / self.temperature
#         labels_global = torch.zeros(logits_global.size(0), dtype=torch.long).cuda()

#         l_pos_local = (y_q * y_k).mean(dim=(1, 2, 3)).unsqueeze(-1)
#         l_neg_local = torch.einsum('bchw,bnchw->bn', y_q, y_negs) / (h * w)
#         logits_local = torch.cat([l_pos_local, l_neg_local], dim=1) / self.temperature
#         labels_local = torch.zeros(logits_local.size(0), dtype=torch.long).cuda()

#         return logits_global, labels_global, logits_local, labels_local, z_q_0, y_q_0






class MoCoWithUNetShared(nn.Module):
    def __init__(self, shared_base, m=0.999, temperature=1.0, global_cond_channels=64, spatial_cond_channels=16):
        super(MoCoWithUNet, self).__init__()
        
        self.shared_base = shared_base  
        
        self.unet_q = HybridConditionModuleHigh(shared_base.out_channels, global_cond_channels, spatial_cond_channels)
        
        self.unet_k = HybridConditionModuleHigh(shared_base.out_channels, global_cond_channels, spatial_cond_channels)
        
        self.m = m
        self.temperature = temperature

        for param_q, param_k in zip(self.unet_q.parameters(), self.unet_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

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

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.unet_q.parameters(), self.unet_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

    def forward(self, im_q, im_k, im_negs):
        b, n_neg, c, h, w = im_negs.shape

        z_q_0, y_q_0 = self.unet_q(im_q)
        z_q = self.global_proj_q(z_q_0.squeeze(-1).squeeze(-1))
        z_q = F.normalize(z_q, dim=1) # b,c
        y_q = self.local_proj_q(y_q_0)
        y_q = F.normalize(y_q, dim=1) # b,c,h,w

        with torch.no_grad():
            self._momentum_update_key_encoder()

            z_k, y_k = self.unet_k(im_k)
            z_k = self.global_proj_k(z_k.squeeze(-1).squeeze(-1))
            z_k = F.normalize(z_k, dim=1) # b,c
            y_k = self.local_proj_k(y_k)
            y_k = F.normalize(y_k, dim=1) # b,c,h,w

            im_negs = im_negs.view(b * n_neg, c, h, w)
            z_negs, y_negs = self.unet_k(im_negs)
            z_negs = self.global_proj_k(z_negs.squeeze(-1).squeeze(-1))  # 全局负样本
            z_negs = F.normalize(z_negs, dim=1)
            z_negs = z_negs.view(b, n_neg, -1)
            y_negs = self.local_proj_k(y_negs)
            y_negs = F.normalize(y_negs, dim=1)
            y_negs = y_negs.view(b, n_neg, *y_negs.shape[1:])

        l_pos_global = (z_q * z_k).sum(dim=-1, keepdim=True)
        l_neg_global = torch.einsum('bd,bnd->bn', z_q, z_negs)
        logits_global = torch.cat([l_pos_global, l_neg_global], dim=1) / self.temperature
        labels_global = torch.zeros(logits_global.size(0), dtype=torch.long).cuda()

        l_pos_local = (y_q * y_k).mean(dim=(1, 2, 3)).unsqueeze(-1)
        l_neg_local = torch.einsum('bchw,bnchw->bn', y_q, y_negs) / (h * w)
        logits_local = torch.cat([l_pos_local, l_neg_local], dim=1) / self.temperature
        labels_local = torch.zeros(logits_local.size(0), dtype=torch.long).cuda()

        return logits_global, labels_global, logits_local, labels_local, z_q_0, y_q_0
