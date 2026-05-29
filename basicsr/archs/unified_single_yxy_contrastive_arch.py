import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .RealRep_fuse_zero_norm_arch import (
    HybridConditionEncoder,
    HybridConditionGlobalHead,
    HybridConditionLocalDecoder,
)


class UnifiedBranchConditionModule(nn.Module):
    def __init__(self, bottleneck_channels, global_cond_channels, spatial_cond_channels, init_mid_channels=16, up_method='bilinear'):
        super().__init__()
        self.global_head = HybridConditionGlobalHead(bottleneck_channels, global_cond_channels)
        self.local_decoder = HybridConditionLocalDecoder(
            spatial_cond_channels,
            init_mid_channels=init_mid_channels,
            up_method=up_method,
        )

    def forward(self, x_1, x_2, x_3, x_4):
        z = self.global_head(x_4)
        y = self.local_decoder(x_1, x_2, x_3, x_4)
        return z, y


@ARCH_REGISTRY.register()
class UnifiedSingleYXYMoCo(nn.Module):
    def __init__(
        self,
        m=0.999,
        temperature=1.0,
        in_channels=3,
        global_cond_channels=64,
        spatial_cond_channels=16,
        init_mid_channels=16,
        alpha_min=0.2,
        alpha_max=0.8,
        alpha_mode='beta',
        alpha_beta=0.5,
        alpha_dominance='rank',
        alpha_mix_policy='replace',
        use_alpha_mix=True,
        down_method='stride',
        up_method='bilinear',
    ):
        super().__init__()
        self.m = m
        self.temperature = temperature
        self.use_alpha_mix = use_alpha_mix
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_mode = alpha_mode
        self.alpha_beta = alpha_beta
        self.alpha_dominance = alpha_dominance
        self.alpha_mix_policy = alpha_mix_policy
        self.init_mid_channels = init_mid_channels

        bottleneck_channels = init_mid_channels * 8

        self.encoder_q = HybridConditionEncoder(in_channels=in_channels, init_mid_channels=init_mid_channels, down_method=down_method)
        self.encoder_k = HybridConditionEncoder(in_channels=in_channels, init_mid_channels=init_mid_channels, down_method=down_method)

        self.branch_q_I = UnifiedBranchConditionModule(bottleneck_channels, global_cond_channels, spatial_cond_channels, init_mid_channels=init_mid_channels, up_method=up_method)
        self.branch_q_TP = UnifiedBranchConditionModule(bottleneck_channels, global_cond_channels, spatial_cond_channels, init_mid_channels=init_mid_channels, up_method=up_method)
        self.branch_k_I = UnifiedBranchConditionModule(bottleneck_channels, global_cond_channels, spatial_cond_channels, init_mid_channels=init_mid_channels, up_method=up_method)
        self.branch_k_TP = UnifiedBranchConditionModule(bottleneck_channels, global_cond_channels, spatial_cond_channels, init_mid_channels=init_mid_channels, up_method=up_method)

        self.global_proj_q_I = nn.Sequential(
            nn.Linear(global_cond_channels, global_cond_channels),
            nn.ReLU(inplace=True),
            nn.Linear(global_cond_channels, global_cond_channels),
        )
        self.global_proj_q_TP = nn.Sequential(
            nn.Linear(global_cond_channels, global_cond_channels),
            nn.ReLU(inplace=True),
            nn.Linear(global_cond_channels, global_cond_channels),
        )
        self.global_proj_k_I = nn.Sequential(
            nn.Linear(global_cond_channels, global_cond_channels),
            nn.ReLU(inplace=True),
            nn.Linear(global_cond_channels, global_cond_channels),
        )
        self.global_proj_k_TP = nn.Sequential(
            nn.Linear(global_cond_channels, global_cond_channels),
            nn.ReLU(inplace=True),
            nn.Linear(global_cond_channels, global_cond_channels),
        )

        self.local_proj_q_I = nn.Sequential(
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
        )
        self.local_proj_q_TP = nn.Sequential(
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
        )
        self.local_proj_k_I = nn.Sequential(
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
        )
        self.local_proj_k_TP = nn.Sequential(
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_cond_channels, spatial_cond_channels, 1),
        )

        self._initialize_momentum()

    def _iter_qk_pairs(self):
        yield self.encoder_q, self.encoder_k
        yield self.branch_q_I, self.branch_k_I
        yield self.branch_q_TP, self.branch_k_TP
        yield self.global_proj_q_I, self.global_proj_k_I
        yield self.global_proj_q_TP, self.global_proj_k_TP
        yield self.local_proj_q_I, self.local_proj_k_I
        yield self.local_proj_q_TP, self.local_proj_k_TP

    def _initialize_momentum(self):
        for module_q, module_k in self._iter_qk_pairs():
            for param_q, param_k in zip(module_q.parameters(), module_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for module_q, module_k in self._iter_qk_pairs():
            for param_q, param_k in zip(module_q.parameters(), module_k.parameters()):
                param_k.data = self.m * param_k.data + (1.0 - self.m) * param_q.data

    def _encode_query_feat(self, x):
        return self.encoder_q(x)

    def _encode_key_feat(self, x):
        return self.encoder_k(x)

    def _branch_heads(self, q_or_k='q', branch='I'):
        if q_or_k == 'q' and branch == 'I':
            return self.branch_q_I, self.global_proj_q_I, self.local_proj_q_I
        if q_or_k == 'q' and branch == 'TP':
            return self.branch_q_TP, self.global_proj_q_TP, self.local_proj_q_TP
        if q_or_k == 'k' and branch == 'I':
            return self.branch_k_I, self.global_proj_k_I, self.local_proj_k_I
        if q_or_k == 'k' and branch == 'TP':
            return self.branch_k_TP, self.global_proj_k_TP, self.local_proj_k_TP
        raise ValueError(f'Unknown pair ({q_or_k}, {branch})')

    def _build_conditions(self, feat_pyramid, q_or_k='q'):
        x_1, x_2, x_3, x_4 = feat_pyramid
        head_I, proj_I, local_I = self._branch_heads(q_or_k=q_or_k, branch='I')
        head_TP, proj_TP, local_TP = self._branch_heads(q_or_k=q_or_k, branch='TP')

        z_I_0, y_I_0 = head_I(x_1, x_2, x_3, x_4)
        z_TP_0, y_TP_0 = head_TP(x_1, x_2, x_3, x_4)

        z_I = proj_I(z_I_0.squeeze(-1).squeeze(-1))
        z_TP = proj_TP(z_TP_0.squeeze(-1).squeeze(-1))
        z_I = F.normalize(z_I, dim=1)
        z_TP = F.normalize(z_TP, dim=1)

        y_I = F.normalize(local_I(y_I_0), dim=1)
        y_TP = F.normalize(local_TP(y_TP_0), dim=1)
        return z_I, z_TP, y_I, y_TP

    @torch.no_grad()
    def copy_q_to_k(self):
        self._initialize_momentum()

    def _init_module(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    @torch.no_grad()
    def reinit_key_encoder_all(self):
        for module in [
            self.encoder_k,
            self.branch_k_I,
            self.branch_k_TP,
            self.global_proj_k_I,
            self.global_proj_k_TP,
            self.local_proj_k_I,
            self.local_proj_k_TP,
        ]:
            module.apply(self._init_module)
            for param in module.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def stage2_init_key_encoder(self, mode: str = 'keep'):
        mode = (mode or 'keep').lower()
        if mode == 'keep':
            return
        if mode == 'copy_q':
            self.copy_q_to_k()
            return
        if mode in ['reinit_all', 'reinit_full', 'reinit_unet', 'reinit_backbone', 'reinit_unet_only']:
            self.reinit_key_encoder_all()
            return
        raise ValueError(f'Unknown stage2_init_k mode: {mode}')

    def encode_q(self, im_q):
        feat_q = self._encode_query_feat(im_q)
        return self._build_conditions(feat_q, q_or_k='q')

    def encode_versions(self, views):
        b, n, c, h, w = views.shape
        flat = views.view(b * n, c, h, w)
        feat = self._encode_query_feat(flat)
        z_I, z_TP, _, _ = self._build_conditions(feat, q_or_k='q')
        return z_I.view(b, n, -1), z_TP.view(b, n, -1)

    def _mix_negs(self, z_k, y_k, z_negs, y_negs, neg_ranks=None):
        b, n_neg, _ = z_negs.shape
        device = z_negs.device
        if not self.use_alpha_mix or n_neg == 0:
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
            alpha = torch.full((b, n_neg, 1), (self.alpha_min + self.alpha_max) * 0.5, device=device, dtype=torch.float32)

        if self.alpha_dominance == 'rank' and neg_ranks is not None:
            if neg_ranks.dim() == 1:
                neg_ranks = neg_ranks.view(b, n_neg)
            neg_ranks_f = neg_ranks.to(device=device, dtype=torch.float32).clamp_min(0.0)
            rmax = neg_ranks_f.max(dim=1, keepdim=True).values + 1e-6
            r_norm = (neg_ranks_f / rmax).unsqueeze(-1)
            w_hard = 1.0 - r_norm
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * w_hard

        z_mix = alpha * z_k.unsqueeze(1) + (1.0 - alpha) * z_negs
        z_mix = F.normalize(z_mix, dim=-1)

        alpha_y = alpha.view(b, n_neg, 1, 1, 1)
        y_mix = alpha_y * y_k.unsqueeze(1) + (1.0 - alpha_y) * y_negs
        y_mix = F.normalize(y_mix, dim=2)

        if self.alpha_mix_policy == 'replace':
            return z_mix, y_mix
        return torch.cat([z_negs, z_mix], dim=1), torch.cat([y_negs, y_mix], dim=1)

    def _compute_logits(self, z_q, y_q, z_k, y_k, z_negs, y_negs, temperature):
        l_pos_global = (z_q * z_k).sum(dim=-1, keepdim=True)
        l_neg_global = torch.einsum('bd,bnd->bn', z_q, z_negs)
        logits_global = torch.cat([l_pos_global, l_neg_global], dim=1) / temperature
        labels_global = torch.zeros(logits_global.size(0), dtype=torch.long, device=z_q.device)

        h, w = y_q.shape[-2:]
        l_pos_local = (y_q * y_k).mean(dim=(1, 2, 3)).unsqueeze(-1)
        l_neg_local = torch.einsum('bchw,bnchw->bn', y_q, y_negs) / (h * w)
        logits_local = torch.cat([l_pos_local, l_neg_local], dim=1) / temperature
        labels_local = torch.zeros(logits_local.size(0), dtype=torch.long, device=z_q.device)
        return logits_global, labels_global, logits_local, labels_local

    def forward(self, im_q, im_k, im_negs=None, neg_ranks=None):
        if im_negs is None:
            im_negs = im_k.unsqueeze(1)
        if im_negs.dim() == 4:
            im_negs = im_negs.unsqueeze(1)

        assert im_negs.dim() == 5, f'im_negs must be 5D [B,N,C,H,W], got {im_negs.shape}'
        b, n_neg, c, h, w = im_negs.shape

        feat_q = self._encode_query_feat(im_q)
        z_q_I, z_q_TP, y_q_I, y_q_TP = self._build_conditions(feat_q, q_or_k='q')

        with torch.no_grad():
            self._momentum_update_key_encoder()
            feat_k = self._encode_key_feat(im_k)
            z_k_I, z_k_TP, y_k_I, y_k_TP = self._build_conditions(feat_k, q_or_k='k')

            feat_negs = self._encode_key_feat(im_negs.view(b * n_neg, c, h, w))
            z_n_I, z_n_TP, y_n_I, y_n_TP = self._build_conditions(feat_negs, q_or_k='k')
            z_n_I = z_n_I.view(b, n_neg, -1)
            z_n_TP = z_n_TP.view(b, n_neg, -1)
            y_n_I = y_n_I.view(b, n_neg, *y_n_I.shape[1:])
            y_n_TP = y_n_TP.view(b, n_neg, *y_n_TP.shape[1:])

            z_n_I, y_n_I = self._mix_negs(z_k_I, y_k_I, z_n_I, y_n_I, neg_ranks=neg_ranks)
            z_n_TP, y_n_TP = self._mix_negs(z_k_TP, y_k_TP, z_n_TP, y_n_TP, neg_ranks=neg_ranks)

        logits_global_I, labels_global_I, logits_local_I, labels_local_I = self._compute_logits(
            z_q_I, y_q_I, z_k_I, y_k_I, z_n_I, y_n_I, self.temperature
        )
        logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP = self._compute_logits(
            z_q_TP, y_q_TP, z_k_TP, y_k_TP, z_n_TP, y_n_TP, self.temperature
        )

        return (
            logits_global_I, labels_global_I,
            logits_local_I, labels_local_I,
            logits_global_TP, labels_global_TP,
            logits_local_TP, labels_local_TP,
            z_q_I.unsqueeze(-1).unsqueeze(-1),
            z_q_TP.unsqueeze(-1).unsqueeze(-1),
            y_q_I,
            y_q_TP,
        )
