import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from os import path as osp

import numpy as np
from torch.cuda.amp import autocast
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.models.itm_model import ITMModel, ContrastYXYMidSupITMModel, apply_random_flip
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.ICTCP_convert import SDR_to_YCbCr, YCbCr_to_SDR
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ContrastSingleMoCoBaseITMModel(ITMModel):
    @staticmethod
    def _srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
        a = 0.055
        return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1 + a)) ** 2.4)

    @staticmethod
    def _linear_rgb_to_xyz(lin_rgb: torch.Tensor) -> torch.Tensor:
        M = lin_rgb.new_tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ])
        B, C, H, W = lin_rgb.shape
        rgb_flat = lin_rgb.permute(0, 2, 3, 1).reshape(-1, 3)
        xyz_flat = rgb_flat @ M.T
        xyz = xyz_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)
        return xyz

    @classmethod
    def _rgb_to_yxy(cls, rgb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        lin = cls._srgb_to_linear(torch.clamp(rgb, 0.0, 1.0))
        xyz = cls._linear_rgb_to_xyz(lin)
        X = xyz[:, 0:1, :, :]
        Y = xyz[:, 1:2, :, :]
        Z = xyz[:, 2:3, :, :]
        denom = X + Y + Z + eps
        x = X / denom
        y = Y / denom
        return torch.cat([Y, x, y], dim=1)


    def _check_if_zero_initialized(self, model):
        for name, module in model.named_modules():
            if module.__class__.__name__ in ['ZeroConv2d', 'ZeroLinear']:
                print(f"[DEBUG] Found zero layer: {name} ({module.__class__.__name__})")
                return True
        return False

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.use_bf16 = train_opt.get('use_bf16', False)
        if self.use_bf16:
            logger = get_root_logger()
            logger.info('Using bfloat16 mixed precision training ...')

        self.scaler = torch.cuda.amp.GradScaler() if self.use_bf16 else None

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)

            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('mid_sup_opt'):
            mid_sup_opt = train_opt['mid_sup_opt']
            if mid_sup_opt.get('pixel_opt'):
                self.cri_pix_mid = build_loss(mid_sup_opt['pixel_opt']).to(self.device)
            else:
                self.cri_pix_mid = None
        else:
            self.cri_pix_mid = None

        if train_opt.get('contra_opt'):
            self.criterion_contra = torch.nn.CrossEntropyLoss().to(self.device)
            self.cri_contra_weight = train_opt['contra_opt']['loss_weight']
            self.cri_contra_weight_global = train_opt['contra_opt']['loss_weight_global']
            self.cri_contra_weight_local = train_opt['contra_opt']['loss_weight_local']
        else:
            self.criterion_contra = None

        self.setup_optimizers()
        self.setup_schedulers()

    def __init__(self, opt):
        optim_cfg_backup = None
        scheduler_cfg_backup = None
        if opt.get('is_train', False):
            train_opt = opt['train']
            optim_cfg_backup = dict(train_opt['optim_g'])
            scheduler_cfg_backup = dict(train_opt['scheduler'])

        super(SRModel, self).__init__(opt)

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        self.average_metrics = {}

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        self.use_bf16 = self.opt.get('use_bf16', False)
        self.stage2_key_inited = False
        self._set_stage2_resume_flags()

        self.feat_extractor = build_network(opt['network_feat'])
        self.feat_extractor = self.model_to_device(self.feat_extractor)
        self.print_network(self.feat_extractor)

        if hasattr(self, 'feat_extractor_I'):
            del self.feat_extractor_I
        if hasattr(self, 'feat_extractor_TP'):
            del self.feat_extractor_TP

        load_path_feat = self.opt['path'].get('pretrain_network_g', None)
        if load_path_feat is not None:
            param_key = self.opt['path'].get('param_key_feat_single_moco_base', 'feat_params_single_moco_base')
            self.load_network(self.feat_extractor, load_path_feat, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.opt['train']['optim_g'] = optim_cfg_backup
            self.opt['train']['scheduler'] = scheduler_cfg_backup
            self.optimizers = []
            self.schedulers = []
            self.init_training_settings()

        if self._check_if_zero_initialized(self.net_g):
            self.zero_switched = True
        else:
            self.zero_switched = False

    def _infer_resume_iter_for_stage2(self):
        path_opt = self.opt.get('path', {})
        candidates = [
            path_opt.get('resume_state', None),
            path_opt.get('pretrain_network_g', None),
        ]
        for path in candidates:
            if not path:
                continue
            stem = osp.splitext(osp.basename(path))[0]
            digits = ''.join(ch for ch in stem if ch.isdigit())
            if digits:
                try:
                    return int(digits)
                except ValueError:
                    continue
        return None

    def _set_stage2_resume_flags(self):
        if not self.opt.get('is_train', False):
            return
        train_opt = self.opt.get('train', {})
        stage1_iters = train_opt.get('stage1_iters', None)
        if stage1_iters is None:
            return
        resume_iter = self._infer_resume_iter_for_stage2()
        if resume_iter is not None and resume_iter > stage1_iters:
            self.stage2_key_inited = True

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        self.im_q = data['im_q'].to(self.device)
        self.im_k = data['im_k'].to(self.device)

        self.im_negs_I = None
        if data.get('im_negs_I_raw', None) is not None and data['im_negs_I_raw'] is not None:
            view_list = []
            for neg_cpu in data['im_negs_I_raw']:
                view_list.append(neg_cpu.to(self.device))
            if len(view_list) > 0:
                self.im_negs_I = torch.stack(view_list, dim=1)
                self.im_negs_I = apply_random_flip(self.im_negs_I)

        self.im_negs_TP = None
        if data.get('im_negs_TP_raw', None) is not None and data['im_negs_TP_raw'] is not None:
            view_list = []
            for neg_cpu in data['im_negs_TP_raw']:
                view_list.append(neg_cpu.to(self.device))
            if len(view_list) > 0:
                self.im_negs_TP = torch.stack(view_list, dim=1)
                self.im_negs_TP = apply_random_flip(self.im_negs_TP)


    def feed_data_test(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        self.im_q = data['im_q'].to(self.device)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        logger = get_root_logger()

        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger.warning(f'Params {k} in net_g will not be optimized.')

        optim_params_feat = []
        for k, v in self.feat_extractor.named_parameters():
            if v.requires_grad:
                optim_params_feat.append(v)
            else:
                logger.warning(f'Params {k} in feat_extractor will not be optimized.')

        optim_cfg = dict(train_opt['optim_g'])
        optim_type = optim_cfg.pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **optim_cfg)
        self.optimizer_feat = self.get_optimizer(optim_type, optim_params_feat, **optim_cfg)
        self.optimizers.append(self.optimizer_g)
        self.optimizers.append(self.optimizer_feat)

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']
        self.optimizer_g.zero_grad()
        self.optimizer_feat.zero_grad()

        l_total = 0
        loss_dict = OrderedDict()

        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            if current_iter <= train_opt['stage1_iters']:
                z_q, y_q = self.feat_extractor.encode_q(self.im_q)
                self.output = self.net_g(self.lq, z_q, y_q)
                self.output_mid = self.net_g.mid_result

                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                if self.cri_pix_mid:
                    l_pix_mid = self.cri_pix_mid(self.output_mid, self.gt)
                    l_total += l_pix_mid
                    loss_dict['l_pix_mid'] = l_pix_mid
            else:
                if not self.stage2_key_inited:
                    mode = train_opt.get('stage2_init_k', 'keep')
                    self.feat_extractor.stage2_init_key_encoder(mode)
                    logger = get_root_logger()
                    logger.info(f'[INFO] Stage2 key init(single-moco-base): mode={mode}, iter={current_iter}')
                    self.stage2_key_inited = True

                if train_opt.get('enable_zero_switch', True) and (not self.zero_switched):
                    if hasattr(self.net_g, 'switch_to_zero'):
                        self.net_g.switch_to_zero(use_bf16=self.use_bf16, device=self.device)
                    self.zero_switched = True

                (
                    logits_global_I, labels_global_I,
                    logits_local_I, labels_local_I,
                    logits_global_TP, labels_global_TP,
                    logits_local_TP, labels_local_TP,
                    z_q, y_q,
                ) = self.feat_extractor(
                    self.im_q,
                    self.im_k,
                    self.im_negs_I,
                    self.im_negs_TP,
                )

                global_contra_loss_I = self.criterion_contra(logits_global_I, labels_global_I)
                loss_dict['l_global_contra_I'] = global_contra_loss_I
                l_total += global_contra_loss_I * self.cri_contra_weight_global * self.cri_contra_weight

                global_contra_loss_TP = self.criterion_contra(logits_global_TP, labels_global_TP)
                loss_dict['l_global_contra_TP'] = global_contra_loss_TP
                l_total += global_contra_loss_TP * self.cri_contra_weight_global * self.cri_contra_weight

                local_contra_loss_I = self.criterion_contra(logits_local_I, labels_local_I)
                loss_dict['l_local_contra_I'] = local_contra_loss_I
                l_total += local_contra_loss_I * self.cri_contra_weight_local * self.cri_contra_weight

                local_contra_loss_TP = self.criterion_contra(logits_local_TP, labels_local_TP)
                loss_dict['l_local_contra_TP'] = local_contra_loss_TP
                l_total += local_contra_loss_TP * self.cri_contra_weight_local * self.cri_contra_weight

                self.output = self.net_g(self.lq, z_q, y_q)
                self.output_mid = self.net_g.mid_result

                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                if self.cri_pix_mid:
                    l_pix_mid = self.cri_pix_mid(self.output_mid, self.gt)
                    l_total += l_pix_mid
                    loss_dict['l_pix_mid'] = l_pix_mid

        if self.scaler is not None:
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.step(self.optimizer_feat)
            self.scaler.update()
        else:
            l_total.backward()
            self.optimizer_g.step()
            self.optimizer_feat.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.feat_extractor.eval()
                with torch.no_grad():
                    z_q, y_q = self.feat_extractor.encode_k(self.im_q)
                    self.output = self.net_g_ema(self.lq, z_q, y_q)
                    self.output_mid = self.net_g_ema.mid_result
                self.feat_extractor.train()
            else:
                self.net_g.eval()
                self.feat_extractor.eval()
                with torch.no_grad():
                    z_q, y_q = self.feat_extractor.encode_k(self.im_q)
                    self.output = self.net_g(self.lq, z_q, y_q)
                    self.output_mid = self.net_g.mid_result
                self.net_g.train()
                self.feat_extractor.train()
    def test_stage1(self):
        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.feat_extractor.eval()
                with torch.no_grad():
                    z_q, y_q = self.feat_extractor.encode_q(self.im_q)
                    self.output = self.net_g_ema(self.lq, z_q, y_q)
                    self.output_mid = self.net_g_ema.mid_result
                self.feat_extractor.train()
            else:
                self.net_g.eval()
                self.feat_extractor.eval()
                with torch.no_grad():
                    z_q, y_q = self.feat_extractor.encode_q(self.im_q)
                    self.output = self.net_g(self.lq, z_q, y_q)
                    self.output_mid = self.net_g.mid_result
                self.net_g.train()
                self.feat_extractor.train()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data_test(val_data)
            # self.test()
            self.test_stage1()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], out_type=np.uint16)
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], out_type=np.uint16)
                metric_data['img2'] = gt_img
                del self.gt

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def save(self, epoch, current_iter):
        nets_to_save = [self.net_g]
        param_keys = ['params']
        if hasattr(self, 'net_g_ema'):
            nets_to_save.append(self.net_g_ema)
            param_keys.append('params_ema')
        nets_to_save.append(self.feat_extractor)
        param_keys.append('feat_params_single_moco_base')
        self.save_network(net=nets_to_save, net_label='net_g', current_iter=current_iter, param_key=param_keys)
        self.save_training_state(epoch, current_iter)


@MODEL_REGISTRY.register()
class ContrastSingleMoCoSoftMiningITMModel(ContrastSingleMoCoBaseITMModel):
    pass


@MODEL_REGISTRY.register()
class ContrastSingleMoCoRankITMModel(ContrastSingleMoCoSoftMiningITMModel):
    def init_training_settings(self):
        super().init_training_settings()
        train_opt = self.opt['train']
        self._warned_missing_negs = False
        self._warned_missing_ranks = False
        if train_opt.get('contra_opt'):
            self.use_invariance = train_opt['contra_opt'].get('use_invariance', False)
            self.inv_weight = train_opt['contra_opt'].get('inv_weight', 0.5)
            self.dist_pool = train_opt['contra_opt'].get('dist_pool', 32)
            self.inv_every = train_opt['contra_opt'].get('inv_every', 2)
            self.inv_max_versions = train_opt['contra_opt'].get('inv_max_versions', 4)
        else:
            self.use_invariance = False
            self.inv_weight = 0.5
            self.dist_pool = 32
            self.inv_every = 2
            self.inv_max_versions = 4

    def feed_data(self, data):
        super().feed_data(data)
        self.im_negs_I_rank = None
        if self.im_negs_I is not None and data.get('im_negs_I_rank', None) is not None:
            self.im_negs_I_rank = data['im_negs_I_rank'].to(self.device)

        self.im_negs_TP_rank = None
        if self.im_negs_TP is not None and data.get('im_negs_TP_rank', None) is not None:
            self.im_negs_TP_rank = data['im_negs_TP_rank'].to(self.device)

        rank_chunks = []
        if self.im_negs_I_rank is not None:
            rank_chunks.append(self.im_negs_I_rank)
        if self.im_negs_TP_rank is not None:
            rank_chunks.append(self.im_negs_TP_rank)
        self.im_negs_rank = torch.cat(rank_chunks, dim=1) if len(rank_chunks) > 0 else None

        self.negs_rgb_cpu = []
        k = max(self.inv_max_versions - 2, 0)
        if k > 0:
            if data.get('im_negs_I_raw', None) is not None:
                self.negs_rgb_cpu.extend(data['im_negs_I_raw'][:k])
            remaining = max(k - len(self.negs_rgb_cpu), 0)
            if remaining > 0 and data.get('im_negs_TP_raw', None) is not None:
                self.negs_rgb_cpu.extend(data['im_negs_TP_raw'][:remaining])
            self.negs_rgb_cpu = self.negs_rgb_cpu[:k]
        if len(self.negs_rgb_cpu) == 0:
            self.negs_rgb_cpu = None

    def _warn_if_neg_fallback(self, current_iter):
        logger = get_root_logger()
        has_I = self.im_negs_I is not None
        has_TP = self.im_negs_TP is not None
        has_rank_I = self.im_negs_I_rank is not None
        has_rank_TP = self.im_negs_TP_rank is not None

        if (not has_I) and (not has_TP) and (not self._warned_missing_negs):
            logger.warning(
                f'No negative samples were provided to rank/latentmix/inv path at iter={current_iter}; '
                'falling back to key-only negatives (k.unsqueeze(1)). Please check dataset/model negative field compatibility.'
            )
            self._warned_missing_negs = True

        if (has_I or has_TP) and (not has_rank_I) and (not has_rank_TP) and (not self._warned_missing_ranks):
            logger.warning(
                f'Negative samples are present but rank tensors are missing at iter={current_iter}; '
                'rank-aware and latentmix rank-dominance logic may be inactive.'
            )
            self._warned_missing_ranks = True

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']
        self.optimizer_g.zero_grad()
        self.optimizer_feat.zero_grad()

        l_total = 0
        loss_dict = OrderedDict()

        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            if current_iter <= train_opt['stage1_iters']:
                z_q, y_q = self.feat_extractor.encode_q(self.im_q)
                self.output = self.net_g(self.lq, z_q, y_q)
                self.output_mid = self.net_g.mid_result

                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                if self.cri_pix_mid:
                    l_pix_mid = self.cri_pix_mid(self.output_mid, self.gt)
                    l_total += l_pix_mid
                    loss_dict['l_pix_mid'] = l_pix_mid
            else:
                if not self.stage2_key_inited:
                    mode = train_opt.get('stage2_init_k', 'keep')
                    self.feat_extractor.stage2_init_key_encoder(mode)
                    logger = get_root_logger()
                    logger.info(f'[INFO] Stage2 key init(single-moco-rank): mode={mode}, iter={current_iter}')
                    self.stage2_key_inited = True

                if train_opt.get('enable_zero_switch', True) and (not self.zero_switched):
                    if hasattr(self.net_g, 'switch_to_zero'):
                        self.net_g.switch_to_zero(use_bf16=self.use_bf16, device=self.device)
                    self.zero_switched = True

                self._warn_if_neg_fallback(current_iter)
                (
                    logits_global_I, labels_global_I,
                    logits_local_I, labels_local_I,
                    logits_global_TP, labels_global_TP,
                    logits_local_TP, labels_local_TP,
                    z_q, y_q,
                ) = self.feat_extractor(
                    self.im_q,
                    self.im_k,
                    self.im_negs_I,
                    self.im_negs_TP,
                    neg_ranks_I=self.im_negs_I_rank,
                    neg_ranks_TP=self.im_negs_TP_rank,
                )

                global_contra_loss_I = self.criterion_contra(logits_global_I, labels_global_I)
                loss_dict['l_global_contra_I'] = global_contra_loss_I
                l_total += global_contra_loss_I * self.cri_contra_weight_global * self.cri_contra_weight

                global_contra_loss_TP = self.criterion_contra(logits_global_TP, labels_global_TP)
                loss_dict['l_global_contra_TP'] = global_contra_loss_TP
                l_total += global_contra_loss_TP * self.cri_contra_weight_global * self.cri_contra_weight

                local_contra_loss_I = self.criterion_contra(logits_local_I, labels_local_I)
                loss_dict['l_local_contra_I'] = local_contra_loss_I
                l_total += local_contra_loss_I * self.cri_contra_weight_local * self.cri_contra_weight

                local_contra_loss_TP = self.criterion_contra(logits_local_TP, labels_local_TP)
                loss_dict['l_local_contra_TP'] = local_contra_loss_TP
                l_total += local_contra_loss_TP * self.cri_contra_weight_local * self.cri_contra_weight

                self.output = self.net_g(self.lq, z_q, y_q)
                self.output_mid = self.net_g.mid_result

                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                if self.cri_pix_mid:
                    l_pix_mid = self.cri_pix_mid(self.output_mid, self.gt)
                    l_total += l_pix_mid
                    loss_dict['l_pix_mid'] = l_pix_mid

        if self.scaler is not None:
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.step(self.optimizer_feat)
            self.scaler.update()
        else:
            l_total.backward()
            self.optimizer_g.step()
            self.optimizer_feat.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


@MODEL_REGISTRY.register()
class ContrastSingleMoCoLatentMixITMModel(ContrastSingleMoCoRankITMModel):
    def init_training_settings(self):
        super().init_training_settings()
        train_opt = self.opt['train']
        self.latentmix_alpha_schedule_iter = train_opt.get('latentmix_alpha_schedule_iter', 120000)
        self.latentmix_alpha_min_early = train_opt.get('latentmix_alpha_min_early', 0.0)
        self.latentmix_alpha_max_early = train_opt.get('latentmix_alpha_max_early', 0.2)
        self.latentmix_alpha_min_late = train_opt.get('latentmix_alpha_min_late', 0.2)
        self.latentmix_alpha_max_late = train_opt.get('latentmix_alpha_max_late', 0.4)
        self._logged_latentmix_alpha_early = False
        self._logged_latentmix_alpha_late = False

    def _apply_latentmix_alpha_schedule(self, current_iter):
        logger = get_root_logger()
        if current_iter < self.latentmix_alpha_schedule_iter:
            self.feat_extractor.alpha_min = self.latentmix_alpha_min_early
            self.feat_extractor.alpha_max = self.latentmix_alpha_max_early
            if not self._logged_latentmix_alpha_early:
                logger.info(
                    f'[INFO] LatentMix alpha schedule(single-moco-latentmix): early phase, '
                    f'iter<{self.latentmix_alpha_schedule_iter}, '
                    f'alpha_min={self.latentmix_alpha_min_early}, alpha_max={self.latentmix_alpha_max_early}'
                )
                self._logged_latentmix_alpha_early = True
        else:
            self.feat_extractor.alpha_min = self.latentmix_alpha_min_late
            self.feat_extractor.alpha_max = self.latentmix_alpha_max_late
            if not self._logged_latentmix_alpha_late:
                logger.info(
                    f'[INFO] LatentMix alpha schedule(single-moco-latentmix): late phase, '
                    f'iter>={self.latentmix_alpha_schedule_iter}, '
                    f'alpha_min={self.latentmix_alpha_min_late}, alpha_max={self.latentmix_alpha_max_late}'
                )
                self._logged_latentmix_alpha_late = True

    def optimize_parameters(self, current_iter):
        self._apply_latentmix_alpha_schedule(current_iter)
        super().optimize_parameters(current_iter)


@MODEL_REGISTRY.register()
class ContrastSingleMoCoOldNegITMModel(ContrastSingleMoCoBaseITMModel):
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        self.im_q = data['im_q'].to(self.device)
        self.im_k = data['im_k'].to(self.device)

        dim = 1
        sdr_ictcp_base = SDR_to_YCbCr(self.im_q, dim=dim)

        self.im_negs_I = None
        if data.get('im_negs_I_raw', None) is not None and data['im_negs_I_raw'] is not None:
            im_negs_I_list = []
            for neg_i_cpu in data['im_negs_I_raw']:
                neg_i_gpu = neg_i_cpu.to(self.device)
                neg_ictcp = SDR_to_YCbCr(neg_i_gpu, dim=dim)
                sdr_ictcp_i = sdr_ictcp_base.clone()
                sdr_ictcp_i[:, 0, :, :] = neg_ictcp[:, 0, :, :]
                replaced_sdr_i = YCbCr_to_SDR(sdr_ictcp_i, dim=dim)
                im_negs_I_list.append(replaced_sdr_i)

            if len(im_negs_I_list) > 0:
                self.im_negs_I = torch.stack(im_negs_I_list, dim=1)
                self.im_negs_I = apply_random_flip(self.im_negs_I)

        self.im_negs_TP = None
        if data.get('im_negs_TP_raw', None) is not None and data['im_negs_TP_raw'] is not None:
            im_negs_TP_list = []
            for neg_tp_cpu in data['im_negs_TP_raw']:
                neg_tp_gpu = neg_tp_cpu.to(self.device)
                neg_ictcp = SDR_to_YCbCr(neg_tp_gpu, dim=dim)
                sdr_ictcp_tp = sdr_ictcp_base.clone()
                sdr_ictcp_tp[:, 1:3, :, :] = neg_ictcp[:, 1:3, :, :]
                replaced_sdr_tp = YCbCr_to_SDR(sdr_ictcp_tp, dim=dim)
                im_negs_TP_list.append(replaced_sdr_tp)

            if len(im_negs_TP_list) > 0:
                self.im_negs_TP = torch.stack(im_negs_TP_list, dim=1)
                self.im_negs_TP = apply_random_flip(self.im_negs_TP)


@MODEL_REGISTRY.register()
class ContrastSingleMoCoLatentMixOldNegITMModel(ContrastSingleMoCoLatentMixITMModel):
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        self.im_q = data['im_q'].to(self.device)
        self.im_k = data['im_k'].to(self.device)

        dim = 1
        sdr_ictcp_base = SDR_to_YCbCr(self.im_q, dim=dim)

        self.im_negs_I = None
        if data.get('im_negs_I_raw', None) is not None and data['im_negs_I_raw'] is not None:
            im_negs_I_list = []
            for neg_i_cpu in data['im_negs_I_raw']:
                neg_i_gpu = neg_i_cpu.to(self.device)
                neg_ictcp = SDR_to_YCbCr(neg_i_gpu, dim=dim)
                sdr_ictcp_i = sdr_ictcp_base.clone()
                sdr_ictcp_i[:, 0, :, :] = neg_ictcp[:, 0, :, :]
                replaced_sdr_i = YCbCr_to_SDR(sdr_ictcp_i, dim=dim)
                im_negs_I_list.append(replaced_sdr_i)

            if len(im_negs_I_list) > 0:
                self.im_negs_I = torch.stack(im_negs_I_list, dim=1)
                self.im_negs_I = apply_random_flip(self.im_negs_I)

        self.im_negs_TP = None
        if data.get('im_negs_TP_raw', None) is not None and data['im_negs_TP_raw'] is not None:
            im_negs_TP_list = []
            for neg_tp_cpu in data['im_negs_TP_raw']:
                neg_tp_gpu = neg_tp_cpu.to(self.device)
                neg_ictcp = SDR_to_YCbCr(neg_tp_gpu, dim=dim)
                sdr_ictcp_tp = sdr_ictcp_base.clone()
                sdr_ictcp_tp[:, 1:3, :, :] = neg_ictcp[:, 1:3, :, :]
                replaced_sdr_tp = YCbCr_to_SDR(sdr_ictcp_tp, dim=dim)
                im_negs_TP_list.append(replaced_sdr_tp)

            if len(im_negs_TP_list) > 0:
                self.im_negs_TP = torch.stack(im_negs_TP_list, dim=1)
                self.im_negs_TP = apply_random_flip(self.im_negs_TP)

        self.im_negs_I_rank = None
        if self.im_negs_I is not None and data.get('im_negs_I_rank', None) is not None:
            self.im_negs_I_rank = data['im_negs_I_rank'].to(self.device)

        self.im_negs_TP_rank = None
        if self.im_negs_TP is not None and data.get('im_negs_TP_rank', None) is not None:
            self.im_negs_TP_rank = data['im_negs_TP_rank'].to(self.device)

        rank_chunks = []
        if self.im_negs_I_rank is not None:
            rank_chunks.append(self.im_negs_I_rank)
        if self.im_negs_TP_rank is not None:
            rank_chunks.append(self.im_negs_TP_rank)
        self.im_negs_rank = torch.cat(rank_chunks, dim=1) if len(rank_chunks) > 0 else None

        self.negs_rgb_cpu = []
        k = max(self.inv_max_versions - 2, 0)
        if k > 0:
            if data.get('im_negs_I_raw', None) is not None:
                self.negs_rgb_cpu.extend(data['im_negs_I_raw'][:k])
            remaining = max(k - len(self.negs_rgb_cpu), 0)
            if remaining > 0 and data.get('im_negs_TP_raw', None) is not None:
                self.negs_rgb_cpu.extend(data['im_negs_TP_raw'][:remaining])
            self.negs_rgb_cpu = self.negs_rgb_cpu[:k]
        if len(self.negs_rgb_cpu) == 0:
            self.negs_rgb_cpu = None


@MODEL_REGISTRY.register()
class ContrastSingleMoCoInvITMModel(ContrastSingleMoCoLatentMixITMModel):
    def _down(self, t: torch.Tensor, size: int):
        if size is None or size <= 0:
            return t
        return F.adaptive_avg_pool2d(t, (size, size))

    @torch.no_grad()
    def _pairwise_dL_dC(self, rgb_versions, eps=1e-8):
        B, N, C, H, W = rgb_versions.shape
        x = rgb_versions.view(B * N, 3, H, W)
        yxy = self._rgb_to_yxy(x, eps=eps)

        Y = torch.log(yxy[:, 0:1] + eps)
        xy = yxy[:, 1:3]

        Y = self._down(Y, self.dist_pool)
        xy = self._down(xy, self.dist_pool)

        _, _, h, w = Y.shape
        HW = h * w

        Yv = Y.contiguous().view(B, N, HW)
        xyv = xy.contiguous().view(B, N, 2 * HW)

        dL = torch.cdist(Yv, Yv, p=1) / HW
        dC = torch.cdist(xyv, xyv, p=1) / (2 * HW)
        return dL, dC

    @torch.no_grad()
    def _soft_weights(self, dL: torch.Tensor, dC: torch.Tensor, eps: float = 1e-8):
        B, N, _ = dL.shape
        eye = torch.eye(N, device=dL.device, dtype=torch.bool).unsqueeze(0)
        eyeB = eye.expand(B, -1, -1)
        offB = ~eyeB

        mean_dL = (dL * offB).sum(dim=(1, 2), keepdim=True) / offB.sum(dim=(1, 2), keepdim=True).clamp_min(1)
        mean_dC = (dC * offB).sum(dim=(1, 2), keepdim=True) / offB.sum(dim=(1, 2), keepdim=True).clamp_min(1)
        std_dL = torch.sqrt((((dL - mean_dL) ** 2) * offB).sum(dim=(1, 2), keepdim=True) / offB.sum(dim=(1, 2), keepdim=True).clamp_min(1) + eps)
        std_dC = torch.sqrt((((dC - mean_dC) ** 2) * offB).sum(dim=(1, 2), keepdim=True) / offB.sum(dim=(1, 2), keepdim=True).clamp_min(1) + eps)

        zL = (dL - mean_dL) / (std_dL + eps)
        zC = (dC - mean_dC) / (std_dC + eps)
        wL = torch.softmax(-zL.masked_fill(eyeB, float('inf')), dim=-1)
        wC = torch.softmax(-zC.masked_fill(eyeB, float('inf')), dim=-1)
        wL = wL.masked_fill(eyeB, 0.0)
        wC = wC.masked_fill(eyeB, 0.0)
        return wL, wC

    def _encode_proj_global_single(self, view_versions: torch.Tensor):
        b, n, c, h, w = view_versions.shape
        flat = view_versions.view(b * n, c, h, w)
        z_0, y_0 = self.feat_extractor.unet_q(flat)
        emb = self.feat_extractor.global_proj_q(z_0.squeeze(-1).squeeze(-1))
        emb = F.normalize(emb, dim=1)
        return emb.view(b, n, -1)

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']
        self.optimizer_g.zero_grad()
        self.optimizer_feat.zero_grad()

        l_total = 0
        loss_dict = OrderedDict()

        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            if current_iter <= train_opt['stage1_iters']:
                z_q, y_q = self.feat_extractor.encode_q(self.im_q)
                self.output = self.net_g(self.lq, z_q, y_q)
                self.output_mid = self.net_g.mid_result

                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                if self.cri_pix_mid:
                    l_pix_mid = self.cri_pix_mid(self.output_mid, self.gt)
                    l_total += l_pix_mid
                    loss_dict['l_pix_mid'] = l_pix_mid
            else:
                if not self.stage2_key_inited:
                    mode = train_opt.get('stage2_init_k', 'keep')
                    self.feat_extractor.stage2_init_key_encoder(mode)
                    logger = get_root_logger()
                    logger.info(f'[INFO] Stage2 key init(single-moco-inv): mode={mode}, iter={current_iter}')
                    self.stage2_key_inited = True

                if train_opt.get('enable_zero_switch', True) and (not self.zero_switched):
                    if hasattr(self.net_g, 'switch_to_zero'):
                        self.net_g.switch_to_zero(use_bf16=self.use_bf16, device=self.device)
                    self.zero_switched = True

                self._warn_if_neg_fallback(current_iter)
                (
                    logits_global_I, labels_global_I,
                    logits_local_I, labels_local_I,
                    logits_global_TP, labels_global_TP,
                    logits_local_TP, labels_local_TP,
                    z_q, y_q,
                ) = self.feat_extractor(
                    self.im_q,
                    self.im_k,
                    self.im_negs_I,
                    self.im_negs_TP,
                    neg_ranks_I=self.im_negs_I_rank,
                    neg_ranks_TP=self.im_negs_TP_rank,
                )

                global_contra_loss_I = self.criterion_contra(logits_global_I, labels_global_I)
                loss_dict['l_global_contra_I'] = global_contra_loss_I
                l_total += global_contra_loss_I * self.cri_contra_weight_global * self.cri_contra_weight

                global_contra_loss_TP = self.criterion_contra(logits_global_TP, labels_global_TP)
                loss_dict['l_global_contra_TP'] = global_contra_loss_TP
                l_total += global_contra_loss_TP * self.cri_contra_weight_global * self.cri_contra_weight

                local_contra_loss_I = self.criterion_contra(logits_local_I, labels_local_I)
                loss_dict['l_local_contra_I'] = local_contra_loss_I
                l_total += local_contra_loss_I * self.cri_contra_weight_local * self.cri_contra_weight

                local_contra_loss_TP = self.criterion_contra(logits_local_TP, labels_local_TP)
                loss_dict['l_local_contra_TP'] = local_contra_loss_TP
                l_total += local_contra_loss_TP * self.cri_contra_weight_local * self.cri_contra_weight

                do_inv = self.use_invariance and (self.inv_every > 0) and (current_iter % self.inv_every == 0)
                if do_inv and (self.negs_rgb_cpu is not None):
                    k = max(self.inv_max_versions - 2, 0)
                    raw_gpu = [t.to(self.device, non_blocking=True) for t in self.negs_rgb_cpu[:k]]
                    if len(raw_gpu) > 0:
                        negs_rgb = torch.stack(raw_gpu, dim=1)
                        neg_views = []
                        if self.im_negs_I is not None:
                            neg_views.append(self.im_negs_I)
                        if self.im_negs_TP is not None:
                            neg_views.append(self.im_negs_TP)
                        neg_view = torch.cat(neg_views, dim=1)[:, :k] if len(neg_views) > 0 else None
                        if neg_view is not None:
                            rgb_versions = torch.cat([self.im_q.unsqueeze(1), self.im_k.unsqueeze(1), negs_rgb], dim=1)
                            view_versions = torch.cat([self.im_q.unsqueeze(1), self.im_k.unsqueeze(1), neg_view], dim=1)
                            if view_versions.size(1) >= 2:
                                with torch.amp.autocast('cuda', enabled=False):
                                    dL, dC = self._pairwise_dL_dC(rgb_versions)
                                    wL, wC = self._soft_weights(dL, dC)
                                emb = self._encode_proj_global_single(view_versions)
                                sim = torch.einsum('bmd,bnd->bmn', emb, emb)
                                diff = 1.0 - sim
                                l_inv_I = (wL * diff).sum() / (wL.sum() + 1e-8)
                                l_inv_TP = (wC * diff).sum() / (wC.sum() + 1e-8)
                                l_total += self.inv_weight * l_inv_I
                                l_total += self.inv_weight * l_inv_TP
                                loss_dict['l_inv_I'] = l_inv_I
                                loss_dict['l_inv_TP'] = l_inv_TP

                self.output = self.net_g(self.lq, z_q, y_q)
                self.output_mid = self.net_g.mid_result

                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                if self.cri_pix_mid:
                    l_pix_mid = self.cri_pix_mid(self.output_mid, self.gt)
                    l_total += l_pix_mid
                    loss_dict['l_pix_mid'] = l_pix_mid

        if self.scaler is not None:
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.step(self.optimizer_feat)
            self.scaler.update()
        else:
            l_total.backward()
            self.optimizer_g.step()
            self.optimizer_feat.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


@MODEL_REGISTRY.register()
class ContrastYXYUnifiedSingleMidSupITMModel(ContrastYXYMidSupITMModel):
    def __init__(self, opt):
        optim_cfg_backup = None
        scheduler_cfg_backup = None
        if opt.get('is_train', False):
            train_opt = opt['train']
            optim_cfg_backup = dict(train_opt['optim_g'])
            scheduler_cfg_backup = dict(train_opt['scheduler'])

        super().__init__(opt)

        self.feat_extractor = build_network(opt['network_feat'])
        self.feat_extractor = self.model_to_device(self.feat_extractor)
        self.print_network(self.feat_extractor)

        if hasattr(self, 'feat_extractor_I'):
            del self.feat_extractor_I
        if hasattr(self, 'feat_extractor_TP'):
            del self.feat_extractor_TP

        load_path_feat = self.opt['path'].get('pretrain_network_g', None)
        if load_path_feat is not None:
            param_key = self.opt['path'].get('param_key_feat_single', 'feat_params_single')
            self.load_network(self.feat_extractor, load_path_feat, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.opt['train']['optim_g'] = optim_cfg_backup
            self.opt['train']['scheduler'] = scheduler_cfg_backup
            self.optimizers = []
            self.schedulers = []
            self.setup_optimizers()
            self.setup_schedulers()

    @classmethod
    def _to_unified3_yxy(cls, rgb: torch.Tensor) -> torch.Tensor:
        return cls._rgb_to_yxy(rgb)

    def _encode_proj_global_single(self, view_versions: torch.Tensor, branch: str):
        b, n, c, h, w = view_versions.shape
        flat = view_versions.view(b * n, c, h, w)
        feat = self.feat_extractor._encode_query_feat(flat)
        z_I, z_TP, _, _ = self.feat_extractor._build_conditions(feat, q_or_k='q')
        emb = z_I if branch == 'I' else z_TP
        emb = F.normalize(emb, dim=1)
        return emb.view(b, n, -1)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        self.im_q = data['im_q'].to(self.device)
        self.im_k = data['im_k'].to(self.device)
        self.im_q_u = self._to_unified3_yxy(self.im_q)
        self.im_k_u = self._to_unified3_yxy(self.im_k)

        self.im_negs = None
        self.negs_rgb_cpu = None
        k = self.inv_max_versions - 2
        if data.get('im_negs_raw', None) is not None and data['im_negs_raw'] is not None:
            view_list = []
            for neg_cpu in data['im_negs_raw']:
                neg = neg_cpu.to(self.device)
                view_list.append(self._to_unified3_yxy(neg))
            if len(view_list) > 0:
                self.im_negs = torch.stack(view_list, dim=1)
                self.im_negs = apply_random_flip(self.im_negs)
            self.negs_rgb_cpu = data['im_negs_raw'][:k] if k > 0 else None

        self.im_negs_rank = None
        if data.get('im_negs_rank', None) is not None:
            self.im_negs_rank = data['im_negs_rank'].to(self.device)

    def feed_data_test(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        self.im_q = data['im_q'].to(self.device)
        self.im_q_u = self._to_unified3_yxy(self.im_q)

    def setup_optimizers(self):
        if not hasattr(self, 'feat_extractor'):
            self.optimizers = []
            return

        train_opt = self.opt['train']
        logger = get_root_logger()

        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger.warning(f'Params {k} in net_g will not be optimized.')

        optim_params_feat = []
        for k, v in self.feat_extractor.named_parameters():
            if v.requires_grad:
                optim_params_feat.append(v)
            else:
                logger.warning(f'Params {k} in feat_extractor will not be optimized.')

        optim_cfg = dict(train_opt['optim_g'])
        optim_type = optim_cfg.pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **optim_cfg)
        self.optimizer_feat = self.get_optimizer(optim_type, optim_params_feat, **optim_cfg)
        self.optimizers.append(self.optimizer_g)
        self.optimizers.append(self.optimizer_feat)

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']
        self.optimizer_g.zero_grad()
        self.optimizer_feat.zero_grad()

        l_total = 0
        loss_dict = OrderedDict()

        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            if current_iter <= train_opt['stage1_iters']:
                z_I, z_TP, y_I, y_TP = self.feat_extractor.encode_q(self.im_q_u)
                self.output = self.net_g(self.lq, z_I.unsqueeze(-1).unsqueeze(-1), z_TP.unsqueeze(-1).unsqueeze(-1), y_I, y_TP)
                self.output_mid = self.net_g.mid_result

                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                if self.cri_pix_mid:
                    l_pix_mid = self.cri_pix_mid(self.output_mid, self.gt)
                    l_total += l_pix_mid
                    loss_dict['l_pix_mid'] = l_pix_mid
            else:
                if not self.stage2_key_inited:
                    mode = train_opt.get('stage2_init_k', 'keep')
                    self.feat_extractor.stage2_init_key_encoder(mode)
                    logger = get_root_logger()
                    logger.info(f'[INFO] Stage2 key init(unified-single): mode={mode}, iter={current_iter}')
                    self.stage2_key_inited = True

                if train_opt.get('enable_zero_switch', True) and (not self.zero_switched):
                    if hasattr(self.net_g, 'switch_to_zero'):
                        self.net_g.switch_to_zero(use_bf16=self.use_bf16, device=self.device)
                    self.zero_switched = True

                (
                    logits_global_I, labels_global_I,
                    logits_local_I, labels_local_I,
                    logits_global_TP, labels_global_TP,
                    logits_local_TP, labels_local_TP,
                    z_I, z_TP, y_I, y_TP,
                ) = self.feat_extractor(self.im_q_u, self.im_k_u, self.im_negs, neg_ranks=self.im_negs_rank)

                global_contra_loss_I = self.criterion_contra(logits_global_I, labels_global_I)
                loss_dict['l_global_contra_I'] = global_contra_loss_I
                l_total += global_contra_loss_I * self.cri_contra_weight_global * self.cri_contra_weight

                global_contra_loss_TP = self.criterion_contra(logits_global_TP, labels_global_TP)
                loss_dict['l_global_contra_TP'] = global_contra_loss_TP
                l_total += global_contra_loss_TP * self.cri_contra_weight_global * self.cri_contra_weight

                local_contra_loss_I = self.criterion_contra(logits_local_I, labels_local_I)
                loss_dict['l_local_contra_I'] = local_contra_loss_I
                l_total += local_contra_loss_I * self.cri_contra_weight_local * self.cri_contra_weight

                local_contra_loss_TP = self.criterion_contra(logits_local_TP, labels_local_TP)
                loss_dict['l_local_contra_TP'] = local_contra_loss_TP
                l_total += local_contra_loss_TP * self.cri_contra_weight_local * self.cri_contra_weight

                do_inv = self.use_invariance and (self.inv_every > 0) and (current_iter % self.inv_every == 0)
                if do_inv and (self.negs_rgb_cpu is not None) and (self.im_negs is not None):
                    k = max(self.inv_max_versions - 2, 0)
                    raw_gpu = [t.to(self.device, non_blocking=True) for t in self.negs_rgb_cpu[:k]]
                    negs_rgb = torch.stack(raw_gpu, dim=1)
                    negs_view = self.im_negs[:, :k] if (self.im_negs is not None and k > 0) else None
                    rgb_versions = torch.cat([self.im_q.unsqueeze(1), self.im_k.unsqueeze(1), negs_rgb], dim=1)
                    view_versions = torch.cat([self.im_q_u.unsqueeze(1), self.im_k_u.unsqueeze(1), negs_view], dim=1)
                    if view_versions.size(1) >= 2:
                        with torch.amp.autocast('cuda', enabled=False):
                            dL, dC = self._pairwise_dL_dC(rgb_versions)
                            wL, wC = self._soft_weights(dL, dC)
                        emb_I = self._encode_proj_global_single(view_versions, branch='I')
                        emb_TP = self._encode_proj_global_single(view_versions, branch='TP')
                        sim_I = torch.einsum('bmd,bnd->bmn', emb_I, emb_I)
                        sim_TP = torch.einsum('bmd,bnd->bmn', emb_TP, emb_TP)
                        diff_I = 1.0 - sim_I
                        diff_TP = 1.0 - sim_TP
                        l_inv_I = (wL * diff_I).sum() / (wL.sum() + 1e-8)
                        l_inv_TP = (wC * diff_TP).sum() / (wC.sum() + 1e-8)
                        l_total += self.inv_weight * l_inv_I
                        l_total += self.inv_weight * l_inv_TP
                        loss_dict['l_inv_I'] = l_inv_I
                        loss_dict['l_inv_TP'] = l_inv_TP

                self.output = self.net_g(self.lq, z_I, z_TP, y_I, y_TP)
                self.output_mid = self.net_g.mid_result

                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                if self.cri_pix_mid:
                    l_pix_mid = self.cri_pix_mid(self.output_mid, self.gt)
                    l_total += l_pix_mid
                    loss_dict['l_pix_mid'] = l_pix_mid

        if self.scaler is not None:
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.step(self.optimizer_feat)
            self.scaler.update()
        else:
            l_total.backward()
            self.optimizer_g.step()
            self.optimizer_feat.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.feat_extractor.eval()
                with torch.no_grad():
                    z_I, z_TP, y_I, y_TP = self.feat_extractor.encode_q(self.im_q_u)
                    self.output = self.net_g_ema(self.lq, z_I.unsqueeze(-1).unsqueeze(-1), z_TP.unsqueeze(-1).unsqueeze(-1), y_I, y_TP)
                    self.output_mid = self.net_g_ema.mid_result
            else:
                self.net_g.eval()
                self.feat_extractor.eval()
                with torch.no_grad():
                    z_I, z_TP, y_I, y_TP = self.feat_extractor.encode_q(self.im_q_u)
                    self.output = self.net_g(self.lq, z_I.unsqueeze(-1).unsqueeze(-1), z_TP.unsqueeze(-1).unsqueeze(-1), y_I, y_TP)
                    self.output_mid = self.net_g.mid_result
                self.net_g.train()

    def save(self, epoch, current_iter):
        nets_to_save = [self.net_g]
        param_keys = ['params']
        if hasattr(self, 'net_g_ema'):
            nets_to_save.append(self.net_g_ema)
            param_keys.append('params_ema')
        nets_to_save.append(self.feat_extractor)
        param_keys.append('feat_params_single')
        self.save_network(net=nets_to_save, net_label='net_g', current_iter=current_iter, param_key=param_keys)
        self.save_training_state(epoch, current_iter)
