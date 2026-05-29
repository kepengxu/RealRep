from typing import List, Tuple
from matplotlib.colors import to_rgb
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import numpy as np
import kornia

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.losses.losses import gradient_penalty_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.hdr_util import HdrConvertBackToRGB
from basicsr.utils.img_util import dilator, thresholder
from .sr_model import SRModel
from basicsr.archs.contrastive import MoCo, MoCoV2, MoCoWithUNet, MoCoWithUNetSharedGlobal, HighlightContrastLoss

import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import os, cv2
import torchvision.transforms as transforms
import torch.autograd.profiler as profiler


@MODEL_REGISTRY.register()
class ITMModel(SRModel):
    """Base ITM model for single image inverse tone mapping."""
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], out_type=np.uint16)
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], out_type=np.uint16)
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
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
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)



@MODEL_REGISTRY.register()
class CondITMModel(ITMModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('color_opt'):
            self.cri_color = build_loss(train_opt['color_opt']).to(self.device)
        else:
            self.cri_color = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_stat is None:
            raise ValueError('No final loss function defined.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        # torch.autograd.set_detect_anomaly(True)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # color loss
        if self.cri_color:
            l_color = self.cri_color(self.output, self.gt)
            l_total += l_color
            loss_dict['l_color'] = l_color

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                # self.output, self.output_mid = self.net_g_ema(self.lq)
                self.output = self.net_g_ema(self.lq, self.lq)
                # self.output_mid = self.net_g_ema.mid_result
        else:
            self.net_g.eval()
            with torch.no_grad():
                # self.output, self.output_mid = self.net_g(self.lq)
                self.output = self.net_g(self.lq, self.lq)
                # self.output_mid = self.net_g.mid_result
            self.net_g.train()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        to_rgb = self.opt['val'].get('to_rgb', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = visuals['result']
            gt_img = visuals['gt'] if 'gt' in visuals else None

            if to_rgb:
                convert_to_rgb = HdrConvertBackToRGB(src=to_rgb, zero_one_normed=True)
                sr_img = convert_to_rgb(sr_img)
                gt_img = convert_to_rgb(gt_img) if gt_img is not None else None

            sr_img = tensor2img([sr_img], out_type=np.uint16)
            metric_data['img'] = sr_img
            if gt_img is not None:
                gt_img = tensor2img([gt_img], out_type=np.uint16)
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')

                imwrite(sr_img, save_img_path)
                # imwrite(sr_img_mid, save_img_mid_path)

            if with_metrics:
                # calculate metrics
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
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        # out_dict['mid_result'] = self.output_mid.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict


@MODEL_REGISTRY.register()
class CondTINYITMModel(CondITMModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()
    
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.tiny = data['tiny'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.tiny)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # color loss
        if self.cri_color:
            l_color = self.cri_color(self.output, self.gt)
            l_total += l_color
            loss_dict['l_color'] = l_color

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                # self.output, self.output_mid = self.net_g_ema(self.lq)
                self.output = self.net_g_ema(self.lq, self.tiny)
                # self.output_mid = self.net_g_ema.mid_result
        else:
            self.net_g.eval()
            with torch.no_grad():
                # self.output, self.output_mid = self.net_g(self.lq)
                self.output = self.net_g(self.lq, self.tiny)
                # self.output_mid = self.net_g.mid_result
            self.net_g.train()

import torch.nn.functional as F
@MODEL_REGISTRY.register()
class SimpleITMModel(ITMModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        self.use_bf16 = self.opt.get('use_bf16', False)  
        if self.use_bf16:
            logger = get_root_logger()
            logger.info('Using bfloat16 mixed precision training ...')  
        
        self.scaler = GradScaler() if self.use_bf16 else None  
        
        if self.is_train:
            self.init_training_settings()
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('color_opt'):
            self.cri_color = build_loss(train_opt['color_opt']).to(self.device)
        else:
            self.cri_color = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_stat is None:
            raise ValueError('No final loss function defined.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        # torch.autograd.set_detect_anomaly(True)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        
        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # color loss
        if self.cri_color:
            l_color = self.cri_color(self.output, self.gt)
            l_total += l_color
            loss_dict['l_color'] = l_color

        if self.scaler is not None:
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            l_total.backward()
            self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        to_rgb = self.opt['val'].get('to_rgb', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
            
        # all_prompts = []
        # prompts_save_dir = osp.join(self.opt['path']['visualization'], dataset_name)
        # os.makedirs(prompts_save_dir, exist_ok=True)

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            lq_img = visuals['lq']
            sr_img = visuals['result']
            gt_img = visuals['gt'] if 'gt' in visuals else None

            if to_rgb:
                convert_to_rgb = HdrConvertBackToRGB(src=to_rgb, zero_one_normed=True)
                sr_img = convert_to_rgb(sr_img)
                gt_img = convert_to_rgb(gt_img) if gt_img is not None else None

            sr_img = tensor2img([sr_img], rgb2bgr=False, out_type=np.uint16)
            metric_data['img'] = sr_img
            if gt_img is not None:
                gt_img = tensor2img([gt_img], rgb2bgr=False, out_type=np.uint16)
                metric_data['img2'] = gt_img
                del self.gt
                
            
            lq_img = tensor2img([lq_img], rgb2bgr=False, out_type=np.uint8)

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            
            

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        pred_save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')
                        
                        gt_save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,'gt',
                                                 f'{img_name}.png')
                        
                        lq_save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,'lq',
                                                 f'{img_name}.png')
                        
                        os.makedirs(osp.join(self.opt['path']['visualization'], dataset_name,'gt'),exist_ok=True)
                        os.makedirs(osp.join(self.opt['path']['visualization'], dataset_name,'lq'),exist_ok=True)
                        
                        
                        
                        

                        imwrite(sr_img, pred_save_img_path)
                        imwrite(gt_img, gt_save_img_path)
                        imwrite(lq_img, lq_save_img_path)
                        # print(pred_save_img_path,gt_save_img_path,lq_save_img_path)
                # imwrite(sr_img_mid, save_img_mid_path)

            if with_metrics:
                # calculate metrics
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
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        # out_dict['mid_result'] = self.output_mid.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict


    
    
@MODEL_REGISTRY.register()
class SimpleContrastITMModel(ITMModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        self.use_bf16 = self.opt.get('use_bf16', False)  
        if self.use_bf16:
            logger = get_root_logger()
            logger.info('Using bfloat16 mixed precision training ...')  
        
        self.scaler = GradScaler() if self.use_bf16 else None  
        
        if self.is_train:
            self.init_training_settings()
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('color_opt'):
            self.cri_color = build_loss(train_opt['color_opt']).to(self.device)
        else:
            self.cri_color = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_stat is None:
            raise ValueError('No final loss function defined.')

        if train_opt.get('contra_opt'):
            self.criterion_contra = torch.nn.CrossEntropyLoss().to(self.device)
            self.cri_contra_weight = train_opt['contra_opt']['loss_weight']

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        # torch.autograd.set_detect_anomaly(True)
        
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        
        self.k = data['im_k'].to(self.device)
    
    def feed_data_test(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        

        

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']
        self.optimizer_g.zero_grad()
        
        l_total = 0
        loss_dict = OrderedDict()
        
        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            if current_iter <= train_opt['stage1_iters']:
                _, output, target, _ = self.net_g.E(x_query=self.lq, x_key=self.k)
                contrast_loss = self.criterion_contra(output, target)
                loss_dict['l_contra'] = contrast_loss
                l_total += contrast_loss * self.cri_contra_weight
            else:
                self.output, output, target = self.net_g(x_query=self.lq, x_key=self.k)
                contrast_loss = self.criterion_contra(output, target)
                loss_dict['l_contra'] = contrast_loss
                l_total += contrast_loss * self.cri_contra_weight
                
                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

        if self.scaler is not None:
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            # 否则直接普通 backward
            l_total.backward()
            self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(x_query=self.lq, x_key=self.lq)
            self.net_g.train()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        to_rgb = self.opt['val'].get('to_rgb', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data_test(val_data)
            self.test()

            visuals = self.get_current_visuals()
            lq_img = visuals['lq']
            sr_img = visuals['result']
            gt_img = visuals['gt'] if 'gt' in visuals else None

            if to_rgb:
                convert_to_rgb = HdrConvertBackToRGB(src=to_rgb, zero_one_normed=True)
                sr_img = convert_to_rgb(sr_img)
                gt_img = convert_to_rgb(gt_img) if gt_img is not None else None

            sr_img = tensor2img([sr_img], rgb2bgr=False, out_type=np.uint16)
            metric_data['img'] = sr_img
            if gt_img is not None:
                gt_img = tensor2img([gt_img], rgb2bgr=False, out_type=np.uint16)
                metric_data['img2'] = gt_img
                del self.gt
                
            
            lq_img = tensor2img([lq_img], rgb2bgr=False, out_type=np.uint8)

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            # if save_img:
            #     if self.opt['is_train']:
            #         save_img_path = osp.join(self.opt['path']['visualization'], img_name,
            #                                  f'{img_name}_{current_iter}.png')
            #     else:
            #         if self.opt['val']['suffix']:
            #             save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
            #                                      f'{img_name}_{self.opt["val"]["suffix"]}.png')
            #         else:
            #             save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
            #                                      f'{img_name}_{self.opt["name"]}.png')

            #     imwrite(sr_img, save_img_path)
            #     # imwrite(sr_img_mid, save_img_mid_path)
                
            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        pred_save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')
                        
                        gt_save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,'gt',
                                                 f'{img_name}.png')
                        
                        lq_save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,'lq',
                                                 f'{img_name}.png')
                        
                        os.makedirs(osp.join(self.opt['path']['visualization'], dataset_name,'gt'),exist_ok=True)
                        os.makedirs(osp.join(self.opt['path']['visualization'], dataset_name,'lq'),exist_ok=True)
                        
                        
                        
                        

                        imwrite(sr_img, pred_save_img_path)
                        imwrite(gt_img, gt_save_img_path)
                        imwrite(lq_img, lq_save_img_path)
                        print(pred_save_img_path,gt_save_img_path,lq_save_img_path)

            if with_metrics:
                # calculate metrics
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
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        # out_dict['mid_result'] = self.output_mid.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

            
    
    

from basicsr.utils.ICTCP_convert import SDR_to_YCbCr, YCbCr_to_SDR
from .pcgrad import PCGrad
import torchvision.transforms.functional as tF
import random
def apply_random_flip(batch_tensor, horizontal_flip=True, vertical_flip=True):
    transformed = []
    for img in batch_tensor:
        if horizontal_flip and random.random() > 0.5:
            img = tF.hflip(img)
        if vertical_flip and random.random() > 0.5:
            img = tF.vflip(img)
        transformed.append(img)
    return torch.stack(transformed)

def save_tensor_as_image(tensor, filename, normalize=True):
    tensor = tensor.detach().cpu()
    if normalize:
        tensor = torch.clamp(tensor, 0, 1)  
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor)
    img.save(filename)

@MODEL_REGISTRY.register()
class ContrastYCBCRMidSupITMModelshared(ITMModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        
        self.feat_extractor = MoCoWithUNet(in_channels=3, global_cond_channels=64, spatial_cond_channels=16, temperature=1.0)
        self.feat_extractor = self.model_to_device(self.feat_extractor)
        self.print_network(self.feat_extractor)
        
        # self.scaler = GradScaler()
        self.average_metrics = {}
        
        load_path = self.opt['path'].get('pretrain_network_g', None)
        load_path_feat = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        if load_path_feat is not None:
            param_key = self.opt['path'].get('param_key_feat', 'feat_params')
            self.load_network(self.feat_extractor, load_path_feat, self.opt['path'].get('strict_load_g', True), param_key)
        
        self.use_bf16 = self.opt.get('use_bf16', False)

        if self.is_train:
            self.init_training_settings()
        
        if self._check_if_zero_initialized(self.net_g):
            self.zero_switched = True
            # print('检测到网络包含 ZeroConv2d / ZeroLinear，设置 zero_switched=True')
        else:
            self.zero_switched = False
            # print('网络尚未切换为零卷积，zero_switched=False，将在阶段2调用 switch_to_zero()')

    
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
        
        self.scaler = GradScaler() if self.use_bf16 else None  
        
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            load_path_feat = self.opt['path'].get('pretrain_network_g', None)
            
    
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        
        # define losses
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

            if mid_sup_opt.get('perceptual_opt'):
                self.cri_perceptual_mid = build_loss(mid_sup_opt['perceptual_opt']).to(self.device)
            else:
                self.cri_perceptual_mid = None

            if mid_sup_opt.get('color_opt'):
                self.cri_color_mid = build_loss(mid_sup_opt['color_opt']).to(self.device)
            else:
                self.cri_color_mid = None
        
        if train_opt.get('contra_opt'):
            self.criterion_contra = torch.nn.CrossEntropyLoss().to(self.device)
            self.cri_contra_weight = train_opt['contra_opt']['loss_weight']
            self.cri_contra_weight_global = train_opt['contra_opt']['loss_weight_global']
            self.cri_contra_weight_local = train_opt['contra_opt']['loss_weight_local']
        else:
            self.criterion_contra = None
        
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        """
        data: 来自 DataLoader 的一个 batch (通常是 batch_size=1 或更多)
            data['lq'], data['im_negs_I_raw'], data['im_negs_TP_raw'] 等，都是 CPU tensor/list
        """
        self.lq = data['lq'].to(self.device)  # [B, 3, H, W]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        
        self.im_q = data['im_q'].to(self.device)
        self.im_k = data['im_k'].to(self.device)


        dim = 1  
        sdr_ictcp_base = SDR_to_YCbCr(self.lq, dim=dim)  

        # --------------- 亮度负样本 ---------------
        self.im_negs_I = None
        if data['im_negs_I_raw'] is not None:
            im_negs_I_list = []  
            for i, neg_i_cpu in enumerate(data['im_negs_I_raw']):
                neg_i_gpu = neg_i_cpu.to(self.device)  
                
                neg_ictcp = SDR_to_YCbCr(neg_i_gpu, dim=dim)  
                
                sdr_ictcp_i = sdr_ictcp_base.clone()
                
                sdr_ictcp_i[:, 0, :, :] = neg_ictcp[:, 0, :, :]
                
                replaced_sdr_i = YCbCr_to_SDR(sdr_ictcp_i, dim=dim)  
                im_negs_I_list.append(replaced_sdr_i)
            
            if len(im_negs_I_list) > 0:
                self.im_negs_I = torch.stack(im_negs_I_list, dim=1)  
                if self.im_negs_I.size(1) == 1:
                    self.im_negs_I = self.im_negs_I.squeeze(1)  
                self.im_negs_I = apply_random_flip(self.im_negs_I)
        
        # --------------- 色度负样本 ---------------
        self.im_negs_TP = None
        if data['im_negs_TP_raw'] is not None:
            im_negs_TP_list = []
            for i, neg_tp_cpu in enumerate(data['im_negs_TP_raw']):
                neg_tp_gpu = neg_tp_cpu.to(self.device)  
                
                neg_ictcp = SDR_to_YCbCr(neg_tp_gpu, dim=dim)  
                
                sdr_ictcp_tp = sdr_ictcp_base.clone()
                
                sdr_ictcp_tp[:, 1:3, :, :] = neg_ictcp[:, 1:3, :, :]
                
                replaced_sdr_tp = YCbCr_to_SDR(sdr_ictcp_tp, dim=dim)  
                im_negs_TP_list.append(replaced_sdr_tp)
            
            if len(im_negs_TP_list) > 0:
                self.im_negs_TP = torch.stack(im_negs_TP_list, dim=1)  
                if self.im_negs_TP.size(1) == 1:
                    self.im_negs_TP = self.im_negs_TP.squeeze(1)  
                self.im_negs_TP = apply_random_flip(self.im_negs_TP)


    
    def feed_data_test(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        self.im_q = data['im_q'].to(self.device)

        
    
    def setup_optimizers(self):
        train_opt = self.opt['train']
        
        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} in net_g will not be optimized.')

        optim_params_feat = []
        for k, v in self.feat_extractor.named_parameters():
            if v.requires_grad:
                optim_params_feat.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} in feat_extractor will not be optimized.')
                
        
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt['optim_g'])
        self.optimizer_feat = self.get_optimizer(optim_type, optim_params_feat, **train_opt['optim_g'])
        
        self.optimizers.append(self.optimizer_g)
        self.optimizers.append(self.optimizer_feat)
    
    
    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']
        self.optimizer_g.zero_grad()
        self.optimizer_feat_I.zero_grad()
        self.optimizer_feat_TP.zero_grad()
        
        l_total = 0
        loss_dict = OrderedDict()

        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            if current_iter <= train_opt['stage1_iters']:  # 阶段 1

                z_I, y_I = self.feat_extractor.unet_q(self.im_q, mode='I')   
                z_TP, y_TP = self.feat_extractor.unet_q(self.im_q, mode='TP') 
                self.output = self.net_g(self.lq, z_I, z_TP, y_I, y_TP)
                self.output_mid = self.net_g.mid_result

                # pixel loss
                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                # intermediate supervision loss
                if self.cri_pix_mid:
                    l_pix_mid = self.cri_pix_mid(self.output_mid, self.gt)
                    l_total += l_pix_mid
                    loss_dict['l_pix_mid'] = l_pix_mid
                
                

            else:  # 阶段 2

                if not self.zero_switched:
                    if hasattr(self.net_g, "switch_to_zero"):
                        self.net_g.switch_to_zero(use_bf16=self.use_bf16, device=self.device)
                        print(f"[INFO] switch_to_zero() called at iteration {current_iter} with BF16={self.use_bf16}, device={self.device}")
                    self.zero_switched = True
                

                logits_global_I, labels_global_I, logits_local_I, labels_local_I, z_I, y_I = self.feat_extractor(self.im_q, self.im_k, self.im_negs_I, mode='I')
                
                logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP, z_TP, y_TP = self.feat_extractor(self.im_q, self.im_k, self.im_negs_TP, mode='TP')

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

                self.output = self.net_g(self.lq, z_I, z_TP, y_I, y_TP)
                self.output_mid = self.net_g.mid_result

                # pixel loss
                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                # intermediate supervision loss
                if self.cri_pix_mid:
                    l_pix_mid = self.cri_pix_mid(self.output_mid, self.gt)
                    l_total += l_pix_mid
                    loss_dict['l_pix_mid'] = l_pix_mid
                

        if self.scaler is not None:
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.step(self.optimizer_feat_I)
            self.scaler.step(self.optimizer_feat_TP)
            self.scaler.update()
        else:
            l_total.backward()
            self.optimizer_g.step()
            self.optimizer_feat_I.step()
            self.optimizer_feat_TP.step()


        # 保存日志
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):  # (###### [FP16 CHANGE])
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.feat_extractor.eval()
                with torch.no_grad():
                    z_I, y_I = self.feat_extractor.unet_k(self.im_q, mode='I')
                    z_TP, y_TP = self.feat_extractor.unet_k(self.im_q, mode='TP')
                    self.output = self.net_g_ema(self.lq, z_I, z_TP, y_I, y_TP)
                    self.output_mid = self.net_g_ema.mid_result
            else:
                self.net_g.eval()
                self.feat_extractor.eval()
                with torch.no_grad():
                    z_I, y_I = self.feat_extractor.unet_k(self.im_q, mode='I')
                    z_TP, y_TP = self.feat_extractor.unet_k(self.im_q, mode='TP')
                    self.output = self.net_g(self.lq, z_I, z_TP, y_I, y_TP)
                    self.output_mid = self.net_g.mid_result
                self.net_g.train()
    
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        to_rgb = self.opt['val'].get('to_rgb', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            # self.save_batch_images(val_data, idx)
            self.feed_data_test(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = visuals['result']
            sr_img_mid = visuals['mid_result']
            gt_img = visuals['gt'] if 'gt' in visuals else None

            sr_img = torch.clamp(sr_img, 0.0, 1.0)
            gt_img = torch.clamp(gt_img, 0.0, 1.0)
            
            if to_rgb:
                convert_to_rgb = HdrConvertBackToRGB(src=to_rgb, zero_one_normed=True)
                sr_img = convert_to_rgb(sr_img)
                sr_img_mid = convert_to_rgb(sr_img_mid)
                gt_img = convert_to_rgb(gt_img) if gt_img is not None else None

            sr_img = tensor2img([sr_img], out_type=np.uint16)
            sr_img_mid = tensor2img([sr_img_mid])
            metric_data['img'] = sr_img
            if gt_img is not None:
                gt_img = tensor2img([gt_img], out_type=np.uint16)
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                    save_img_mid_path = osp.join(self.opt['path']['visualization'], img_name,
                                                 f'{img_name}_{current_iter}_mid.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        save_img_mid_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["val"]["suffix"]}_mid.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                        save_img_mid_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["name"]}_mid.png')
                imwrite(sr_img, save_img_path)
                imwrite(sr_img_mid, save_img_mid_path)

            if with_metrics:
                # calculate metrics
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
                
                if metric not in self.average_metrics:
                    self.average_metrics[metric] = {'total': 0, 'count': 0}
                self.average_metrics[metric]['total'] += self.metric_results[metric]
                self.average_metrics[metric]['count'] += 1
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
    
    def print_final_metrics(self):
        # Calculate and print the mean metric across datasets
        print("\nFinal Average Metrics Across All Datasets:")
        for metric, vals in self.average_metrics.items():
            mean_metric = vals['total'] / vals['count']
            print(f"{metric}: {mean_metric:.4f}")
    def save(self, epoch, current_iter):
        nets_to_save = [self.net_g]
        param_keys = ['params']

        if hasattr(self, 'net_g_ema'):
            nets_to_save.append(self.net_g_ema)
            param_keys.append('params_ema')

        nets_to_save.append(self.feat_extractor)
        param_keys.append('feat_params')
        

        self.save_network(
            net=nets_to_save,
            net_label='net_g',
            current_iter=current_iter,
            param_key=param_keys
        )

        self.save_training_state(epoch, current_iter)
    
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['mid_result'] = self.output_mid.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict


@MODEL_REGISTRY.register()
class ContrastYCBCRMidSupITMModel(ITMModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        
        self.feat_extractor_I = MoCoWithUNet(in_channels=3, global_cond_channels=64, spatial_cond_channels=16, temperature=1.0)
        self.feat_extractor_I = self.model_to_device(self.feat_extractor_I)
        self.print_network(self.feat_extractor_I)
        
        self.feat_extractor_TP = MoCoWithUNet(in_channels=3, global_cond_channels=64, spatial_cond_channels=16, temperature=1.0)
        self.feat_extractor_TP = self.model_to_device(self.feat_extractor_TP)
        self.print_network(self.feat_extractor_TP)

        
        # self.scaler = GradScaler()
        self.average_metrics = {}
        
        load_path = self.opt['path'].get('pretrain_network_g', None)
        load_path_feat_I = self.opt['path'].get('pretrain_network_g', None)
        load_path_feat_TP = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        if load_path_feat_I is not None:
            param_key = self.opt['path'].get('param_key_feat_I', 'feat_params_I')
            self.load_network(self.feat_extractor_I, load_path_feat_I, self.opt['path'].get('strict_load_g', True), param_key)
        if load_path_feat_TP is not None:
            param_key = self.opt['path'].get('param_key_feat_TP', 'feat_params_TP')
            self.load_network(self.feat_extractor_TP, load_path_feat_TP, self.opt['path'].get('strict_load_g', True), param_key)
        
        self.use_bf16 = self.opt.get('use_bf16', False)

        self.stage2_key_inited = False

        if self.is_train:
            self.init_training_settings()
        
        if self._check_if_zero_initialized(self.net_g):
            self.zero_switched = True
        else:
            self.zero_switched = False

    
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
        
        self.scaler = GradScaler() if self.use_bf16 else None  
        
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # self.feat_extractor_ema = MoCo(in_channels=3, out_channels=32, dim=64, temperature=1.0).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            load_path_feat = self.opt['path'].get('pretrain_network_g', None)
            

    
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
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

            if mid_sup_opt.get('perceptual_opt'):
                self.cri_perceptual_mid = build_loss(mid_sup_opt['perceptual_opt']).to(self.device)
            else:
                self.cri_perceptual_mid = None

            if mid_sup_opt.get('color_opt'):
                self.cri_color_mid = build_loss(mid_sup_opt['color_opt']).to(self.device)
            else:
                self.cri_color_mid = None
        
        if train_opt.get('contra_opt'):
            self.criterion_contra = torch.nn.CrossEntropyLoss().to(self.device)
            self.cri_contra_weight = train_opt['contra_opt']['loss_weight']
            self.cri_contra_weight_global = train_opt['contra_opt']['loss_weight_global']
            self.cri_contra_weight_local = train_opt['contra_opt']['loss_weight_local']
        else:
            self.criterion_contra = None
        
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)  # [B, 3, H, W]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        
        self.im_q = data['im_q'].to(self.device)
        self.im_k = data['im_k'].to(self.device)

        dim = 1  
        sdr_ictcp_base = SDR_to_YCbCr(self.lq, dim=dim)  # shape [B, 3, H, W]

        # --------------- 亮度负样本 ---------------
        self.im_negs_I = None
        if data['im_negs_I_raw'] is not None:
            im_negs_I_list = [] 
            for i, neg_i_cpu in enumerate(data['im_negs_I_raw']):
      
                neg_i_gpu = neg_i_cpu.to(self.device)  
                
                neg_ictcp = SDR_to_YCbCr(neg_i_gpu, dim=dim)  
                
                sdr_ictcp_i = sdr_ictcp_base.clone()
                
                sdr_ictcp_i[:, 0, :, :] = neg_ictcp[:, 0, :, :]
                
                replaced_sdr_i = YCbCr_to_SDR(sdr_ictcp_i, dim=dim) 
                im_negs_I_list.append(replaced_sdr_i)
            
            if len(im_negs_I_list) > 0:
                self.im_negs_I = torch.stack(im_negs_I_list, dim=1)  
                if self.im_negs_I.size(1) == 1:
                    self.im_negs_I = self.im_negs_I.squeeze(1)  
                self.im_negs_I = apply_random_flip(self.im_negs_I)
        
        # --------------- 色度负样本 ---------------
        self.im_negs_TP = None
        if data['im_negs_TP_raw'] is not None:
            im_negs_TP_list = []
            for i, neg_tp_cpu in enumerate(data['im_negs_TP_raw']):
                neg_tp_gpu = neg_tp_cpu.to(self.device)  
                
                neg_ictcp = SDR_to_YCbCr(neg_tp_gpu, dim=dim)  
                
                sdr_ictcp_tp = sdr_ictcp_base.clone()
                
                sdr_ictcp_tp[:, 1:3, :, :] = neg_ictcp[:, 1:3, :, :]
                
                replaced_sdr_tp = YCbCr_to_SDR(sdr_ictcp_tp, dim=dim)  
                im_negs_TP_list.append(replaced_sdr_tp)
            
            if len(im_negs_TP_list) > 0:
                self.im_negs_TP = torch.stack(im_negs_TP_list, dim=1)  
                if self.im_negs_TP.size(1) == 1:
                    self.im_negs_TP = self.im_negs_TP.squeeze(1)  
                self.im_negs_TP = apply_random_flip(self.im_negs_TP)

    
    def feed_data_test(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        self.im_q = data['im_q'].to(self.device)
    
    @staticmethod
    def save_batch_images(batch_data, idx, output_dir="test_batch"):
        os.makedirs(output_dir, exist_ok=True)
        to_pil = transforms.ToPILImage()
        batch_size = batch_data['lq'].shape[0]

        for i in range(batch_size):
            # 获取 lq 和 gt 图像
            lq_image = batch_data['lq'][i]
            gt_image = batch_data['gt'][i]

            # 保存 lq 图像
            lq_img = to_pil(lq_image)
            lq_img.save(os.path.join(output_dir, f"lq_image_iter_{idx}_sample_{i}.png"))

            # 保存 gt 图像
            gt_img = to_pil(gt_image)
            gt_img.save(os.path.join(output_dir, f"gt_image_iter_{idx}_sample_{i}.png"))
        
    
    def setup_optimizers(self):
        train_opt = self.opt['train']
        
        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} in net_g will not be optimized.')

        optim_params_feat_I = []
        for k, v in self.feat_extractor_I.named_parameters():
            if v.requires_grad:
                optim_params_feat_I.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} in feat_extractor will not be optimized.')
                
        optim_params_feat_TP = []
        for k, v in self.feat_extractor_TP.named_parameters():
            if v.requires_grad:
                optim_params_feat_TP.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} in feat_extractor will not be optimized.')
        
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt['optim_g'])
        self.optimizer_feat_I = self.get_optimizer(optim_type, optim_params_feat_I, **train_opt['optim_g'])
        self.optimizer_feat_TP = self.get_optimizer(optim_type, optim_params_feat_TP, **train_opt['optim_g'])
        
        self.optimizers.append(self.optimizer_g)
        self.optimizers.append(self.optimizer_feat_I)
        self.optimizers.append(self.optimizer_feat_TP)
    
    def save_visualizations(self, current_iter, save_dir='saved_images'):
        os.makedirs(save_dir, exist_ok=True)
        
        iter_suffix = f'iter_{current_iter}'
        
        idx = 0
        
        # 保存 im_q
        if hasattr(self, 'im_q'):
            im_q = self.im_q[idx]
            save_tensor_as_image(im_q, os.path.join(save_dir, f'im_q_{iter_suffix}.png'))
        
        # 保存 lq
        if hasattr(self, 'lq'):
            lq = self.lq[idx]
            save_tensor_as_image(lq, os.path.join(save_dir, f'lq_{iter_suffix}.png'))
        
        # 保存 mid_result
        if hasattr(self, 'output_mid'):
            mid_result = self.output_mid[idx]
            save_tensor_as_image(mid_result, os.path.join(save_dir, f'mid_result_{iter_suffix}.png'))
        
        # 保存 output
        if hasattr(self, 'output'):
            output = self.output[idx]
            save_tensor_as_image(output, os.path.join(save_dir, f'output_{iter_suffix}.png'))
        
        # 保存 gt（如果存在）
        if hasattr(self, 'gt'):
            gt = self.gt[idx]
            save_tensor_as_image(gt, os.path.join(save_dir, f'gt_{iter_suffix}.png'))

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']

        self.optimizer_g.zero_grad()
        self.optimizer_feat_I.zero_grad()
        self.optimizer_feat_TP.zero_grad()
        
        l_total = 0
        loss_dict = OrderedDict()

        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            if current_iter <= train_opt['stage1_iters']:  
                z_I, y_I = self.feat_extractor_I.unet_q(self.im_q)  
                z_TP, y_TP = self.feat_extractor_TP.unet_q(self.im_q) 

                self.output = self.net_g(self.lq, z_I, z_TP, y_I, y_TP)
                self.output_mid = self.net_g.mid_result

                # pixel loss
                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                # intermediate supervision loss
                if self.cri_pix_mid:
                    l_pix_mid = self.cri_pix_mid(self.output_mid, self.gt)
                    l_total += l_pix_mid
                    loss_dict['l_pix_mid'] = l_pix_mid
                
                # log_interval = train_opt.get('log_interval', 1)  # 默认每100次迭代保存一次
                # if current_iter % log_interval == 0:
                #     self.save_visualizations(current_iter, save_dir='saved_images_stage1')
                
                # self.feat_extractor_I._momentum_update_key_encoder()
                # self.feat_extractor_TP._momentum_update_key_encoder()
                
                # self.feat_extractor_I.reset_unet_k()
                # self.feat_extractor_TP.reset_unet_k()


            else:  
                # stage2 init mode
                # if not self.stage2_key_inited:
                #     mode = train_opt.get('stage2_init_k', 'keep')   # keep/copy_q/reinit_unet/reinit_all
                #     self.feat_extractor_I.stage2_init_key_encoder(mode)
                #     self.feat_extractor_TP.stage2_init_key_encoder(mode)
                #     print(f"[INFO] Stage2 key init: mode={mode}, iter={current_iter}")
                #     self.stage2_key_inited = True

                # # if not self.zero_switched:
                # if train_opt.get('enable_zero_switch', True) and (not self.zero_switched):
                #     if hasattr(self.net_g, "switch_to_zero"):
                #         self.net_g.switch_to_zero(use_bf16=self.use_bf16, device=self.device)
                #         print(f"[INFO] switch_to_zero() called at iteration {current_iter} with BF16={self.use_bf16}, device={self.device}")
                #     self.zero_switched = True

                if not self.stage2_key_inited:
                    mode = train_opt.get('stage2_init_k', 'keep')   # keep/copy_q/reinit_unet/reinit_all
                    self.feat_extractor_I.stage2_init_key_encoder(mode)
                    self.feat_extractor_TP.stage2_init_key_encoder(mode)
                    logger = get_root_logger()
                    logger.info(f"[INFO] Stage2 key init: mode={mode}, iter={current_iter}")
                    self.stage2_key_inited = True

                if train_opt.get('enable_zero_switch', True) and (not self.zero_switched):
                    if hasattr(self.net_g, "switch_to_zero"):
                        self.net_g.switch_to_zero(use_bf16=self.use_bf16, device=self.device)
                        logger = get_root_logger()
                        logger.info(
                            f"[INFO] switch_to_zero() called at iteration {current_iter} "
                            f"with BF16={self.use_bf16}, device={self.device}"
                        )
                    self.zero_switched = True

                

                logits_global_I, labels_global_I, logits_local_I, labels_local_I, z_I, y_I = self.feat_extractor_I(self.im_q, self.im_k, self.im_negs_I)
                
                logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP, z_TP, y_TP = self.feat_extractor_TP(self.im_q, self.im_k, self.im_negs_TP)

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


                self.output = self.net_g(self.lq, z_I, z_TP, y_I, y_TP)
                self.output_mid = self.net_g.mid_result

                # pixel loss
                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                # intermediate supervision loss
                if self.cri_pix_mid:
                    l_pix_mid = self.cri_pix_mid(self.output_mid, self.gt)
                    l_total += l_pix_mid
                    loss_dict['l_pix_mid'] = l_pix_mid
                    

                
        if self.scaler is not None:
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.step(self.optimizer_feat_I)
            self.scaler.step(self.optimizer_feat_TP)
            self.scaler.update()
        else:
            l_total.backward()
            self.optimizer_g.step()
            self.optimizer_feat_I.step()
            self.optimizer_feat_TP.step()

       

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
            # self.contrast_model_ema(decay=self.ema_decay)

    def test(self):
        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):  # (###### [FP16 CHANGE])
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.feat_extractor_I.eval()
                self.feat_extractor_TP.eval()
                with torch.no_grad():
                  
                    z_I, y_I = self.feat_extractor_I.unet_k(self.im_q)
                    z_TP, y_TP = self.feat_extractor_TP.unet_k(self.im_q)
                    
                    self.output = self.net_g_ema(self.lq, z_I, z_TP, y_I, y_TP)
                    self.output_mid = self.net_g_ema.mid_result
            else:
                self.net_g.eval()
                self.feat_extractor_I.eval()
                self.feat_extractor_TP.eval()
                with torch.no_grad():
                    z_I, y_I = self.feat_extractor_I.unet_k(self.im_q)
                    z_TP, y_TP = self.feat_extractor_TP.unet_k(self.im_q)
                    self.output = self.net_g(self.lq, z_I, z_TP, y_I, y_TP)
                    self.output_mid = self.net_g.mid_result
                self.net_g.train()
    
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        to_rgb = self.opt['val'].get('to_rgb', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            # self.save_batch_images(val_data, idx)
            self.feed_data_test(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = visuals['result']
            sr_img_mid = visuals['mid_result']
            gt_img = visuals['gt'] if 'gt' in visuals else None

            sr_img = torch.clamp(sr_img, 0.0, 1.0)
            gt_img = torch.clamp(gt_img, 0.0, 1.0)
            
            if to_rgb:
                convert_to_rgb = HdrConvertBackToRGB(src=to_rgb, zero_one_normed=True)
                sr_img = convert_to_rgb(sr_img)
                sr_img_mid = convert_to_rgb(sr_img_mid)
                gt_img = convert_to_rgb(gt_img) if gt_img is not None else None

            sr_img = tensor2img([sr_img], out_type=np.uint16)
            sr_img_mid = tensor2img([sr_img_mid])
            metric_data['img'] = sr_img
            if gt_img is not None:
                gt_img = tensor2img([gt_img], out_type=np.uint16)
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                    save_img_mid_path = osp.join(self.opt['path']['visualization'], img_name,
                                                 f'{img_name}_{current_iter}_mid.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        save_img_mid_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["val"]["suffix"]}_mid.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                        save_img_mid_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["name"]}_mid.png')
                imwrite(sr_img, save_img_path)
                imwrite(sr_img_mid, save_img_mid_path)

            if with_metrics:
                # calculate metrics
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
                
                if metric not in self.average_metrics:
                    self.average_metrics[metric] = {'total': 0, 'count': 0}
                self.average_metrics[metric]['total'] += self.metric_results[metric]
                self.average_metrics[metric]['count'] += 1
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
    
    def print_final_metrics(self):
        # Calculate and print the mean metric across datasets
        print("\nFinal Average Metrics Across All Datasets:")
        for metric, vals in self.average_metrics.items():
            mean_metric = vals['total'] / vals['count']
            print(f"{metric}: {mean_metric:.4f}")
    def save(self, epoch, current_iter):
        nets_to_save = [self.net_g]
        param_keys = ['params']

        if hasattr(self, 'net_g_ema'):
            nets_to_save.append(self.net_g_ema)
            param_keys.append('params_ema')

        nets_to_save.append(self.feat_extractor_I)
        param_keys.append('feat_params_I')
        
        nets_to_save.append(self.feat_extractor_TP)
        param_keys.append('feat_params_TP')

        self.save_network(
            net=nets_to_save,
            net_label='net_g',
            current_iter=current_iter,
            param_key=param_keys
        )

        self.save_training_state(epoch, current_iter)
    
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['mid_result'] = self.output_mid.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict




@MODEL_REGISTRY.register()
class ContrastYXYMidSupITMModel(ITMModel):
    """
    Pure-view dual-encoder model using CIE Yxy:
      - I branch (luma): input = Y only (replicated to 3ch)
      - TP branch (chroma): input = xy only (padded to 3ch as [0, x, y])
    No pixel-domain swap is used here.
    """
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # keep in_channels=3, we will convert Y (1ch) -> 3ch and xy (2ch) -> 3ch
        self.feat_extractor_I = MoCoWithUNet(in_channels=3, global_cond_channels=64, spatial_cond_channels=16, temperature=1.0)
        self.feat_extractor_I = self.model_to_device(self.feat_extractor_I)
        self.print_network(self.feat_extractor_I)

        self.feat_extractor_TP = MoCoWithUNet(in_channels=3, global_cond_channels=64, spatial_cond_channels=16, temperature=1.0)
        self.feat_extractor_TP = self.model_to_device(self.feat_extractor_TP)
        self.print_network(self.feat_extractor_TP)

        self.average_metrics = {}

        load_path = self.opt['path'].get('pretrain_network_g', None)
        load_path_feat_I = self.opt['path'].get('pretrain_network_g', None)
        load_path_feat_TP = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        if load_path_feat_I is not None:
            param_key = self.opt['path'].get('param_key_feat_I', 'feat_params_I')
            self.load_network(self.feat_extractor_I, load_path_feat_I, self.opt['path'].get('strict_load_g', True), param_key)
        if load_path_feat_TP is not None:
            param_key = self.opt['path'].get('param_key_feat_TP', 'feat_params_TP')
            self.load_network(self.feat_extractor_TP, load_path_feat_TP, self.opt['path'].get('strict_load_g', True), param_key)

        self.use_bf16 = self.opt.get('use_bf16', False)
        self.stage2_key_inited = False

        if self.is_train:
            self.init_training_settings()

        if self._check_if_zero_initialized(self.net_g):
            self.zero_switched = True
        else:
            self.zero_switched = False

    # -------------------------
    # Yxy helpers (pure-view)
    # -------------------------
    @staticmethod
    def _srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
        """
        rgb: [B,3,H,W] in [0,1] sRGB (gamma-encoded)
        return linear rgb in [0,1]
        """
        a = 0.055
        return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1 + a)) ** 2.4)

    @staticmethod
    def _linear_rgb_to_xyz(lin_rgb: torch.Tensor) -> torch.Tensor:
        """
        lin_rgb: [B,3,H,W] linear sRGB, D65
        xyz: [B,3,H,W]
        """
        # sRGB(D65) -> XYZ matrix
        M = lin_rgb.new_tensor([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ])  # [3,3]
        # reshape to do matrix multiply per pixel
        B, C, H, W = lin_rgb.shape
        rgb_flat = lin_rgb.permute(0, 2, 3, 1).reshape(-1, 3)          # [BHW,3]
        xyz_flat = rgb_flat @ M.T                                       # [BHW,3]
        xyz = xyz_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)          # [B,3,H,W]
        return xyz

    @classmethod
    def _rgb_to_yxy(cls, rgb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        rgb: [B,3,H,W] in [0,1] (assumed sRGB)
        return yxy: [B,3,H,W] where channels are [Y, x, y]
        """
        lin = cls._srgb_to_linear(torch.clamp(rgb, 0.0, 1.0))
        xyz = cls._linear_rgb_to_xyz(lin)
        X = xyz[:, 0:1, :, :]
        Y = xyz[:, 1:2, :, :]
        Z = xyz[:, 2:3, :, :]
        denom = X + Y + Z + eps
        x = X / denom
        y = Y / denom
        return torch.cat([Y, x, y], dim=1)

    @classmethod
    def _to_luma3_yxy(cls, rgb: torch.Tensor) -> torch.Tensor:
        """RGB -> Yxy -> take Y (1ch) -> repeat to 3ch"""
        yxy = cls._rgb_to_yxy(rgb)
        Y = yxy[:, 0:1, :, :]
        return Y.repeat(1, 3, 1, 1)

    @classmethod
    def _to_chroma3_yxy(cls, rgb: torch.Tensor) -> torch.Tensor:
        """RGB -> Yxy -> take xy (2ch) -> pad to 3ch [0,x,y]"""
        yxy = cls._rgb_to_yxy(rgb)
        xy = yxy[:, 1:3, :, :]  # [B,2,H,W]
        pad = torch.zeros_like(xy[:, 0:1, :, :])
        return torch.cat([pad, xy], dim=1)  # [B,3,H,W]

    def _down(self, t: torch.Tensor, size: int):
        if size is None or size <= 0:
            return t
        return F.adaptive_avg_pool2d(t, (size, size))

    # @torch.no_grad()
    # def _pairwise_dL_dC(self, rgb_versions: torch.Tensor, eps: float = 1e-8):
    #     """
    #     rgb_versions: [B,N,3,H,W]
    #     return dL_mat, dC_mat: [B,N,N]
    #     """
    #     B, N, C, H, W = rgb_versions.shape
    #     x = rgb_versions.view(B * N, 3, H, W)
    #     yxy = self._rgb_to_yxy(x, eps=eps)  # [BN,3,H,W] (Y,x,y)

    #     Y = torch.log(yxy[:, 0:1] + eps)    # [BN,1,H,W]
    #     xy = yxy[:, 1:3]                    # [BN,2,H,W]

    #     Y = self._down(Y, self.dist_pool)
    #     xy = self._down(xy, self.dist_pool)

    #     h2, w2 = Y.shape[-2], Y.shape[-1]
    #     Y = Y.view(B, N, 1, h2, w2)         # [B,N,1,h,w]
    #     xy = xy.view(B, N, 2, h2, w2)       # [B,N,2,h,w]

    #     # broadcast diffs
    #     dY = (Y[:, :, None] - Y[:, None, :]).abs()      # [B,N,N,1,h,w]
    #     dxy = (xy[:, :, None] - xy[:, None, :]).abs()   # [B,N,N,2,h,w]

    #     # ✅ only average over (channel, h, w), NOT over N dims
    #     dL = dY.mean(dim=(3, 4, 5))        # [B,N,N]
    #     dC = dxy.mean(dim=(3, 4, 5))       # [B,N,N]
    #     return dL, dC
    
    @torch.no_grad()
    def _pairwise_dL_dC(self, rgb_versions, eps=1e-8):
        B, N, C, H, W = rgb_versions.shape
        x = rgb_versions.view(B*N, 3, H, W)
        yxy = self._rgb_to_yxy(x, eps=eps)

        Y  = torch.log(yxy[:, 0:1] + eps)
        xy = yxy[:, 1:3]

        Y  = self._down(Y,  self.dist_pool)
        xy = self._down(xy, self.dist_pool)
        h2, w2 = Y.shape[-2], Y.shape[-1]
        HW = h2*w2

        Yv  = Y.view(B, N, HW)
        xyv = xy.contiguous().view(B, N, 2*HW)

        dL = torch.cdist(Yv,  Yv,  p=1) / HW
        dC = torch.cdist(xyv, xyv, p=1) / (2*HW)
        return dL, dC


    @torch.no_grad()
    def _soft_weights(self, dL: torch.Tensor, dC: torch.Tensor, eps: float = 1e-8):
        """
        dL,dC: [B,N,N]
        return wL,wC: [B,N,N], diagonal zeroed
        """
        B, N, _ = dL.shape
        eye = torch.eye(N, device=dL.device, dtype=torch.bool).unsqueeze(0)  # [1,N,N]
        eyeB = eye.expand(B, -1, -1)                                         # [B,N,N]
        offB = ~eyeB                                                         # [B,N,N]

        # off-diagonal values per batch: [B, N*(N-1)]
        dL_off = dL.masked_select(offB).view(B, -1)
        dC_off = dC.masked_select(offB).view(B, -1)

        # robust stats
        sigma_L = dL_off.median(dim=1).values.clamp_min(eps).view(B, 1, 1)
        sigma_C = dC_off.median(dim=1).values.clamp_min(eps).view(B, 1, 1)

        medL = dL_off.median(dim=1).values.clamp_min(eps).view(B, 1, 1)
        medC = dC_off.median(dim=1).values.clamp_min(eps).view(B, 1, 1)

        dC_norm = dC / medC
        dL_norm = dL / medL

        wL = torch.exp(-dL / sigma_L) * dC_norm  # luma: luma-close + chroma-different
        wC = torch.exp(-dC / sigma_C) * dL_norm  # chroma: chroma-close + luma-different

        # zero diagonal
        wL = wL.masked_fill(eyeB, 0.0)
        wC = wC.masked_fill(eyeB, 0.0)
        return wL, wC

    def _encode_proj_global(self, feat_extractor: nn.Module, view_versions: torch.Tensor):
        """
        view_versions: [B,N,3,H,W] (pure-view inputs)
        return embeds: [B,N,D] projected + normalized (A方案)
        """
        B, N, C, H, W = view_versions.shape
        x = view_versions.view(B * N, C, H, W)
        z0, _ = feat_extractor.unet_q(x)                       # z0: [BN,Cg,1,1]
        z = z0.squeeze(-1).squeeze(-1)                        # [BN,Cg]
        z = feat_extractor.global_proj_q(z)                   # [BN,Cg]
        z = F.normalize(z, dim=1)
        return z.view(B, N, -1)


    # -------------------------
    # original helpers
    # -------------------------
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

        self.scaler = GradScaler() if self.use_bf16 else None

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

        # define losses
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

            if mid_sup_opt.get('perceptual_opt'):
                self.cri_perceptual_mid = build_loss(mid_sup_opt['perceptual_opt']).to(self.device)
            else:
                self.cri_perceptual_mid = None

            if mid_sup_opt.get('color_opt'):
                self.cri_color_mid = build_loss(mid_sup_opt['color_opt']).to(self.device)
            else:
                self.cri_color_mid = None

        if train_opt.get('contra_opt'):
            self.criterion_contra = torch.nn.CrossEntropyLoss().to(self.device)
            self.cri_contra_weight = train_opt['contra_opt']['loss_weight']
            self.cri_contra_weight_global = train_opt['contra_opt']['loss_weight_global']
            self.cri_contra_weight_local = train_opt['contra_opt']['loss_weight_local']

            # ---- invariance (soft weighted) ----
            self.use_invariance = train_opt['contra_opt'].get('use_invariance', True)
            self.inv_weight = train_opt['contra_opt'].get('inv_weight', 0.5)
            self.dist_pool = train_opt['contra_opt'].get('dist_pool', 32)  # downsample size for distance computing
            self.inv_every = train_opt['contra_opt'].get('inv_every', 2)  # ✅ default 4
            self.inv_max_versions = train_opt['contra_opt'].get('inv_max_versions', 4)  # q,k + up to (M-2) negs


        else:
            self.criterion_contra = None

        self.setup_optimizers()
        self.setup_schedulers()

    # -------------------------
    # feed_data (PURE VIEW, Yxy)
    # -------------------------
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)  # [B, 3, H, W]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        # original rgb views (kept if you still need them elsewhere)
        self.im_q = data['im_q'].to(self.device)
        self.im_k = data['im_k'].to(self.device)

        # ---- pure-view inputs for MoCo encoders ----
        self.im_q_I  = self._to_luma3_yxy(self.im_q)
        self.im_k_I  = self._to_luma3_yxy(self.im_k)

        self.im_q_TP = self._to_chroma3_yxy(self.im_q)
        self.im_k_TP = self._to_chroma3_yxy(self.im_k)

        K = self.inv_max_versions - 2
        # ---- negatives (NO pixel swap; just convert raw RGB neg -> view) ----
        self.im_negs_I = None
        self.negs_rgb_I_cpu = None  # NEW: [B,N,3,H,W] raw RGB for invariance distances

        if data.get('im_negs_I_raw', None) is not None and data['im_negs_I_raw'] is not None:
            view_list = []
            for neg_i_cpu in data['im_negs_I_raw']:
                neg_i = neg_i_cpu.to(self.device)            # expected [B,3,H,W]
                view_list.append(self._to_luma3_yxy(neg_i))  # [B,3,H,W] pure-view
            if len(view_list) > 0:
                self.im_negs_I = torch.stack(view_list, dim=1)      # [B,N,3,H,W]
                self.im_negs_I = apply_random_flip(self.im_negs_I)
            self.negs_rgb_I_cpu = data['im_negs_I_raw'][:K] if K > 0 else None

        self.im_negs_TP = None
        self.negs_rgb_TP_cpu = None  # NEW: [B,N,3,H,W] raw RGB for invariance distances

        if data.get('im_negs_TP_raw', None) is not None and data['im_negs_TP_raw'] is not None:
            view_list = []
            for neg_tp_cpu in data['im_negs_TP_raw']:
                neg_tp = neg_tp_cpu.to(self.device)               # [B,3,H,W]
                view_list.append(self._to_chroma3_yxy(neg_tp))     # [B,3,H,W] pure-view
            if len(view_list) > 0:
                self.im_negs_TP = torch.stack(view_list, dim=1)    # [B,N,3,H,W]
                self.im_negs_TP = apply_random_flip(self.im_negs_TP)
            self.negs_rgb_TP_cpu = data['im_negs_TP_raw'][:K] if K > 0 else None
        
        self.im_negs_I_rank = None
        if data.get('im_negs_I_rank', None) is not None:
            # dataloader will stack to [B,N] if each sample is Tensor[N]
            self.im_negs_I_rank = data['im_negs_I_rank'].to(self.device)

        self.im_negs_TP_rank = None
        if data.get('im_negs_TP_rank', None) is not None:
            self.im_negs_TP_rank = data['im_negs_TP_rank'].to(self.device)


    def feed_data_test(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        self.im_q = data['im_q'].to(self.device)

        # pure-view for test-time key encoder forward
        self.im_q_I  = self._to_luma3_yxy(self.im_q)
        self.im_q_TP = self._to_chroma3_yxy(self.im_q)

    # -------------------------
    # optimizers
    # -------------------------
    def setup_optimizers(self):
        train_opt = self.opt['train']

        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} in net_g will not be optimized.')

        optim_params_feat_I = []
        for k, v in self.feat_extractor_I.named_parameters():
            if v.requires_grad:
                optim_params_feat_I.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} in feat_extractor will not be optimized.')

        optim_params_feat_TP = []
        for k, v in self.feat_extractor_TP.named_parameters():
            if v.requires_grad:
                optim_params_feat_TP.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} in feat_extractor will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt['optim_g'])
        self.optimizer_feat_I = self.get_optimizer(optim_type, optim_params_feat_I, **train_opt['optim_g'])
        self.optimizer_feat_TP = self.get_optimizer(optim_type, optim_params_feat_TP, **train_opt['optim_g'])

        self.optimizers.append(self.optimizer_g)
        self.optimizers.append(self.optimizer_feat_I)
        self.optimizers.append(self.optimizer_feat_TP)

    # -------------------------
    # optimize_parameters (PURE VIEW used by feat_extractor_*)
    # -------------------------
    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']

        self.optimizer_g.zero_grad()
        self.optimizer_feat_I.zero_grad()
        self.optimizer_feat_TP.zero_grad()

        l_total = 0
        loss_dict = OrderedDict()

        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            if current_iter <= train_opt['stage1_iters']:
                # stage1: pure-view inputs
                z_I, y_I   = self.feat_extractor_I.unet_q(self.im_q_I)
                z_TP, y_TP = self.feat_extractor_TP.unet_q(self.im_q_TP)

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

            else:
                if not self.stage2_key_inited:
                    mode = train_opt.get('stage2_init_k', 'keep')
                    self.feat_extractor_I.stage2_init_key_encoder(mode)
                    self.feat_extractor_TP.stage2_init_key_encoder(mode)
                    logger = get_root_logger()
                    logger.info(f"[INFO] Stage2 key init: mode={mode}, iter={current_iter}")
                    self.stage2_key_inited = True

                if train_opt.get('enable_zero_switch', True) and (not self.zero_switched):
                    if hasattr(self.net_g, "switch_to_zero"):
                        self.net_g.switch_to_zero(use_bf16=self.use_bf16, device=self.device)
                        logger = get_root_logger()
                        logger.info(
                            f"[INFO] switch_to_zero() called at iteration {current_iter} "
                            f"with BF16={self.use_bf16}, device={self.device}"
                        )
                    self.zero_switched = True

                # stage2: contrastive with pure-view q/k/negs
                logits_global_I, labels_global_I, logits_local_I, labels_local_I, z_I, y_I = \
                    self.feat_extractor_I(self.im_q_I, self.im_k_I, self.im_negs_I, neg_ranks=self.im_negs_I_rank)

                logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP, z_TP, y_TP = \
                    self.feat_extractor_TP(self.im_q_TP, self.im_k_TP, self.im_negs_TP, neg_ranks=self.im_negs_TP_rank)

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
                # ================= Soft-weight invariance (global embedding A) =================
                if do_inv and (self.negs_rgb_I_cpu is not None) and (self.im_negs_I is not None):
                    # ---- build versions set for I branch: q,k,negs ----
                    # rgb_I = torch.cat([self.im_q.unsqueeze(1), self.im_k.unsqueeze(1), self.negs_rgb_I], dim=1)  # [B, 2+N, 3, H, W]
                    # view_I = torch.cat([self.im_q_I.unsqueeze(1), self.im_k_I.unsqueeze(1), self.im_negs_I], dim=1)
                    K = max(self.inv_max_versions - 2, 0)

                    # 截断 negs（若 negs 少于 K，就取全部）
                    raw_gpu_I = [t.to(self.device, non_blocking=True) for t in self.negs_rgb_I_cpu[:K]]
                    negs_rgb_I = torch.stack(raw_gpu_I, dim=1)  # [B,K,3,H,W]
                    negs_view_I = self.im_negs_I[:, :K] if (self.im_negs_I is not None and K > 0) else None

                    rgb_I = torch.cat([self.im_q.unsqueeze(1), self.im_k.unsqueeze(1), negs_rgb_I], dim=1)   # [B,2+K_eff,3,H,W]
                    view_I = torch.cat([self.im_q_I.unsqueeze(1), self.im_k_I.unsqueeze(1), negs_view_I], dim=1)
                    
                    if view_I.size(1) >= 2:
                        with torch.amp.autocast('cuda', enabled=False):
                            dL, dC = self._pairwise_dL_dC(rgb_I)
                            wL, _  = self._soft_weights(dL, dC)

                        emb = self._encode_proj_global(self.feat_extractor_I, view_I)  # [B,M,D] normalized
                        sim  = torch.einsum('bmd,bnd->bmn', emb, emb)                   # [B,M,M]
                        diff = 1.0 - sim

                        l_inv_I = (wL * diff).sum() / (wL.sum() + 1e-8)
                        l_total += self.inv_weight * l_inv_I
                        loss_dict['l_inv_I'] = l_inv_I
                    del raw_gpu_I, negs_rgb_I, negs_view_I, rgb_I, view_I
                    
                if do_inv and (self.negs_rgb_TP_cpu is not None) and (self.im_negs_TP is not None):
                    # ---- build versions set for TP branch: q,k,negs ----
                    # rgb_TP = torch.cat([self.im_q.unsqueeze(1), self.im_k.unsqueeze(1), self.negs_rgb_TP], dim=1)
                    # view_TP = torch.cat([self.im_q_TP.unsqueeze(1), self.im_k_TP.unsqueeze(1), self.im_negs_TP], dim=1)
                    K = max(self.inv_max_versions - 2, 0)
                    raw_gpu_TP = [t.to(self.device, non_blocking=True) for t in self.negs_rgb_TP_cpu[:K]]
                    negs_rgb_TP = torch.stack(raw_gpu_TP, dim=1)  # [B,K,3,H,W]
                    negs_view_TP = self.im_negs_TP[:, :K] if (self.im_negs_TP is not None and K > 0) else None

                    rgb_TP = torch.cat([self.im_q.unsqueeze(1), self.im_k.unsqueeze(1), negs_rgb_TP], dim=1)
                    view_TP = torch.cat([self.im_q_TP.unsqueeze(1), self.im_k_TP.unsqueeze(1), negs_view_TP], dim=1)
                    if view_TP.size(1) >= 2:
                        with torch.amp.autocast('cuda', enabled=False):
                            dL, dC = self._pairwise_dL_dC(rgb_TP)
                            _, wC  = self._soft_weights(dL, dC)

                        emb = self._encode_proj_global(self.feat_extractor_TP, view_TP)
                        sim  = torch.einsum('bmd,bnd->bmn', emb, emb)
                        diff = 1.0 - sim

                        l_inv_TP = (wC * diff).sum() / (wC.sum() + 1e-8)
                        l_total += self.inv_weight * l_inv_TP
                        loss_dict['l_inv_TP'] = l_inv_TP
                    del raw_gpu_TP, negs_rgb_TP, negs_view_TP, rgb_TP, view_TP
                    if 'dL' in locals(): del dL
                    if 'dC' in locals(): del dC
                    if 'wL' in locals(): del wL
                    if 'emb' in locals(): del emb
                    if 'sim' in locals(): del sim
                    if 'diff' in locals(): del diff

                # generator forward (still uses z_I/z_TP and y_I/y_TP from pure-view encoders)
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
            self.scaler.step(self.optimizer_feat_I)
            self.scaler.step(self.optimizer_feat_TP)
            self.scaler.update()
        else:
            l_total.backward()
            self.optimizer_g.step()
            self.optimizer_feat_I.step()
            self.optimizer_feat_TP.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    # -------------------------
    # test (PURE VIEW)
    # -------------------------
    def test(self):
        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.feat_extractor_I.eval()
                self.feat_extractor_TP.eval()
                with torch.no_grad():
                    z_I, y_I   = self.feat_extractor_I.unet_k(self.im_q_I)
                    z_TP, y_TP = self.feat_extractor_TP.unet_k(self.im_q_TP)
                    self.output = self.net_g_ema(self.lq, z_I, z_TP, y_I, y_TP)
                    self.output_mid = self.net_g_ema.mid_result
            else:
                self.net_g.eval()
                self.feat_extractor_I.eval()
                self.feat_extractor_TP.eval()
                with torch.no_grad():
                    z_I, y_I   = self.feat_extractor_I.unet_k(self.im_q_I)
                    z_TP, y_TP = self.feat_extractor_TP.unet_k(self.im_q_TP)
                    self.output = self.net_g(self.lq, z_I, z_TP, y_I, y_TP)
                    self.output_mid = self.net_g.mid_result
                self.net_g.train()
    
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        to_rgb = self.opt['val'].get('to_rgb', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            # self.save_batch_images(val_data, idx)
            self.feed_data_test(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = visuals['result']
            sr_img_mid = visuals['mid_result']
            gt_img = visuals['gt'] if 'gt' in visuals else None

            sr_img = torch.clamp(sr_img, 0.0, 1.0)
            gt_img = torch.clamp(gt_img, 0.0, 1.0)
            
            if to_rgb:
                convert_to_rgb = HdrConvertBackToRGB(src=to_rgb, zero_one_normed=True)
                sr_img = convert_to_rgb(sr_img)
                sr_img_mid = convert_to_rgb(sr_img_mid)
                gt_img = convert_to_rgb(gt_img) if gt_img is not None else None

            sr_img = tensor2img([sr_img], out_type=np.uint16)
            sr_img_mid = tensor2img([sr_img_mid])
            metric_data['img'] = sr_img
            if gt_img is not None:
                gt_img = tensor2img([gt_img], out_type=np.uint16)
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                    save_img_mid_path = osp.join(self.opt['path']['visualization'], img_name,
                                                 f'{img_name}_{current_iter}_mid.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        save_img_mid_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["val"]["suffix"]}_mid.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                        save_img_mid_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["name"]}_mid.png')
                imwrite(sr_img, save_img_path)
                imwrite(sr_img_mid, save_img_mid_path)

            if with_metrics:
                # calculate metrics
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
                
                if metric not in self.average_metrics:
                    self.average_metrics[metric] = {'total': 0, 'count': 0}
                self.average_metrics[metric]['total'] += self.metric_results[metric]
                self.average_metrics[metric]['count'] += 1
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
    
    def print_final_metrics(self):
        # Calculate and print the mean metric across datasets
        print("\nFinal Average Metrics Across All Datasets:")
        for metric, vals in self.average_metrics.items():
            mean_metric = vals['total'] / vals['count']
            print(f"{metric}: {mean_metric:.4f}")
    def save(self, epoch, current_iter):
        nets_to_save = [self.net_g]
        param_keys = ['params']

        if hasattr(self, 'net_g_ema'):
            nets_to_save.append(self.net_g_ema)
            param_keys.append('params_ema')

        nets_to_save.append(self.feat_extractor_I)
        param_keys.append('feat_params_I')
        
        nets_to_save.append(self.feat_extractor_TP)
        param_keys.append('feat_params_TP')

        self.save_network(
            net=nets_to_save,
            net_label='net_g',
            current_iter=current_iter,
            param_key=param_keys
        )

        self.save_training_state(epoch, current_iter)
    
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['mid_result'] = self.output_mid.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict


@MODEL_REGISTRY.register()
class ContrastYXYSharedGlobalMidSupITMModel(ContrastYXYMidSupITMModel):
    """
    Pure-view dual-encoder model in Yxy with shared global encoder path and split local decoders.
    Global condition extraction shares parameters across I/TP branches, while local spatial
    conditions remain branch-specific.
    """

    def __init__(self, opt):
        optim_cfg_backup = None
        scheduler_cfg_backup = None
        if opt.get('is_train', False):
            train_opt = opt['train']
            optim_cfg_backup = dict(train_opt['optim_g'])
            scheduler_cfg_backup = dict(train_opt['scheduler'])

        super().__init__(opt)

        self.feat_extractor_shared = MoCoWithUNetSharedGlobal(
            0.999,
            1.0,
            3,
            64,
            16,
        )
        self.feat_extractor_shared = self.model_to_device(self.feat_extractor_shared)
        self.print_network(self.feat_extractor_shared)

        if hasattr(self, 'feat_extractor_I'):
            del self.feat_extractor_I
        if hasattr(self, 'feat_extractor_TP'):
            del self.feat_extractor_TP

        load_path_feat = self.opt['path'].get('pretrain_network_g', None)
        if load_path_feat is not None:
            param_key = self.opt['path'].get('param_key_feat_shared', 'feat_params_shared')
            self.load_network(self.feat_extractor_shared, load_path_feat, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.opt['train']['optim_g'] = optim_cfg_backup
            self.opt['train']['scheduler'] = scheduler_cfg_backup
            self.optimizers = []
            self.schedulers = []
            self.setup_optimizers()
            self.setup_schedulers()

    def _encode_proj_global_branch(self, feat_extractor: nn.Module, view_versions: torch.Tensor, branch: str):
        B, N, C, H, W = view_versions.shape
        x = view_versions.view(B * N, C, H, W)
        z0, _ = feat_extractor.unet_q(x, branch=branch)
        z = z0.squeeze(-1).squeeze(-1)
        z = feat_extractor._global_proj_q(branch)(z)
        z = F.normalize(z, dim=1)
        return z.view(B, N, -1)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        logger = get_root_logger()

        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger.warning(f'Params {k} in net_g will not be optimized.')

        optim_cfg = dict(train_opt['optim_g'])
        optim_type = optim_cfg.pop('type')

        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **optim_cfg)
        self.optimizers = [self.optimizer_g]

        if not hasattr(self, 'feat_extractor_shared'):
            optim_params_feat_I = []
            for k, v in self.feat_extractor_I.named_parameters():
                if v.requires_grad:
                    optim_params_feat_I.append(v)
                else:
                    logger.warning(f'Params {k} in feat_extractor_I will not be optimized.')

            optim_params_feat_TP = []
            for k, v in self.feat_extractor_TP.named_parameters():
                if v.requires_grad:
                    optim_params_feat_TP.append(v)
                else:
                    logger.warning(f'Params {k} in feat_extractor_TP will not be optimized.')

            self.optimizer_feat_I = self.get_optimizer(optim_type, optim_params_feat_I, **optim_cfg)
            self.optimizer_feat_TP = self.get_optimizer(optim_type, optim_params_feat_TP, **optim_cfg)
            self.optimizers.append(self.optimizer_feat_I)
            self.optimizers.append(self.optimizer_feat_TP)
            return

        optim_params_feat = []
        for k, v in self.feat_extractor_shared.named_parameters():
            if v.requires_grad:
                optim_params_feat.append(v)
            else:
                logger.warning(f'Params {k} in feat_extractor_shared will not be optimized.')

        self.optimizer_feat = self.get_optimizer(optim_type, optim_params_feat, **optim_cfg)
        self.optimizers.append(self.optimizer_feat)

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']

        self.optimizer_g.zero_grad()
        self.optimizer_feat.zero_grad()

        l_total = 0
        loss_dict = OrderedDict()

        with torch.amp.autocast('cuda', enabled=self.use_bf16, dtype=torch.bfloat16):
            if current_iter <= train_opt['stage1_iters']:
                z_I, y_I = self.feat_extractor_shared.unet_q(self.im_q_I, branch='I')
                z_TP, y_TP = self.feat_extractor_shared.unet_q(self.im_q_TP, branch='TP')

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
            else:
                if not self.stage2_key_inited:
                    mode = train_opt.get('stage2_init_k', 'keep')
                    self.feat_extractor_shared.stage2_init_key_encoder(mode)
                    logger = get_root_logger()
                    logger.info(f"[INFO] Stage2 key init(shared-global): mode={mode}, iter={current_iter}")
                    self.stage2_key_inited = True

                if train_opt.get('enable_zero_switch', True) and (not self.zero_switched):
                    if hasattr(self.net_g, 'switch_to_zero'):
                        self.net_g.switch_to_zero(use_bf16=self.use_bf16, device=self.device)
                    self.zero_switched = True

                logits_global_I, labels_global_I, logits_local_I, labels_local_I, z_I, y_I = \
                    self.feat_extractor_shared(self.im_q_I, self.im_k_I, self.im_negs_I, branch='I', neg_ranks=self.im_negs_I_rank)
                logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP, z_TP, y_TP = \
                    self.feat_extractor_shared(self.im_q_TP, self.im_k_TP, self.im_negs_TP, branch='TP', neg_ranks=self.im_negs_TP_rank)

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
                if do_inv and (self.negs_rgb_I_cpu is not None) and (self.im_negs_I is not None):
                    K = max(self.inv_max_versions - 2, 0)
                    raw_gpu_I = [t.to(self.device, non_blocking=True) for t in self.negs_rgb_I_cpu[:K]]
                    negs_rgb_I = torch.stack(raw_gpu_I, dim=1)
                    negs_view_I = self.im_negs_I[:, :K] if (self.im_negs_I is not None and K > 0) else None
                    rgb_I = torch.cat([self.im_q.unsqueeze(1), self.im_k.unsqueeze(1), negs_rgb_I], dim=1)
                    view_I = torch.cat([self.im_q_I.unsqueeze(1), self.im_k_I.unsqueeze(1), negs_view_I], dim=1)
                    if view_I.size(1) >= 2:
                        with torch.amp.autocast('cuda', enabled=False):
                            dL, dC = self._pairwise_dL_dC(rgb_I)
                            wL, _ = self._soft_weights(dL, dC)
                        emb = self._encode_proj_global_branch(self.feat_extractor_shared, view_I, branch='I')
                        sim = torch.einsum('bmd,bnd->bmn', emb, emb)
                        diff = 1.0 - sim
                        l_inv_I = (wL * diff).sum() / (wL.sum() + 1e-8)
                        l_total += self.inv_weight * l_inv_I
                        loss_dict['l_inv_I'] = l_inv_I

                if do_inv and (self.negs_rgb_TP_cpu is not None) and (self.im_negs_TP is not None):
                    K = max(self.inv_max_versions - 2, 0)
                    raw_gpu_TP = [t.to(self.device, non_blocking=True) for t in self.negs_rgb_TP_cpu[:K]]
                    negs_rgb_TP = torch.stack(raw_gpu_TP, dim=1)
                    negs_view_TP = self.im_negs_TP[:, :K] if (self.im_negs_TP is not None and K > 0) else None
                    rgb_TP = torch.cat([self.im_q.unsqueeze(1), self.im_k.unsqueeze(1), negs_rgb_TP], dim=1)
                    view_TP = torch.cat([self.im_q_TP.unsqueeze(1), self.im_k_TP.unsqueeze(1), negs_view_TP], dim=1)
                    if view_TP.size(1) >= 2:
                        with torch.amp.autocast('cuda', enabled=False):
                            dL, dC = self._pairwise_dL_dC(rgb_TP)
                            _, wC = self._soft_weights(dL, dC)
                        emb = self._encode_proj_global_branch(self.feat_extractor_shared, view_TP, branch='TP')
                        sim = torch.einsum('bmd,bnd->bmn', emb, emb)
                        diff = 1.0 - sim
                        l_inv_TP = (wC * diff).sum() / (wC.sum() + 1e-8)
                        l_total += self.inv_weight * l_inv_TP
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
        with torch.amp.autocast('cuda', enabled=self.use_bf16, dtype=torch.bfloat16):
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.feat_extractor_shared.eval()
                with torch.no_grad():
                    z_I, y_I = self.feat_extractor_shared.unet_k(self.im_q_I, branch='I')
                    z_TP, y_TP = self.feat_extractor_shared.unet_k(self.im_q_TP, branch='TP')
                    self.output = self.net_g_ema(self.lq, z_I, z_TP, y_I, y_TP)
                    self.output_mid = self.net_g_ema.mid_result
            else:
                self.net_g.eval()
                self.feat_extractor_shared.eval()
                with torch.no_grad():
                    z_I, y_I = self.feat_extractor_shared.unet_k(self.im_q_I, branch='I')
                    z_TP, y_TP = self.feat_extractor_shared.unet_k(self.im_q_TP, branch='TP')
                    self.output = self.net_g(self.lq, z_I, z_TP, y_I, y_TP)
                    self.output_mid = self.net_g.mid_result
                self.net_g.train()

    def save(self, epoch, current_iter):
        nets_to_save = [self.net_g]
        param_keys = ['params']
        if hasattr(self, 'net_g_ema'):
            nets_to_save.append(self.net_g_ema)
            param_keys.append('params_ema')
        nets_to_save.append(self.feat_extractor_shared)
        param_keys.append('feat_params_shared')
        self.save_network(net=nets_to_save, net_label='net_g', current_iter=current_iter, param_key=param_keys)
        self.save_training_state(epoch, current_iter)


@MODEL_REGISTRY.register()
class ContrastYXYWeakCoupledMidSupITMModel(ContrastYXYMidSupITMModel):
    """
    Weak-coupled dual-encoder model using CIE Yxy:
      - I branch (luma-dominant): input = [Y, eps * x, eps * y]
      - TP branch (chroma-dominant): input = [eps * Y, x, y]

    This is a relaxed variant of ContrastYXYMidSupITMModel:
    the target factor remains dominant, while limited complementary
    context is provided to each branch.
    """

    EPS = 0.1

    @classmethod
    def _to_luma3_yxy(cls, rgb: torch.Tensor) -> torch.Tensor:
        """
        RGB -> Yxy -> weak-coupled luma-dominant 3ch input:
        [Y, eps*x, eps*y]
        """
        yxy = cls._rgb_to_yxy(rgb)
        Y = yxy[:, 0:1, :, :]
        x = yxy[:, 1:2, :, :]
        y = yxy[:, 2:3, :, :]
        return torch.cat([Y, cls.EPS * x, cls.EPS * y], dim=1)

    @classmethod
    def _to_chroma3_yxy(cls, rgb: torch.Tensor) -> torch.Tensor:
        """
        RGB -> Yxy -> weak-coupled chroma-dominant 3ch input:
        [eps*Y, x, y]
        """
        yxy = cls._rgb_to_yxy(rgb)
        Y = yxy[:, 0:1, :, :]
        x = yxy[:, 1:2, :, :]
        y = yxy[:, 2:3, :, :]
        return torch.cat([cls.EPS * Y, x, y], dim=1)
