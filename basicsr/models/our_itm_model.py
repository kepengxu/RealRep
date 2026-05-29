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
from basicsr.archs.contrastive import MoCo, MoCoV2, MoCoWithUNet, HighlightContrastLoss
from torch.cuda.amp import autocast, GradScaler
import os, cv2
import torchvision.transforms as transforms
import torch.autograd.profiler as profiler
from basicsr.utils.ICTCP_convert import SDR_to_YCbCr, YCbCr_to_SDR
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import save_image
import torchvision.transforms.functional as tF
import random


def save_tensor_as_image(tensor, filename, normalize=True):
    """
    将单个张量保存为图像。

    参数:
        tensor (torch.Tensor): 形状为 [C, H, W] 的张量。
        filename (str): 图像保存路径。
        normalize (bool): 是否将张量归一化到 [0,1] 范围。
    """
    # 确保张量在CPU上
    tensor = tensor.detach().cpu()
    if normalize:
        tensor = torch.clamp(tensor, 0, 1)  # 归一化到 [0,1]
    # 转换为PIL图像
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor)
    img.save(filename)

def apply_random_flip(batch_tensor, horizontal_flip=True, vertical_flip=True):
    """
    对批量张量中的每一张图像随机应用水平和/或垂直翻转。

    Args:
        batch_tensor (torch.Tensor): 形状为 [B, 3, H, W] 的张量
        horizontal_flip (bool): 是否应用水平翻转
        vertical_flip (bool): 是否应用垂直翻转

    Returns:
        torch.Tensor: 应用变换后的张量，形状仍为 [B, 3, H, W]
    """
    transformed = []
    for img in batch_tensor:
        if horizontal_flip and random.random() > 0.5:
            img = tF.hflip(img)
        if vertical_flip and random.random() > 0.5:
            img = tF.vflip(img)
        transformed.append(img)
    return torch.stack(transformed)

@MODEL_REGISTRY.register()
class ITMModel_copy(SRModel):
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
class ContrastYCBCRMidSupITMModel_wozlum(ITMModel_copy):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # 定义网络
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        
        # self.feat_extractor_I = MoCoWithUNet(in_channels=3, global_cond_channels=64, spatial_cond_channels=16, temperature=1.0)
        # self.feat_extractor_I = self.model_to_device(self.feat_extractor_I)
        # self.print_network(self.feat_extractor_I)
        
        self.feat_extractor_TP = MoCoWithUNet(in_channels=3, global_cond_channels=64, spatial_cond_channels=16, temperature=1.0)
        self.feat_extractor_TP = self.model_to_device(self.feat_extractor_TP)
        self.print_network(self.feat_extractor_TP)
        
        # self.scaler = GradScaler()
        self.average_metrics = {}
        
        # 加载预训练模型
        load_path = self.opt['path'].get('pretrain_network_g', None)
        # load_path_feat_I = self.opt['path'].get('pretrain_network_g', None)
        load_path_feat_TP = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        # if load_path_feat_I is not None:
        #     param_key = self.opt['path'].get('param_key_feat_I', 'feat_params_I')
        #     self.load_network(self.feat_extractor_I, load_path_feat_I, self.opt['path'].get('strict_load_g', True), param_key)
        if load_path_feat_TP is not None:
            param_key = self.opt['path'].get('param_key_feat_TP', 'feat_params_TP')
            self.load_network(self.feat_extractor_TP, load_path_feat_TP, self.opt['path'].get('strict_load_g', True), param_key)
        
        self.use_bf16 = self.opt.get('use_bf16', False)
        # 如果是训练模式，则初始化训练设置
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        
        # (###### [BF16 CHANGE]) 检查是否使用 BF16
        self.use_bf16 = train_opt.get('use_bf16', False)  # 在配置文件里加一个 use_bf16 = true/false
        if self.use_bf16:
            logger = get_root_logger()
            logger.info('Using bfloat16 mixed precision training ...')  # 修正日志信息
        
        # (###### [BF16 CHANGE]) 初始化 GradScaler
        self.scaler = GradScaler() if self.use_bf16 else None  # BF16 通常需要 GradScaler
        
        # 定义EMA衰减
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
        # 原图
        self.lq = data['lq'].to(self.device)  # [B, 3, H, W]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        
        # im_q, im_k 也上GPU
        self.im_q = data['im_q'].to(self.device)
        self.im_k = data['im_k'].to(self.device)

        # 先对原图做一次 SDR->ICTCP (放到GPU计算)
        # 直接传递 [B, 3, H, W]，无需 unsqueeze 和 squeeze
        dim = 1  # 通道维度
        sdr_ictcp_base = SDR_to_YCbCr(self.lq, dim=dim)  # shape [B, 3, H, W], GPU

        # --------------- 亮度负样本 ---------------
        # self.im_negs_I = None
        # if data['im_negs_I_raw'] is not None:
        #     im_negs_I_list = []  # 存放替换后张量
        #     for i, neg_i_cpu in enumerate(data['im_negs_I_raw']):
        #         # 负样本上 GPU
        #         neg_i_gpu = neg_i_cpu.to(self.device)  # [B, 3, H, W] 或 [3, H, W]
                
        #         # 负样本转换 SDR->ICTCP
        #         neg_ictcp = SDR_to_YCbCr(neg_i_gpu, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                
        #         # 克隆
        #         sdr_ictcp_i = sdr_ictcp_base.clone()
                
        #         # 替换 I 通道 (index=0)
        #         sdr_ictcp_i[:, 0, :, :] = neg_ictcp[:, 0, :, :]
                
        #         # 回到 SDR
        #         replaced_sdr_i = YCbCr_to_SDR(sdr_ictcp_i, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
        #         im_negs_I_list.append(replaced_sdr_i)
            
        #     if len(im_negs_I_list) > 0:
        #         self.im_negs_I = torch.stack(im_negs_I_list, dim=1)  # [N, B, 3, H, W] 或 [N, 3, H, W]
        #         # 如果 N=1 且 B=1，可以简化维度
        #         if self.im_negs_I.size(1) == 1:
        #             self.im_negs_I = self.im_negs_I.squeeze(1)  # [N, 3, H, W] 或 [N, 3, H, W]
        #         self.im_negs_I = apply_random_flip(self.im_negs_I)
        
        # --------------- 色度负样本 ---------------
        self.im_negs_TP = None
        if data['im_negs_TP_raw'] is not None:
            im_negs_TP_list = []
            for i, neg_tp_cpu in enumerate(data['im_negs_TP_raw']):
                neg_tp_gpu = neg_tp_cpu.to(self.device)  # [B, 3, H, W] 或 [3, H, W]
                
                # 负样本转换 SDR->ICTCP
                neg_ictcp = SDR_to_YCbCr(neg_tp_gpu, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                
                # 克隆原图 ictcp
                sdr_ictcp_tp = sdr_ictcp_base.clone()
                
                # 替换 T,P 通道 (index=1,2)
                sdr_ictcp_tp[:, 1:3, :, :] = neg_ictcp[:, 1:3, :, :]
                
                # 回到 SDR
                replaced_sdr_tp = YCbCr_to_SDR(sdr_ictcp_tp, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                im_negs_TP_list.append(replaced_sdr_tp)
            
            if len(im_negs_TP_list) > 0:
                self.im_negs_TP = torch.stack(im_negs_TP_list, dim=1)  # [N, B, 3, H, W] 或 [N, 3, H, W]
                # 如果 N=1 且 B=1，可以简化维度
                if self.im_negs_TP.size(1) == 1:
                    self.im_negs_TP = self.im_negs_TP.squeeze(1)  # [N, 3, H, W] 或 [N, 3, H, W]
                self.im_negs_TP = apply_random_flip(self.im_negs_TP)

        # 这样 self.im_negs_I 和 self.im_negs_TP 就是 GPU 上处理好的负样本

    
    def feed_data_test(self, data):
        # 获取输入数据并放入设备
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        # 生成对比学习的正样本和负样本
        self.im_q = data['im_q'].to(self.device)
        
    
    def setup_optimizers(self):
        train_opt = self.opt['train']
        
        # 定义 net_g 参数
        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} in net_g will not be optimized.')

        # # 定义 feat_extractor 参数
        # optim_params_feat_I = []
        # for k, v in self.feat_extractor_I.named_parameters():
        #     if v.requires_grad:
        #         optim_params_feat_I.append(v)
        #     else:
        #         logger = get_root_logger()
        #         logger.warning(f'Params {k} in feat_extractor will not be optimized.')
                
        optim_params_feat_TP = []
        for k, v in self.feat_extractor_TP.named_parameters():
            if v.requires_grad:
                optim_params_feat_TP.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} in feat_extractor will not be optimized.')
        
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt['optim_g'])
        # self.optimizer_feat_I = self.get_optimizer(optim_type, optim_params_feat_I, **train_opt['optim_g'])
        self.optimizer_feat_TP = self.get_optimizer(optim_type, optim_params_feat_TP, **train_opt['optim_g'])
        
        # 保存优化器到优化器列表
        self.optimizers.append(self.optimizer_g)
        # self.optimizers.append(self.optimizer_feat_I)
        self.optimizers.append(self.optimizer_feat_TP)
    

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']
        # 清除梯度
        self.optimizer_g.zero_grad()
        # self.optimizer_feat_I.zero_grad()
        self.optimizer_feat_TP.zero_grad()
        
        # 初始化总损失
        l_total = 0
        loss_dict = OrderedDict()

        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            if current_iter <= train_opt['stage1_iters']:  # 阶段 1：仅更新主模型，不引入对比学习
                # 使用对比学习网络的 UNet 部分 (unet_q)
                # z_I, y_I = self.feat_extractor_I.unet_q(self.im_q)   # in:(B, 1, H, W)
                z_TP, y_TP = self.feat_extractor_TP.unet_q(self.im_q) # in:(B, 2, H, W)

                # 超分辨率模型前向传播
                self.output = self.net_g(self.lq, z_TP, y_TP)
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

            else:  # 阶段 2：引入对比学习
                # 对比学习网络前向传播
                # save_negatives(self.im_negs_I, self.im_negs_TP, batch_index=0)
                # logits_global_I, labels_global_I, logits_local_I, labels_local_I, z_I, y_I = self.feat_extractor_I(self.im_q, self.im_k, self.im_negs_I)
                
                logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP, z_TP, y_TP = self.feat_extractor_TP(self.im_q, self.im_k, self.im_negs_TP)

                # # 全局对比学习损失
                # global_contra_loss_I = self.criterion_contra(logits_global_I, labels_global_I)
                # loss_dict['l_global_contra_I'] = global_contra_loss_I
                # l_total += global_contra_loss_I * self.cri_contra_weight_global * self.cri_contra_weight
                
                global_contra_loss_TP = self.criterion_contra(logits_global_TP, labels_global_TP)
                loss_dict['l_global_contra_TP'] = global_contra_loss_TP
                l_total += global_contra_loss_TP * self.cri_contra_weight_global * self.cri_contra_weight

                # # 局部对比学习损失
                # local_contra_loss_I = self.criterion_contra(logits_local_I, labels_local_I)
                # loss_dict['l_local_contra_I'] = local_contra_loss_I
                # l_total += local_contra_loss_I * self.cri_contra_weight_local * self.cri_contra_weight
                
                local_contra_loss_TP = self.criterion_contra(logits_local_TP, labels_local_TP)
                loss_dict['l_local_contra_TP'] = local_contra_loss_TP
                l_total += local_contra_loss_TP * self.cri_contra_weight_local * self.cri_contra_weight


                # 超分辨率模型前向传播
                self.output = self.net_g(self.lq, z_TP, y_TP)
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
                    
                # # 可视化部分：按log_interval频率保存图像
                # log_interval = train_opt.get('log_interval', 100)  # 默认每100次迭代保存一次
                # if current_iter % log_interval == 0:
                #     self.save_visualizations(current_iter, save_dir='saved_images_stage2')

                
        # 反向传播和优化
        if self.scaler is not None:
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            # self.scaler.step(self.optimizer_feat_I)
            self.scaler.step(self.optimizer_feat_TP)
            self.scaler.update()
        else:
            # 否则直接普通 backward
            l_total.backward()
            self.optimizer_g.step()
            # self.optimizer_feat_I.step()
            self.optimizer_feat_TP.step()

        # # 计算度量指标（metrics）
        # if self.opt['val'].get('metrics') is not None:
        #     if not hasattr(self, 'metric_results'):  # only execute in the first run
        #         self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        #     # 初始化 metric_data 并进行指标计算
        #     metric_data = {'img': self.output.detach().cpu().numpy(), 'img2': self.gt.detach().cpu().numpy()}
        #     for name, opt_ in self.opt['val']['metrics'].items():
        #         metric_val = calculate_metric(metric_data, opt_)
        #         self.metric_results[name] = metric_val / self.opt['datasets']['train']['batch_size_per_gpu']
        #         # 打印或记录每个度量指标
        #         loss_dict[f'{name}'] = metric_val
        
        # # 创建保存路径
        # save_dir = self.opt['path'].get('visualization', './visualization')
        # os.makedirs(save_dir, exist_ok=True)

        # # 遍历 batch 并保存每张图片
        # for i in range(self.output.size(0)):
        #     output_img = tensor2img([self.output[i]])  # 将 Tensor 转换为图像
        #     gt_img = tensor2img([self.gt[i]])  # 目标图像

        #     # 保存图像路径
        #     save_output_path = os.path.join(save_dir, f'output_{current_iter}_img_{i}.png')
        #     save_gt_path = os.path.join(save_dir, f'gt_{current_iter}_img_{i}.png')
            
        #     cv2.imwrite(save_output_path, output_img)
        #     cv2.imwrite(save_gt_path, gt_img)
                
        # # 反向传播和优化
        # l_total.backward()
        # self.optimizer_g.step()
        # self.optimizer_feat.step()
        # 使用 scaler 缩放损失，进行反向传播和优化
            # self.scaler.scale(l_total).backward()
            # self.scaler.step(self.optimizer_g)
            # self.scaler.step(self.optimizer_feat)
            # self.scaler.update()

        # 保存日志
        self.log_dict = self.reduce_loss_dict(loss_dict)

        # 如果启用了EMA，则更新EMA模型权重
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
            # self.contrast_model_ema(decay=self.ema_decay)

    def test(self):
        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):  # (###### [FP16 CHANGE])
            # 使用EMA模型进行测试
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                # self.feat_extractor_I.eval()
                self.feat_extractor_TP.eval()
                with torch.no_grad():
                    # 获取特征编码
                    # z_I, y_I = self.feat_extractor_I.unet_k(self.im_q)
                    z_TP, y_TP = self.feat_extractor_TP.unet_k(self.im_q)
                    # 使用特征编码进行推理
                    self.output = self.net_g_ema(self.lq, z_TP, y_TP)
                    self.output_mid = self.net_g_ema.mid_result
            else:
                # 使用原始模型和feat_extractor进行测试
                self.net_g.eval()
                # self.feat_extractor_I.eval()
                self.feat_extractor_TP.eval()
                with torch.no_grad():
                    # 获取特征编码
                    # z_I, y_I = self.feat_extractor_I.unet_k(self.im_q)
                    z_TP, y_TP = self.feat_extractor_TP.unet_k(self.im_q)
                    # 使用特征编码进行推理
                    self.output = self.net_g(self.lq, z_TP, y_TP)
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
        # 构建要保存的网络和对应的键名列表
        nets_to_save = [self.net_g]
        param_keys = ['params']

        if hasattr(self, 'net_g_ema'):
            nets_to_save.append(self.net_g_ema)
            param_keys.append('params_ema')

        # nets_to_save.append(self.feat_extractor_I)
        # param_keys.append('feat_params_I')
        
        nets_to_save.append(self.feat_extractor_TP)
        param_keys.append('feat_params_TP')

        # if hasattr(self, 'feat_extractor_ema'):
        #     nets_to_save.append(self.feat_extractor_ema)
        #     param_keys.append('feat_params_ema')

        # 调用 save_network 并确保 nets_to_save 和 param_keys 长度匹配
        self.save_network(
            net=nets_to_save,
            net_label='net_g',
            current_iter=current_iter,
            param_key=param_keys
        )

        # 保存训练状态
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
class ContrastYCBCRMidSupITMModel_wozchr(ITMModel_copy):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # 定义网络
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        
        self.feat_extractor_I = MoCoWithUNet(in_channels=3, global_cond_channels=64, spatial_cond_channels=16, temperature=1.0)
        self.feat_extractor_I = self.model_to_device(self.feat_extractor_I)
        self.print_network(self.feat_extractor_I)
        
        # self.feat_extractor_TP = MoCoWithUNet(in_channels=3, global_cond_channels=64, spatial_cond_channels=16, temperature=1.0)
        # self.feat_extractor_TP = self.model_to_device(self.feat_extractor_TP)
        # self.print_network(self.feat_extractor_TP)
        
        # self.scaler = GradScaler()
        self.average_metrics = {}
        
        # 加载预训练模型
        load_path = self.opt['path'].get('pretrain_network_g', None)
        load_path_feat_I = self.opt['path'].get('pretrain_network_g', None)
        # load_path_feat_TP = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        if load_path_feat_I is not None:
            param_key = self.opt['path'].get('param_key_feat_I', 'feat_params_I')
            self.load_network(self.feat_extractor_I, load_path_feat_I, self.opt['path'].get('strict_load_g', True), param_key)
        # if load_path_feat_TP is not None:
        #     param_key = self.opt['path'].get('param_key_feat_TP', 'feat_params_TP')
        #     self.load_network(self.feat_extractor_TP, load_path_feat_TP, self.opt['path'].get('strict_load_g', True), param_key)
        
        self.use_bf16 = self.opt.get('use_bf16', False)
        # 如果是训练模式，则初始化训练设置
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        
        # (###### [BF16 CHANGE]) 检查是否使用 BF16
        self.use_bf16 = train_opt.get('use_bf16', False)  # 在配置文件里加一个 use_bf16 = true/false
        if self.use_bf16:
            logger = get_root_logger()
            logger.info('Using bfloat16 mixed precision training ...')  # 修正日志信息
        
        # (###### [BF16 CHANGE]) 初始化 GradScaler
        self.scaler = GradScaler() if self.use_bf16 else None  # BF16 通常需要 GradScaler
        
        # 定义EMA衰减
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
        # 原图
        self.lq = data['lq'].to(self.device)  # [B, 3, H, W]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        
        # im_q, im_k 也上GPU
        self.im_q = data['im_q'].to(self.device)
        self.im_k = data['im_k'].to(self.device)

        # 先对原图做一次 SDR->ICTCP (放到GPU计算)
        # 直接传递 [B, 3, H, W]，无需 unsqueeze 和 squeeze
        dim = 1  # 通道维度
        sdr_ictcp_base = SDR_to_YCbCr(self.lq, dim=dim)  # shape [B, 3, H, W], GPU

        # --------------- 亮度负样本 ---------------
        self.im_negs_I = None
        if data['im_negs_I_raw'] is not None:
            im_negs_I_list = []  # 存放替换后张量
            for i, neg_i_cpu in enumerate(data['im_negs_I_raw']):
                # 负样本上 GPU
                neg_i_gpu = neg_i_cpu.to(self.device)  # [B, 3, H, W] 或 [3, H, W]
                
                # 负样本转换 SDR->ICTCP
                neg_ictcp = SDR_to_YCbCr(neg_i_gpu, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                
                # 克隆
                sdr_ictcp_i = sdr_ictcp_base.clone()
                
                # 替换 I 通道 (index=0)
                sdr_ictcp_i[:, 0, :, :] = neg_ictcp[:, 0, :, :]
                
                # 回到 SDR
                replaced_sdr_i = YCbCr_to_SDR(sdr_ictcp_i, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                im_negs_I_list.append(replaced_sdr_i)
            
            if len(im_negs_I_list) > 0:
                self.im_negs_I = torch.stack(im_negs_I_list, dim=1)  # [N, B, 3, H, W] 或 [N, 3, H, W]
                # 如果 N=1 且 B=1，可以简化维度
                if self.im_negs_I.size(1) == 1:
                    self.im_negs_I = self.im_negs_I.squeeze(1)  # [N, 3, H, W] 或 [N, 3, H, W]
                self.im_negs_I = apply_random_flip(self.im_negs_I)
        
        # --------------- 色度负样本 ---------------
        # self.im_negs_TP = None
        # if data['im_negs_TP_raw'] is not None:
        #     im_negs_TP_list = []
        #     for i, neg_tp_cpu in enumerate(data['im_negs_TP_raw']):
        #         neg_tp_gpu = neg_tp_cpu.to(self.device)  # [B, 3, H, W] 或 [3, H, W]
                
        #         # 负样本转换 SDR->ICTCP
        #         neg_ictcp = SDR_to_YCbCr(neg_tp_gpu, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                
        #         # 克隆原图 ictcp
        #         sdr_ictcp_tp = sdr_ictcp_base.clone()
                
        #         # 替换 T,P 通道 (index=1,2)
        #         sdr_ictcp_tp[:, 1:3, :, :] = neg_ictcp[:, 1:3, :, :]
                
        #         # 回到 SDR
        #         replaced_sdr_tp = YCbCr_to_SDR(sdr_ictcp_tp, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
        #         im_negs_TP_list.append(replaced_sdr_tp)
            
        #     if len(im_negs_TP_list) > 0:
        #         self.im_negs_TP = torch.stack(im_negs_TP_list, dim=1)  # [N, B, 3, H, W] 或 [N, 3, H, W]
        #         # 如果 N=1 且 B=1，可以简化维度
        #         if self.im_negs_TP.size(1) == 1:
        #             self.im_negs_TP = self.im_negs_TP.squeeze(1)  # [N, 3, H, W] 或 [N, 3, H, W]
        #         self.im_negs_TP = apply_random_flip(self.im_negs_TP)

        # 这样 self.im_negs_I 和 self.im_negs_TP 就是 GPU 上处理好的负样本

    
    def feed_data_test(self, data):
        # 获取输入数据并放入设备
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        # 生成对比学习的正样本和负样本
        self.im_q = data['im_q'].to(self.device)
        
    
    def setup_optimizers(self):
        train_opt = self.opt['train']
        
        # 定义 net_g 参数
        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} in net_g will not be optimized.')

        # 定义 feat_extractor 参数
        optim_params_feat_I = []
        for k, v in self.feat_extractor_I.named_parameters():
            if v.requires_grad:
                optim_params_feat_I.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} in feat_extractor will not be optimized.')
                
        # optim_params_feat_TP = []
        # for k, v in self.feat_extractor_TP.named_parameters():
        #     if v.requires_grad:
        #         optim_params_feat_TP.append(v)
        #     else:
        #         logger = get_root_logger()
        #         logger.warning(f'Params {k} in feat_extractor will not be optimized.')
        
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt['optim_g'])
        self.optimizer_feat_I = self.get_optimizer(optim_type, optim_params_feat_I, **train_opt['optim_g'])
        # self.optimizer_feat_TP = self.get_optimizer(optim_type, optim_params_feat_TP, **train_opt['optim_g'])
        
        # 保存优化器到优化器列表
        self.optimizers.append(self.optimizer_g)
        self.optimizers.append(self.optimizer_feat_I)
        # self.optimizers.append(self.optimizer_feat_TP)
    

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']
        # 清除梯度
        self.optimizer_g.zero_grad()
        self.optimizer_feat_I.zero_grad()
        # self.optimizer_feat_TP.zero_grad()
        
        # 初始化总损失
        l_total = 0
        loss_dict = OrderedDict()

        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            if current_iter <= train_opt['stage1_iters']:  # 阶段 1：仅更新主模型，不引入对比学习
                # 使用对比学习网络的 UNet 部分 (unet_q)
                z_I, y_I = self.feat_extractor_I.unet_q(self.im_q)   # in:(B, 1, H, W)
                # z_TP, y_TP = self.feat_extractor_TP.unet_q(self.im_q) # in:(B, 2, H, W)

                # 超分辨率模型前向传播
                self.output = self.net_g(self.lq, z_I, y_I)
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

            else:  # 阶段 2：引入对比学习
                # 对比学习网络前向传播
                # save_negatives(self.im_negs_I, self.im_negs_TP, batch_index=0)
                logits_global_I, labels_global_I, logits_local_I, labels_local_I, z_I, y_I = self.feat_extractor_I(self.im_q, self.im_k, self.im_negs_I)
                
                # logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP, z_TP, y_TP = self.feat_extractor_TP(self.im_q, self.im_k, self.im_negs_TP)

                # 全局对比学习损失
                global_contra_loss_I = self.criterion_contra(logits_global_I, labels_global_I)
                loss_dict['l_global_contra_I'] = global_contra_loss_I
                l_total += global_contra_loss_I * self.cri_contra_weight_global * self.cri_contra_weight
                
                # global_contra_loss_TP = self.criterion_contra(logits_global_TP, labels_global_TP)
                # loss_dict['l_global_contra_TP'] = global_contra_loss_TP
                # l_total += global_contra_loss_TP * self.cri_contra_weight_global * self.cri_contra_weight

                # 局部对比学习损失
                local_contra_loss_I = self.criterion_contra(logits_local_I, labels_local_I)
                loss_dict['l_local_contra_I'] = local_contra_loss_I
                l_total += local_contra_loss_I * self.cri_contra_weight_local * self.cri_contra_weight
                
                # local_contra_loss_TP = self.criterion_contra(logits_local_TP, labels_local_TP)
                # loss_dict['l_local_contra_TP'] = local_contra_loss_TP
                # l_total += local_contra_loss_TP * self.cri_contra_weight_local * self.cri_contra_weight


                # 超分辨率模型前向传播
                self.output = self.net_g(self.lq, z_I, y_I)
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
                    
                # # 可视化部分：按log_interval频率保存图像
                # log_interval = train_opt.get('log_interval', 100)  # 默认每100次迭代保存一次
                # if current_iter % log_interval == 0:
                #     self.save_visualizations(current_iter, save_dir='saved_images_stage2')

                
        # 反向传播和优化
        if self.scaler is not None:
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.step(self.optimizer_feat_I)
            # self.scaler.step(self.optimizer_feat_TP)
            self.scaler.update()
        else:
            # 否则直接普通 backward
            l_total.backward()
            self.optimizer_g.step()
            self.optimizer_feat_I.step()
            # self.optimizer_feat_TP.step()


        # 保存日志
        self.log_dict = self.reduce_loss_dict(loss_dict)

        # 如果启用了EMA，则更新EMA模型权重
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
            # self.contrast_model_ema(decay=self.ema_decay)

    def test(self):
        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):  # (###### [FP16 CHANGE])
            # 使用EMA模型进行测试
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.feat_extractor_I.eval()
                # self.feat_extractor_TP.eval()
                with torch.no_grad():
                    # 获取特征编码
                    z_I, y_I = self.feat_extractor_I.unet_k(self.im_q)
                    # z_TP, y_TP = self.feat_extractor_TP.unet_k(self.im_q)
                    # 使用特征编码进行推理
                    self.output = self.net_g_ema(self.lq, z_I, y_I)
                    self.output_mid = self.net_g_ema.mid_result
            else:
                # 使用原始模型和feat_extractor进行测试
                self.net_g.eval()
                self.feat_extractor_I.eval()
                # self.feat_extractor_TP.eval()
                with torch.no_grad():
                    # 获取特征编码
                    z_I, y_I = self.feat_extractor_I.unet_k(self.im_q)
                    # z_TP, y_TP = self.feat_extractor_TP.unet_k(self.im_q)
                    # 使用特征编码进行推理
                    self.output = self.net_g(self.lq, z_I, y_I)
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
        # 构建要保存的网络和对应的键名列表
        nets_to_save = [self.net_g]
        param_keys = ['params']

        if hasattr(self, 'net_g_ema'):
            nets_to_save.append(self.net_g_ema)
            param_keys.append('params_ema')

        nets_to_save.append(self.feat_extractor_I)
        param_keys.append('feat_params_I')
        
        # nets_to_save.append(self.feat_extractor_TP)
        # param_keys.append('feat_params_TP')

        # if hasattr(self, 'feat_extractor_ema'):
        #     nets_to_save.append(self.feat_extractor_ema)
        #     param_keys.append('feat_params_ema')

        # 调用 save_network 并确保 nets_to_save 和 param_keys 长度匹配
        self.save_network(
            net=nets_to_save,
            net_label='net_g',
            current_iter=current_iter,
            param_key=param_keys
        )

        # 保存训练状态
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
class ContrastYCBCRMidSupITMModel_wozlumchr(ITMModel_copy):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # 定义网络
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        
        # self.feat_extractor_I = MoCoWithUNet(in_channels=3, global_cond_channels=64, spatial_cond_channels=16, temperature=1.0)
        # self.feat_extractor_I = self.model_to_device(self.feat_extractor_I)
        # self.print_network(self.feat_extractor_I)
        
        # self.feat_extractor_TP = MoCoWithUNet(in_channels=3, global_cond_channels=64, spatial_cond_channels=16, temperature=1.0)
        # self.feat_extractor_TP = self.model_to_device(self.feat_extractor_TP)
        # self.print_network(self.feat_extractor_TP)
        
        # self.scaler = GradScaler()
        self.average_metrics = {}
        
        # 加载预训练模型
        load_path = self.opt['path'].get('pretrain_network_g', None)
        load_path_feat_I = self.opt['path'].get('pretrain_network_g', None)
        # load_path_feat_TP = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        # if load_path_feat_I is not None:
        #     param_key = self.opt['path'].get('param_key_feat_I', 'feat_params_I')
        #     self.load_network(self.feat_extractor_I, load_path_feat_I, self.opt['path'].get('strict_load_g', True), param_key)
        # if load_path_feat_TP is not None:
        #     param_key = self.opt['path'].get('param_key_feat_TP', 'feat_params_TP')
        #     self.load_network(self.feat_extractor_TP, load_path_feat_TP, self.opt['path'].get('strict_load_g', True), param_key)
        
        self.use_bf16 = self.opt.get('use_bf16', False)
        # 如果是训练模式，则初始化训练设置
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        
        # (###### [BF16 CHANGE]) 检查是否使用 BF16
        self.use_bf16 = train_opt.get('use_bf16', False)  # 在配置文件里加一个 use_bf16 = true/false
        if self.use_bf16:
            logger = get_root_logger()
            logger.info('Using bfloat16 mixed precision training ...')  # 修正日志信息
        
        # (###### [BF16 CHANGE]) 初始化 GradScaler
        self.scaler = GradScaler() if self.use_bf16 else None  # BF16 通常需要 GradScaler
        
        # 定义EMA衰减
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
        # 原图
        self.lq = data['lq'].to(self.device)  # [B, 3, H, W]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        
        # # im_q, im_k 也上GPU
        # self.im_q = data['im_q'].to(self.device)
        # self.im_k = data['im_k'].to(self.device)

        # # 先对原图做一次 SDR->ICTCP (放到GPU计算)
        # # 直接传递 [B, 3, H, W]，无需 unsqueeze 和 squeeze
        # dim = 1  # 通道维度
        # sdr_ictcp_base = SDR_to_YCbCr(self.lq, dim=dim)  # shape [B, 3, H, W], GPU

        # # --------------- 亮度负样本 ---------------
        # self.im_negs_I = None
        # if data['im_negs_I_raw'] is not None:
        #     im_negs_I_list = []  # 存放替换后张量
        #     for i, neg_i_cpu in enumerate(data['im_negs_I_raw']):
        #         # 负样本上 GPU
        #         neg_i_gpu = neg_i_cpu.to(self.device)  # [B, 3, H, W] 或 [3, H, W]
                
        #         # 负样本转换 SDR->ICTCP
        #         neg_ictcp = SDR_to_YCbCr(neg_i_gpu, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                
        #         # 克隆
        #         sdr_ictcp_i = sdr_ictcp_base.clone()
                
        #         # 替换 I 通道 (index=0)
        #         sdr_ictcp_i[:, 0, :, :] = neg_ictcp[:, 0, :, :]
                
        #         # 回到 SDR
        #         replaced_sdr_i = YCbCr_to_SDR(sdr_ictcp_i, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
        #         im_negs_I_list.append(replaced_sdr_i)
            
        #     if len(im_negs_I_list) > 0:
        #         self.im_negs_I = torch.stack(im_negs_I_list, dim=1)  # [N, B, 3, H, W] 或 [N, 3, H, W]
        #         # 如果 N=1 且 B=1，可以简化维度
        #         if self.im_negs_I.size(1) == 1:
        #             self.im_negs_I = self.im_negs_I.squeeze(1)  # [N, 3, H, W] 或 [N, 3, H, W]
        #         self.im_negs_I = apply_random_flip(self.im_negs_I)
        
        # --------------- 色度负样本 ---------------
        # self.im_negs_TP = None
        # if data['im_negs_TP_raw'] is not None:
        #     im_negs_TP_list = []
        #     for i, neg_tp_cpu in enumerate(data['im_negs_TP_raw']):
        #         neg_tp_gpu = neg_tp_cpu.to(self.device)  # [B, 3, H, W] 或 [3, H, W]
                
        #         # 负样本转换 SDR->ICTCP
        #         neg_ictcp = SDR_to_YCbCr(neg_tp_gpu, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                
        #         # 克隆原图 ictcp
        #         sdr_ictcp_tp = sdr_ictcp_base.clone()
                
        #         # 替换 T,P 通道 (index=1,2)
        #         sdr_ictcp_tp[:, 1:3, :, :] = neg_ictcp[:, 1:3, :, :]
                
        #         # 回到 SDR
        #         replaced_sdr_tp = YCbCr_to_SDR(sdr_ictcp_tp, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
        #         im_negs_TP_list.append(replaced_sdr_tp)
            
        #     if len(im_negs_TP_list) > 0:
        #         self.im_negs_TP = torch.stack(im_negs_TP_list, dim=1)  # [N, B, 3, H, W] 或 [N, 3, H, W]
        #         # 如果 N=1 且 B=1，可以简化维度
        #         if self.im_negs_TP.size(1) == 1:
        #             self.im_negs_TP = self.im_negs_TP.squeeze(1)  # [N, 3, H, W] 或 [N, 3, H, W]
        #         self.im_negs_TP = apply_random_flip(self.im_negs_TP)

        # 这样 self.im_negs_I 和 self.im_negs_TP 就是 GPU 上处理好的负样本

    
    def feed_data_test(self, data):
        # 获取输入数据并放入设备
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        # 生成对比学习的正样本和负样本
        self.im_q = data['im_q'].to(self.device)
        
    

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']
        # 清除梯度
        self.optimizer_g.zero_grad()
        # self.optimizer_feat_I.zero_grad()
        # self.optimizer_feat_TP.zero_grad()
        
        # 初始化总损失
        l_total = 0
        loss_dict = OrderedDict()

        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            # 使用对比学习网络的 UNet 部分 (unet_q)
            # z_I, y_I = self.feat_extractor_I.unet_q(self.im_q)   # in:(B, 1, H, W)
            # z_TP, y_TP = self.feat_extractor_TP.unet_q(self.im_q) # in:(B, 2, H, W)

            # 超分辨率模型前向传播
            self.output = self.net_g(self.lq)
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


                
        # 反向传播和优化
        if self.scaler is not None:
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            # self.scaler.step(self.optimizer_feat_I)
            # self.scaler.step(self.optimizer_feat_TP)
            self.scaler.update()
        else:
            # 否则直接普通 backward
            l_total.backward()
            self.optimizer_g.step()
            # self.optimizer_feat_I.step()
            # self.optimizer_feat_TP.step()


        # 保存日志
        self.log_dict = self.reduce_loss_dict(loss_dict)

        # 如果启用了EMA，则更新EMA模型权重
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
            # self.contrast_model_ema(decay=self.ema_decay)

    def test(self):
        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):  # (###### [FP16 CHANGE])
            # 使用EMA模型进行测试
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                # self.feat_extractor_I.eval()
                # self.feat_extractor_TP.eval()
                with torch.no_grad():
                    # 获取特征编码
                    # z_I, y_I = self.feat_extractor_I.unet_k(self.im_q)
                    # z_TP, y_TP = self.feat_extractor_TP.unet_k(self.im_q)
                    # 使用特征编码进行推理
                    self.output = self.net_g_ema(self.lq)
                    self.output_mid = self.net_g_ema.mid_result
            else:
                # 使用原始模型和feat_extractor进行测试
                self.net_g.eval()
                # self.feat_extractor_I.eval()
                # self.feat_extractor_TP.eval()
                with torch.no_grad():
                    # 获取特征编码
                    # z_I, y_I = self.feat_extractor_I.unet_k(self.im_q)
                    # z_TP, y_TP = self.feat_extractor_TP.unet_k(self.im_q)
                    # 使用特征编码进行推理
                    self.output = self.net_g(self.lq)
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
        # 构建要保存的网络和对应的键名列表
        nets_to_save = [self.net_g]
        param_keys = ['params']

        if hasattr(self, 'net_g_ema'):
            nets_to_save.append(self.net_g_ema)
            param_keys.append('params_ema')

        # nets_to_save.append(self.feat_extractor_I)
        # param_keys.append('feat_params_I')
        
        # nets_to_save.append(self.feat_extractor_TP)
        # param_keys.append('feat_params_TP')

        # if hasattr(self, 'feat_extractor_ema'):
        #     nets_to_save.append(self.feat_extractor_ema)
        #     param_keys.append('feat_params_ema')

        # 调用 save_network 并确保 nets_to_save 和 param_keys 长度匹配
        self.save_network(
            net=nets_to_save,
            net_label='net_g',
            current_iter=current_iter,
            param_key=param_keys
        )

        # 保存训练状态
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
class ContrastYCBCRMidSupITMModel_noinit(ITMModel_copy):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # 定义网络
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
        
        # 加载预训练模型
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
        # 如果是训练模式，则初始化训练设置
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        
        # (###### [BF16 CHANGE]) 检查是否使用 BF16
        self.use_bf16 = train_opt.get('use_bf16', False)  # 在配置文件里加一个 use_bf16 = true/false
        if self.use_bf16:
            logger = get_root_logger()
            logger.info('Using bfloat16 mixed precision training ...')  # 修正日志信息
        
        # (###### [BF16 CHANGE]) 初始化 GradScaler
        self.scaler = GradScaler() if self.use_bf16 else None  # BF16 通常需要 GradScaler
        
        # 定义EMA衰减
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
            
            # if load_path is None:
            #     # 没有预训练，就执行自定义初始化
            #     if hasattr(self.net_g, 'init_weights'):
            #         self.net_g.init_weights()
            # else:
            #     logger.info(f'Will load pretrained from {load_path}, skip init_weights')
    
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()
            # if load_path_feat is not None:
            #     self.load_network(self.feat_extractor_ema, load_path_feat, self.opt['path'].get('strict_load_g', True), 'feat_params_ema')
            # else:
            #     self.contrast_model_ema(0)  # copy net_g weight
            # self.feat_extractor_ema.eval()
        
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
        # 原图
        self.lq = data['lq'].to(self.device)  # [B, 3, H, W]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        
        # im_q, im_k 也上GPU
        self.im_q = data['im_q'].to(self.device)
        self.im_k = data['im_k'].to(self.device)

        # 先对原图做一次 SDR->ICTCP (放到GPU计算)
        # 直接传递 [B, 3, H, W]，无需 unsqueeze 和 squeeze
        dim = 1  # 通道维度
        sdr_ictcp_base = SDR_to_YCbCr(self.lq, dim=dim)  # shape [B, 3, H, W], GPU

        # --------------- 亮度负样本 ---------------
        self.im_negs_I = None
        if data['im_negs_I_raw'] is not None:
            im_negs_I_list = []  # 存放替换后张量
            for i, neg_i_cpu in enumerate(data['im_negs_I_raw']):
                # 负样本上 GPU
                neg_i_gpu = neg_i_cpu.to(self.device)  # [B, 3, H, W] 或 [3, H, W]
                
                # 负样本转换 SDR->ICTCP
                neg_ictcp = SDR_to_YCbCr(neg_i_gpu, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                
                # 克隆
                sdr_ictcp_i = sdr_ictcp_base.clone()
                
                # 替换 I 通道 (index=0)
                sdr_ictcp_i[:, 0, :, :] = neg_ictcp[:, 0, :, :]
                
                # 回到 SDR
                replaced_sdr_i = YCbCr_to_SDR(sdr_ictcp_i, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                im_negs_I_list.append(replaced_sdr_i)
            
            if len(im_negs_I_list) > 0:
                self.im_negs_I = torch.stack(im_negs_I_list, dim=1)  # [N, B, 3, H, W] 或 [N, 3, H, W]
                # 如果 N=1 且 B=1，可以简化维度
                if self.im_negs_I.size(1) == 1:
                    self.im_negs_I = self.im_negs_I.squeeze(1)  # [N, 3, H, W] 或 [N, 3, H, W]
                self.im_negs_I = apply_random_flip(self.im_negs_I)
        
        # --------------- 色度负样本 ---------------
        self.im_negs_TP = None
        if data['im_negs_TP_raw'] is not None:
            im_negs_TP_list = []
            for i, neg_tp_cpu in enumerate(data['im_negs_TP_raw']):
                neg_tp_gpu = neg_tp_cpu.to(self.device)  # [B, 3, H, W] 或 [3, H, W]
                
                # 负样本转换 SDR->ICTCP
                neg_ictcp = SDR_to_YCbCr(neg_tp_gpu, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                
                # 克隆原图 ictcp
                sdr_ictcp_tp = sdr_ictcp_base.clone()
                
                # 替换 T,P 通道 (index=1,2)
                sdr_ictcp_tp[:, 1:3, :, :] = neg_ictcp[:, 1:3, :, :]
                
                # 回到 SDR
                replaced_sdr_tp = YCbCr_to_SDR(sdr_ictcp_tp, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                im_negs_TP_list.append(replaced_sdr_tp)
            
            if len(im_negs_TP_list) > 0:
                self.im_negs_TP = torch.stack(im_negs_TP_list, dim=1)  # [N, B, 3, H, W] 或 [N, 3, H, W]
                # 如果 N=1 且 B=1，可以简化维度
                if self.im_negs_TP.size(1) == 1:
                    self.im_negs_TP = self.im_negs_TP.squeeze(1)  # [N, 3, H, W] 或 [N, 3, H, W]
                self.im_negs_TP = apply_random_flip(self.im_negs_TP)

        # 这样 self.im_negs_I 和 self.im_negs_TP 就是 GPU 上处理好的负样本

    
    def feed_data_test(self, data):
        # 获取输入数据并放入设备
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        # 生成对比学习的正样本和负样本
        self.im_q = data['im_q'].to(self.device)
        # self.im_k = data['im_k'].to(self.device)
        # self.im_negs_I = data['im_negs_I'].to(self.device)
        # self.im_negs_TP = data['im_negs_TP'].to(self.device)
    
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
        
        # 定义 net_g 参数
        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} in net_g will not be optimized.')

        # 定义 feat_extractor 参数
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
        
        # 保存优化器到优化器列表
        self.optimizers.append(self.optimizer_g)
        self.optimizers.append(self.optimizer_feat_I)
        self.optimizers.append(self.optimizer_feat_TP)
    
    def save_visualizations(self, current_iter, save_dir='saved_images'):
        """
        保存 im_q、lq、mid_result、output 和 gt 作为图像。

        参数:
            current_iter (int): 当前迭代次数，用于文件命名。
            save_dir (str): 图像保存目录。
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 定义迭代后缀
        iter_suffix = f'iter_{current_iter}'
        
        # 选择要保存的样本索引（例如第一个样本）
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
        # 清除梯度
        self.optimizer_g.zero_grad()
        self.optimizer_feat_I.zero_grad()
        self.optimizer_feat_TP.zero_grad()
        
        # 初始化总损失
        l_total = 0
        loss_dict = OrderedDict()

        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            if current_iter <= train_opt['stage1_iters']:  # 阶段 1：仅更新主模型，不引入对比学习
                # 使用对比学习网络的 UNet 部分 (unet_q)
                z_I, y_I = self.feat_extractor_I.unet_q(self.im_q)   # in:(B, 1, H, W)
                z_TP, y_TP = self.feat_extractor_TP.unet_q(self.im_q) # in:(B, 2, H, W)

                # 超分辨率模型前向传播
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
                
                # # 可视化部分：按log_interval频率保存图像
                # log_interval = train_opt.get('log_interval', 1)  # 默认每100次迭代保存一次
                # if current_iter % log_interval == 0:
                #     self.save_visualizations(current_iter, save_dir='saved_images_stage1')
                
                self.feat_extractor_I._momentum_update_key_encoder()
                self.feat_extractor_TP._momentum_update_key_encoder()
                
                # self.feat_extractor_I.reset_unet_k()
                # self.feat_extractor_TP.reset_unet_k()

            else:  # 阶段 2：引入对比学习
                # 对比学习网络前向传播
                # save_negatives(self.im_negs_I, self.im_negs_TP, batch_index=0)
                logits_global_I, labels_global_I, logits_local_I, labels_local_I, z_I, y_I = self.feat_extractor_I(self.im_q, self.im_k, self.im_negs_I)
                
                logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP, z_TP, y_TP = self.feat_extractor_TP(self.im_q, self.im_k, self.im_negs_TP)

                # 全局对比学习损失
                global_contra_loss_I = self.criterion_contra(logits_global_I, labels_global_I)
                loss_dict['l_global_contra_I'] = global_contra_loss_I
                l_total += global_contra_loss_I * self.cri_contra_weight_global * self.cri_contra_weight
                
                global_contra_loss_TP = self.criterion_contra(logits_global_TP, labels_global_TP)
                loss_dict['l_global_contra_TP'] = global_contra_loss_TP
                l_total += global_contra_loss_TP * self.cri_contra_weight_global * self.cri_contra_weight

                # 局部对比学习损失
                local_contra_loss_I = self.criterion_contra(logits_local_I, labels_local_I)
                loss_dict['l_local_contra_I'] = local_contra_loss_I
                l_total += local_contra_loss_I * self.cri_contra_weight_local * self.cri_contra_weight
                
                local_contra_loss_TP = self.criterion_contra(logits_local_TP, labels_local_TP)
                loss_dict['l_local_contra_TP'] = local_contra_loss_TP
                l_total += local_contra_loss_TP * self.cri_contra_weight_local * self.cri_contra_weight


                # 超分辨率模型前向传播
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
                    
                # # 可视化部分：按log_interval频率保存图像
                # log_interval = train_opt.get('log_interval', 100)  # 默认每100次迭代保存一次
                # if current_iter % log_interval == 0:
                #     self.save_visualizations(current_iter, save_dir='saved_images_stage2')

                
        # 反向传播和优化
        if self.scaler is not None:
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.step(self.optimizer_feat_I)
            self.scaler.step(self.optimizer_feat_TP)
            self.scaler.update()
        else:
            # 否则直接普通 backward
            l_total.backward()
            self.optimizer_g.step()
            self.optimizer_feat_I.step()
            self.optimizer_feat_TP.step()

        # # 计算度量指标（metrics）
        # if self.opt['val'].get('metrics') is not None:
        #     if not hasattr(self, 'metric_results'):  # only execute in the first run
        #         self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        #     # 初始化 metric_data 并进行指标计算
        #     metric_data = {'img': self.output.detach().cpu().numpy(), 'img2': self.gt.detach().cpu().numpy()}
        #     for name, opt_ in self.opt['val']['metrics'].items():
        #         metric_val = calculate_metric(metric_data, opt_)
        #         self.metric_results[name] = metric_val / self.opt['datasets']['train']['batch_size_per_gpu']
        #         # 打印或记录每个度量指标
        #         loss_dict[f'{name}'] = metric_val
        
        # # 创建保存路径
        # save_dir = self.opt['path'].get('visualization', './visualization')
        # os.makedirs(save_dir, exist_ok=True)

        # # 遍历 batch 并保存每张图片
        # for i in range(self.output.size(0)):
        #     output_img = tensor2img([self.output[i]])  # 将 Tensor 转换为图像
        #     gt_img = tensor2img([self.gt[i]])  # 目标图像

        #     # 保存图像路径
        #     save_output_path = os.path.join(save_dir, f'output_{current_iter}_img_{i}.png')
        #     save_gt_path = os.path.join(save_dir, f'gt_{current_iter}_img_{i}.png')
            
        #     cv2.imwrite(save_output_path, output_img)
        #     cv2.imwrite(save_gt_path, gt_img)
                
        # # 反向传播和优化
        # l_total.backward()
        # self.optimizer_g.step()
        # self.optimizer_feat.step()
        # 使用 scaler 缩放损失，进行反向传播和优化
            # self.scaler.scale(l_total).backward()
            # self.scaler.step(self.optimizer_g)
            # self.scaler.step(self.optimizer_feat)
            # self.scaler.update()

        # 保存日志
        self.log_dict = self.reduce_loss_dict(loss_dict)

        # 如果启用了EMA，则更新EMA模型权重
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
            # self.contrast_model_ema(decay=self.ema_decay)

    def test(self):
        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):  # (###### [FP16 CHANGE])
            # 使用EMA模型进行测试
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.feat_extractor_I.eval()
                self.feat_extractor_TP.eval()
                with torch.no_grad():
                    # 获取特征编码
                    z_I, y_I = self.feat_extractor_I.unet_k(self.im_q)
                    z_TP, y_TP = self.feat_extractor_TP.unet_k(self.im_q)
                    # 使用特征编码进行推理
                    self.output = self.net_g_ema(self.lq, z_I, z_TP, y_I, y_TP)
                    self.output_mid = self.net_g_ema.mid_result
            else:
                # 使用原始模型和feat_extractor进行测试
                self.net_g.eval()
                self.feat_extractor_I.eval()
                self.feat_extractor_TP.eval()
                with torch.no_grad():
                    # 获取特征编码
                    z_I, y_I = self.feat_extractor_I.unet_k(self.im_q)
                    z_TP, y_TP = self.feat_extractor_TP.unet_k(self.im_q)
                    # 使用特征编码进行推理
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
        # 构建要保存的网络和对应的键名列表
        nets_to_save = [self.net_g]
        param_keys = ['params']

        if hasattr(self, 'net_g_ema'):
            nets_to_save.append(self.net_g_ema)
            param_keys.append('params_ema')

        nets_to_save.append(self.feat_extractor_I)
        param_keys.append('feat_params_I')
        
        nets_to_save.append(self.feat_extractor_TP)
        param_keys.append('feat_params_TP')

        # if hasattr(self, 'feat_extractor_ema'):
        #     nets_to_save.append(self.feat_extractor_ema)
        #     param_keys.append('feat_params_ema')

        # 调用 save_network 并确保 nets_to_save 和 param_keys 长度匹配
        self.save_network(
            net=nets_to_save,
            net_label='net_g',
            current_iter=current_iter,
            param_key=param_keys
        )

        # 保存训练状态
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
class ContrastYCBCRMidSupITMModel_woglobal(ITMModel_copy):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # 定义网络
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        
        self.feat_extractor_I = MoCoWithUNet_woglobal(in_channels=3, global_cond_channels=64, spatial_cond_channels=16, temperature=1.0)
        self.feat_extractor_I = self.model_to_device(self.feat_extractor_I)
        self.print_network(self.feat_extractor_I)
        
        self.feat_extractor_TP = MoCoWithUNet_woglobal(in_channels=3, global_cond_channels=64, spatial_cond_channels=16, temperature=1.0)
        self.feat_extractor_TP = self.model_to_device(self.feat_extractor_TP)
        self.print_network(self.feat_extractor_TP)
        
        # self.scaler = GradScaler()
        self.average_metrics = {}
        
        # 加载预训练模型
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
        # 如果是训练模式，则初始化训练设置
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        
        # (###### [BF16 CHANGE]) 检查是否使用 BF16
        self.use_bf16 = train_opt.get('use_bf16', False)  # 在配置文件里加一个 use_bf16 = true/false
        if self.use_bf16:
            logger = get_root_logger()
            logger.info('Using bfloat16 mixed precision training ...')  # 修正日志信息
        
        # (###### [BF16 CHANGE]) 初始化 GradScaler
        self.scaler = GradScaler() if self.use_bf16 else None  # BF16 通常需要 GradScaler
        
        # 定义EMA衰减
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
            
            # if load_path is None:
            #     # 没有预训练，就执行自定义初始化
            #     if hasattr(self.net_g, 'init_weights'):
            #         self.net_g.init_weights()
            # else:
            #     logger.info(f'Will load pretrained from {load_path}, skip init_weights')
    
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()
            # if load_path_feat is not None:
            #     self.load_network(self.feat_extractor_ema, load_path_feat, self.opt['path'].get('strict_load_g', True), 'feat_params_ema')
            # else:
            #     self.contrast_model_ema(0)  # copy net_g weight
            # self.feat_extractor_ema.eval()
        
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
        # 原图
        self.lq = data['lq'].to(self.device)  # [B, 3, H, W]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        
        # im_q, im_k 也上GPU
        self.im_q = data['im_q'].to(self.device)
        self.im_k = data['im_k'].to(self.device)

        # 先对原图做一次 SDR->ICTCP (放到GPU计算)
        # 直接传递 [B, 3, H, W]，无需 unsqueeze 和 squeeze
        dim = 1  # 通道维度
        sdr_ictcp_base = SDR_to_YCbCr(self.lq, dim=dim)  # shape [B, 3, H, W], GPU

        # --------------- 亮度负样本 ---------------
        self.im_negs_I = None
        if data['im_negs_I_raw'] is not None:
            im_negs_I_list = []  # 存放替换后张量
            for i, neg_i_cpu in enumerate(data['im_negs_I_raw']):
                # 负样本上 GPU
                neg_i_gpu = neg_i_cpu.to(self.device)  # [B, 3, H, W] 或 [3, H, W]
                
                # 负样本转换 SDR->ICTCP
                neg_ictcp = SDR_to_YCbCr(neg_i_gpu, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                
                # 克隆
                sdr_ictcp_i = sdr_ictcp_base.clone()
                
                # 替换 I 通道 (index=0)
                sdr_ictcp_i[:, 0, :, :] = neg_ictcp[:, 0, :, :]
                
                # 回到 SDR
                replaced_sdr_i = YCbCr_to_SDR(sdr_ictcp_i, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                im_negs_I_list.append(replaced_sdr_i)
            
            if len(im_negs_I_list) > 0:
                self.im_negs_I = torch.stack(im_negs_I_list, dim=1)  # [N, B, 3, H, W] 或 [N, 3, H, W]
                # 如果 N=1 且 B=1，可以简化维度
                if self.im_negs_I.size(1) == 1:
                    self.im_negs_I = self.im_negs_I.squeeze(1)  # [N, 3, H, W] 或 [N, 3, H, W]
                self.im_negs_I = apply_random_flip(self.im_negs_I)
        
        # --------------- 色度负样本 ---------------
        self.im_negs_TP = None
        if data['im_negs_TP_raw'] is not None:
            im_negs_TP_list = []
            for i, neg_tp_cpu in enumerate(data['im_negs_TP_raw']):
                neg_tp_gpu = neg_tp_cpu.to(self.device)  # [B, 3, H, W] 或 [3, H, W]
                
                # 负样本转换 SDR->ICTCP
                neg_ictcp = SDR_to_YCbCr(neg_tp_gpu, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                
                # 克隆原图 ictcp
                sdr_ictcp_tp = sdr_ictcp_base.clone()
                
                # 替换 T,P 通道 (index=1,2)
                sdr_ictcp_tp[:, 1:3, :, :] = neg_ictcp[:, 1:3, :, :]
                
                # 回到 SDR
                replaced_sdr_tp = YCbCr_to_SDR(sdr_ictcp_tp, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                im_negs_TP_list.append(replaced_sdr_tp)
            
            if len(im_negs_TP_list) > 0:
                self.im_negs_TP = torch.stack(im_negs_TP_list, dim=1)  # [N, B, 3, H, W] 或 [N, 3, H, W]
                # 如果 N=1 且 B=1，可以简化维度
                if self.im_negs_TP.size(1) == 1:
                    self.im_negs_TP = self.im_negs_TP.squeeze(1)  # [N, 3, H, W] 或 [N, 3, H, W]
                self.im_negs_TP = apply_random_flip(self.im_negs_TP)

        # 这样 self.im_negs_I 和 self.im_negs_TP 就是 GPU 上处理好的负样本

    
    def feed_data_test(self, data):
        # 获取输入数据并放入设备
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        # 生成对比学习的正样本和负样本
        self.im_q = data['im_q'].to(self.device)
        # self.im_k = data['im_k'].to(self.device)
        # self.im_negs_I = data['im_negs_I'].to(self.device)
        # self.im_negs_TP = data['im_negs_TP'].to(self.device)
    
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
        
        # 定义 net_g 参数
        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} in net_g will not be optimized.')

        # 定义 feat_extractor 参数
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
        
        # 保存优化器到优化器列表
        self.optimizers.append(self.optimizer_g)
        self.optimizers.append(self.optimizer_feat_I)
        self.optimizers.append(self.optimizer_feat_TP)
    
    def save_visualizations(self, current_iter, save_dir='saved_images'):
        """
        保存 im_q、lq、mid_result、output 和 gt 作为图像。

        参数:
            current_iter (int): 当前迭代次数，用于文件命名。
            save_dir (str): 图像保存目录。
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 定义迭代后缀
        iter_suffix = f'iter_{current_iter}'
        
        # 选择要保存的样本索引（例如第一个样本）
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
        # 清除梯度
        self.optimizer_g.zero_grad()
        self.optimizer_feat_I.zero_grad()
        self.optimizer_feat_TP.zero_grad()
        
        # 初始化总损失
        l_total = 0
        loss_dict = OrderedDict()

        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            if current_iter <= train_opt['stage1_iters']:  # 阶段 1：仅更新主模型，不引入对比学习
                # 使用对比学习网络的 UNet 部分 (unet_q)
                y_I = self.feat_extractor_I.unet_q(self.im_q)   # in:(B, 1, H, W)
                y_TP = self.feat_extractor_TP.unet_q(self.im_q) # in:(B, 2, H, W)

                # 超分辨率模型前向传播
                self.output = self.net_g(self.lq, y_I, y_TP)
                # self.output_mid = self.net_g.mid_result

                # pixel loss
                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                # # intermediate supervision loss
                # if self.cri_pix_mid:
                #     l_pix_mid = self.cri_pix_mid(self.output_mid, self.gt)
                #     l_total += l_pix_mid
                #     loss_dict['l_pix_mid'] = l_pix_mid
                
                # # 可视化部分：按log_interval频率保存图像
                # log_interval = train_opt.get('log_interval', 1)  # 默认每100次迭代保存一次
                # if current_iter % log_interval == 0:
                #     self.save_visualizations(current_iter, save_dir='saved_images_stage1')
                
                # self.feat_extractor_I._momentum_update_key_encoder()
                # self.feat_extractor_TP._momentum_update_key_encoder()
                
                # self.feat_extractor_I.reset_unet_k()
                # self.feat_extractor_TP.reset_unet_k()

            else:  # 阶段 2：引入对比学习
                # 对比学习网络前向传播
                # save_negatives(self.im_negs_I, self.im_negs_TP, batch_index=0)
                logits_local_I, labels_local_I, y_I = self.feat_extractor_I(self.im_q, self.im_k, self.im_negs_I)
                
                logits_local_TP, labels_local_TP, y_TP = self.feat_extractor_TP(self.im_q, self.im_k, self.im_negs_TP)

                # # 全局对比学习损失
                # global_contra_loss_I = self.criterion_contra(logits_global_I, labels_global_I)
                # loss_dict['l_global_contra_I'] = global_contra_loss_I
                # l_total += global_contra_loss_I * self.cri_contra_weight_global * self.cri_contra_weight
                
                # global_contra_loss_TP = self.criterion_contra(logits_global_TP, labels_global_TP)
                # loss_dict['l_global_contra_TP'] = global_contra_loss_TP
                # l_total += global_contra_loss_TP * self.cri_contra_weight_global * self.cri_contra_weight

                # 局部对比学习损失
                local_contra_loss_I = self.criterion_contra(logits_local_I, labels_local_I)
                loss_dict['l_local_contra_I'] = local_contra_loss_I
                l_total += local_contra_loss_I * self.cri_contra_weight_local * self.cri_contra_weight
                
                local_contra_loss_TP = self.criterion_contra(logits_local_TP, labels_local_TP)
                loss_dict['l_local_contra_TP'] = local_contra_loss_TP
                l_total += local_contra_loss_TP * self.cri_contra_weight_local * self.cri_contra_weight


                # 超分辨率模型前向传播
                self.output = self.net_g(self.lq, y_I, y_TP)
                # self.output_mid = self.net_g.mid_result

                # pixel loss
                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                # # intermediate supervision loss
                # if self.cri_pix_mid:
                #     l_pix_mid = self.cri_pix_mid(self.output_mid, self.gt)
                #     l_total += l_pix_mid
                #     loss_dict['l_pix_mid'] = l_pix_mid
                    
                # # 可视化部分：按log_interval频率保存图像
                # log_interval = train_opt.get('log_interval', 100)  # 默认每100次迭代保存一次
                # if current_iter % log_interval == 0:
                #     self.save_visualizations(current_iter, save_dir='saved_images_stage2')

                
        # 反向传播和优化
        if self.scaler is not None:
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.step(self.optimizer_feat_I)
            self.scaler.step(self.optimizer_feat_TP)
            self.scaler.update()
        else:
            # 否则直接普通 backward
            l_total.backward()
            self.optimizer_g.step()
            self.optimizer_feat_I.step()
            self.optimizer_feat_TP.step()



        # 保存日志
        self.log_dict = self.reduce_loss_dict(loss_dict)

        # 如果启用了EMA，则更新EMA模型权重
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
            # self.contrast_model_ema(decay=self.ema_decay)

    def test(self):
        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):  # (###### [FP16 CHANGE])
            # 使用EMA模型进行测试
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.feat_extractor_I.eval()
                self.feat_extractor_TP.eval()
                with torch.no_grad():
                    # 获取特征编码
                    y_I = self.feat_extractor_I.unet_k(self.im_q)
                    y_TP = self.feat_extractor_TP.unet_k(self.im_q)
                    # 使用特征编码进行推理
                    self.output = self.net_g_ema(self.lq, y_I, y_TP)
                    # self.output_mid = self.net_g_ema.mid_result
            else:
                # 使用原始模型和feat_extractor进行测试
                self.net_g.eval()
                self.feat_extractor_I.eval()
                self.feat_extractor_TP.eval()
                with torch.no_grad():
                    # 获取特征编码
                    y_I = self.feat_extractor_I.unet_k(self.im_q)
                    y_TP = self.feat_extractor_TP.unet_k(self.im_q)
                    # 使用特征编码进行推理
                    self.output = self.net_g(self.lq, y_I, y_TP)
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
            # self.save_batch_images(val_data, idx)
            self.feed_data_test(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = visuals['result']
            # sr_img_mid = visuals['mid_result']
            gt_img = visuals['gt'] if 'gt' in visuals else None

            sr_img = torch.clamp(sr_img, 0.0, 1.0)
            gt_img = torch.clamp(gt_img, 0.0, 1.0)
            
            if to_rgb:
                convert_to_rgb = HdrConvertBackToRGB(src=to_rgb, zero_one_normed=True)
                sr_img = convert_to_rgb(sr_img)
                # sr_img_mid = convert_to_rgb(sr_img_mid)
                gt_img = convert_to_rgb(gt_img) if gt_img is not None else None

            sr_img = tensor2img([sr_img], out_type=np.uint16)
            # sr_img_mid = tensor2img([sr_img_mid])
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
        # 构建要保存的网络和对应的键名列表
        nets_to_save = [self.net_g]
        param_keys = ['params']

        if hasattr(self, 'net_g_ema'):
            nets_to_save.append(self.net_g_ema)
            param_keys.append('params_ema')

        nets_to_save.append(self.feat_extractor_I)
        param_keys.append('feat_params_I')
        
        nets_to_save.append(self.feat_extractor_TP)
        param_keys.append('feat_params_TP')

        # if hasattr(self, 'feat_extractor_ema'):
        #     nets_to_save.append(self.feat_extractor_ema)
        #     param_keys.append('feat_params_ema')

        # 调用 save_network 并确保 nets_to_save 和 param_keys 长度匹配
        self.save_network(
            net=nets_to_save,
            net_label='net_g',
            current_iter=current_iter,
            param_key=param_keys
        )

        # 保存训练状态
        self.save_training_state(epoch, current_iter)
    
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        # out_dict['mid_result'] = self.output_mid.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
    
@MODEL_REGISTRY.register()
class ContrastYCBCRMidSupITMModel_wolocal(ITMModel_copy):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # 定义网络
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        
        self.feat_extractor_I = MoCoWithUNet_wolocal(in_channels=3, global_cond_channels=64, spatial_cond_channels=16, temperature=1.0)
        self.feat_extractor_I = self.model_to_device(self.feat_extractor_I)
        self.print_network(self.feat_extractor_I)
        
        self.feat_extractor_TP = MoCoWithUNet_wolocal(in_channels=3, global_cond_channels=64, spatial_cond_channels=16, temperature=1.0)
        self.feat_extractor_TP = self.model_to_device(self.feat_extractor_TP)
        self.print_network(self.feat_extractor_TP)
        
        # self.scaler = GradScaler()
        self.average_metrics = {}
        
        # 加载预训练模型
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
        # 如果是训练模式，则初始化训练设置
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        
        # (###### [BF16 CHANGE]) 检查是否使用 BF16
        self.use_bf16 = train_opt.get('use_bf16', False)  # 在配置文件里加一个 use_bf16 = true/false
        if self.use_bf16:
            logger = get_root_logger()
            logger.info('Using bfloat16 mixed precision training ...')  # 修正日志信息
        
        # (###### [BF16 CHANGE]) 初始化 GradScaler
        self.scaler = GradScaler() if self.use_bf16 else None  # BF16 通常需要 GradScaler
        
        # 定义EMA衰减
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
            
            # if load_path is None:
            #     # 没有预训练，就执行自定义初始化
            #     if hasattr(self.net_g, 'init_weights'):
            #         self.net_g.init_weights()
            # else:
            #     logger.info(f'Will load pretrained from {load_path}, skip init_weights')
    
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()
            # if load_path_feat is not None:
            #     self.load_network(self.feat_extractor_ema, load_path_feat, self.opt['path'].get('strict_load_g', True), 'feat_params_ema')
            # else:
            #     self.contrast_model_ema(0)  # copy net_g weight
            # self.feat_extractor_ema.eval()
        
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
        # 原图
        self.lq = data['lq'].to(self.device)  # [B, 3, H, W]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        
        # im_q, im_k 也上GPU
        self.im_q = data['im_q'].to(self.device)
        self.im_k = data['im_k'].to(self.device)

        # 先对原图做一次 SDR->ICTCP (放到GPU计算)
        # 直接传递 [B, 3, H, W]，无需 unsqueeze 和 squeeze
        dim = 1  # 通道维度
        sdr_ictcp_base = SDR_to_YCbCr(self.lq, dim=dim)  # shape [B, 3, H, W], GPU

        # --------------- 亮度负样本 ---------------
        self.im_negs_I = None
        if data['im_negs_I_raw'] is not None:
            im_negs_I_list = []  # 存放替换后张量
            for i, neg_i_cpu in enumerate(data['im_negs_I_raw']):
                # 负样本上 GPU
                neg_i_gpu = neg_i_cpu.to(self.device)  # [B, 3, H, W] 或 [3, H, W]
                
                # 负样本转换 SDR->ICTCP
                neg_ictcp = SDR_to_YCbCr(neg_i_gpu, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                
                # 克隆
                sdr_ictcp_i = sdr_ictcp_base.clone()
                
                # 替换 I 通道 (index=0)
                sdr_ictcp_i[:, 0, :, :] = neg_ictcp[:, 0, :, :]
                
                # 回到 SDR
                replaced_sdr_i = YCbCr_to_SDR(sdr_ictcp_i, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                im_negs_I_list.append(replaced_sdr_i)
            
            if len(im_negs_I_list) > 0:
                self.im_negs_I = torch.stack(im_negs_I_list, dim=1)  # [N, B, 3, H, W] 或 [N, 3, H, W]
                # 如果 N=1 且 B=1，可以简化维度
                if self.im_negs_I.size(1) == 1:
                    self.im_negs_I = self.im_negs_I.squeeze(1)  # [N, 3, H, W] 或 [N, 3, H, W]
                self.im_negs_I = apply_random_flip(self.im_negs_I)
        
        # --------------- 色度负样本 ---------------
        self.im_negs_TP = None
        if data['im_negs_TP_raw'] is not None:
            im_negs_TP_list = []
            for i, neg_tp_cpu in enumerate(data['im_negs_TP_raw']):
                neg_tp_gpu = neg_tp_cpu.to(self.device)  # [B, 3, H, W] 或 [3, H, W]
                
                # 负样本转换 SDR->ICTCP
                neg_ictcp = SDR_to_YCbCr(neg_tp_gpu, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                
                # 克隆原图 ictcp
                sdr_ictcp_tp = sdr_ictcp_base.clone()
                
                # 替换 T,P 通道 (index=1,2)
                sdr_ictcp_tp[:, 1:3, :, :] = neg_ictcp[:, 1:3, :, :]
                
                # 回到 SDR
                replaced_sdr_tp = YCbCr_to_SDR(sdr_ictcp_tp, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                im_negs_TP_list.append(replaced_sdr_tp)
            
            if len(im_negs_TP_list) > 0:
                self.im_negs_TP = torch.stack(im_negs_TP_list, dim=1)  # [N, B, 3, H, W] 或 [N, 3, H, W]
                # 如果 N=1 且 B=1，可以简化维度
                if self.im_negs_TP.size(1) == 1:
                    self.im_negs_TP = self.im_negs_TP.squeeze(1)  # [N, 3, H, W] 或 [N, 3, H, W]
                self.im_negs_TP = apply_random_flip(self.im_negs_TP)

        # 这样 self.im_negs_I 和 self.im_negs_TP 就是 GPU 上处理好的负样本

    
    def feed_data_test(self, data):
        # 获取输入数据并放入设备
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        # 生成对比学习的正样本和负样本
        self.im_q = data['im_q'].to(self.device)
        # self.im_k = data['im_k'].to(self.device)
        # self.im_negs_I = data['im_negs_I'].to(self.device)
        # self.im_negs_TP = data['im_negs_TP'].to(self.device)
    
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
        
        # 定义 net_g 参数
        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} in net_g will not be optimized.')

        # 定义 feat_extractor 参数
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
        
        # 保存优化器到优化器列表
        self.optimizers.append(self.optimizer_g)
        self.optimizers.append(self.optimizer_feat_I)
        self.optimizers.append(self.optimizer_feat_TP)
    
    def save_visualizations(self, current_iter, save_dir='saved_images'):
        """
        保存 im_q、lq、mid_result、output 和 gt 作为图像。

        参数:
            current_iter (int): 当前迭代次数，用于文件命名。
            save_dir (str): 图像保存目录。
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 定义迭代后缀
        iter_suffix = f'iter_{current_iter}'
        
        # 选择要保存的样本索引（例如第一个样本）
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
        # 清除梯度
        self.optimizer_g.zero_grad()
        self.optimizer_feat_I.zero_grad()
        self.optimizer_feat_TP.zero_grad()
        
        # 初始化总损失
        l_total = 0
        loss_dict = OrderedDict()

        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            if current_iter <= train_opt['stage1_iters']:  # 阶段 1：仅更新主模型，不引入对比学习
                # 使用对比学习网络的 UNet 部分 (unet_q)
                z_I = self.feat_extractor_I.unet_q(self.im_q)   # in:(B, 1, H, W)
                z_TP = self.feat_extractor_TP.unet_q(self.im_q) # in:(B, 2, H, W)

                # 超分辨率模型前向传播
                self.output = self.net_g(self.lq, z_I, z_TP)
                # self.output_mid = self.net_g.mid_result

                # pixel loss
                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                # # intermediate supervision loss
                # if self.cri_pix_mid:
                #     l_pix_mid = self.cri_pix_mid(self.output_mid, self.gt)
                #     l_total += l_pix_mid
                #     loss_dict['l_pix_mid'] = l_pix_mid
                
                # # 可视化部分：按log_interval频率保存图像
                # log_interval = train_opt.get('log_interval', 1)  # 默认每100次迭代保存一次
                # if current_iter % log_interval == 0:
                #     self.save_visualizations(current_iter, save_dir='saved_images_stage1')
                
                # self.feat_extractor_I._momentum_update_key_encoder()
                # self.feat_extractor_TP._momentum_update_key_encoder()
                
                # self.feat_extractor_I.reset_unet_k()
                # self.feat_extractor_TP.reset_unet_k()

            else:  # 阶段 2：引入对比学习
                # 对比学习网络前向传播
                # save_negatives(self.im_negs_I, self.im_negs_TP, batch_index=0)
                logits_global_I, labels_global_I, z_I = self.feat_extractor_I(self.im_q, self.im_k, self.im_negs_I)
                
                logits_global_TP, labels_global_TP, z_TP = self.feat_extractor_TP(self.im_q, self.im_k, self.im_negs_TP)

                # 全局对比学习损失
                global_contra_loss_I = self.criterion_contra(logits_global_I, labels_global_I)
                loss_dict['l_global_contra_I'] = global_contra_loss_I
                l_total += global_contra_loss_I * self.cri_contra_weight_global * self.cri_contra_weight
                
                global_contra_loss_TP = self.criterion_contra(logits_global_TP, labels_global_TP)
                loss_dict['l_global_contra_TP'] = global_contra_loss_TP
                l_total += global_contra_loss_TP * self.cri_contra_weight_global * self.cri_contra_weight

                # # 局部对比学习损失
                # local_contra_loss_I = self.criterion_contra(logits_local_I, labels_local_I)
                # loss_dict['l_local_contra_I'] = local_contra_loss_I
                # l_total += local_contra_loss_I * self.cri_contra_weight_local * self.cri_contra_weight
                
                # local_contra_loss_TP = self.criterion_contra(logits_local_TP, labels_local_TP)
                # loss_dict['l_local_contra_TP'] = local_contra_loss_TP
                # l_total += local_contra_loss_TP * self.cri_contra_weight_local * self.cri_contra_weight


                # 超分辨率模型前向传播
                self.output = self.net_g(self.lq, z_I, z_TP)
                # self.output_mid = self.net_g.mid_result

                # pixel loss
                if self.cri_pix:
                    l_pix = self.cri_pix(self.output, self.gt)
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix

                # # intermediate supervision loss
                # if self.cri_pix_mid:
                #     l_pix_mid = self.cri_pix_mid(self.output_mid, self.gt)
                #     l_total += l_pix_mid
                #     loss_dict['l_pix_mid'] = l_pix_mid
                    
                # # 可视化部分：按log_interval频率保存图像
                # log_interval = train_opt.get('log_interval', 100)  # 默认每100次迭代保存一次
                # if current_iter % log_interval == 0:
                #     self.save_visualizations(current_iter, save_dir='saved_images_stage2')

                
        # 反向传播和优化
        if self.scaler is not None:
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.step(self.optimizer_feat_I)
            self.scaler.step(self.optimizer_feat_TP)
            self.scaler.update()
        else:
            # 否则直接普通 backward
            l_total.backward()
            self.optimizer_g.step()
            self.optimizer_feat_I.step()
            self.optimizer_feat_TP.step()



        # 保存日志
        self.log_dict = self.reduce_loss_dict(loss_dict)

        # 如果启用了EMA，则更新EMA模型权重
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
            # self.contrast_model_ema(decay=self.ema_decay)

    def test(self):
        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):  # (###### [FP16 CHANGE])
            # 使用EMA模型进行测试
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.feat_extractor_I.eval()
                self.feat_extractor_TP.eval()
                with torch.no_grad():
                    # 获取特征编码
                    z_I = self.feat_extractor_I.unet_k(self.im_q)
                    z_TP = self.feat_extractor_TP.unet_k(self.im_q)
                    # 使用特征编码进行推理
                    self.output = self.net_g_ema(self.lq, z_I, z_TP)
                    # self.output_mid = self.net_g_ema.mid_result
            else:
                # 使用原始模型和feat_extractor进行测试
                self.net_g.eval()
                self.feat_extractor_I.eval()
                self.feat_extractor_TP.eval()
                with torch.no_grad():
                    # 获取特征编码
                    z_I = self.feat_extractor_I.unet_k(self.im_q)
                    z_TP = self.feat_extractor_TP.unet_k(self.im_q)
                    # 使用特征编码进行推理
                    self.output = self.net_g(self.lq, z_I, z_TP)
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
            # self.save_batch_images(val_data, idx)
            self.feed_data_test(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = visuals['result']
            # sr_img_mid = visuals['mid_result']
            gt_img = visuals['gt'] if 'gt' in visuals else None

            sr_img = torch.clamp(sr_img, 0.0, 1.0)
            gt_img = torch.clamp(gt_img, 0.0, 1.0)
            
            if to_rgb:
                convert_to_rgb = HdrConvertBackToRGB(src=to_rgb, zero_one_normed=True)
                sr_img = convert_to_rgb(sr_img)
                # sr_img_mid = convert_to_rgb(sr_img_mid)
                gt_img = convert_to_rgb(gt_img) if gt_img is not None else None

            sr_img = tensor2img([sr_img], out_type=np.uint16)
            # sr_img_mid = tensor2img([sr_img_mid])
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
        # 构建要保存的网络和对应的键名列表
        nets_to_save = [self.net_g]
        param_keys = ['params']

        if hasattr(self, 'net_g_ema'):
            nets_to_save.append(self.net_g_ema)
            param_keys.append('params_ema')

        nets_to_save.append(self.feat_extractor_I)
        param_keys.append('feat_params_I')
        
        nets_to_save.append(self.feat_extractor_TP)
        param_keys.append('feat_params_TP')

        # if hasattr(self, 'feat_extractor_ema'):
        #     nets_to_save.append(self.feat_extractor_ema)
        #     param_keys.append('feat_params_ema')

        # 调用 save_network 并确保 nets_to_save 和 param_keys 长度匹配
        self.save_network(
            net=nets_to_save,
            net_label='net_g',
            current_iter=current_iter,
            param_key=param_keys
        )

        # 保存训练状态
        self.save_training_state(epoch, current_iter)
    
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        # out_dict['mid_result'] = self.output_mid.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

@MODEL_REGISTRY.register()
class ContrastYCBCRMidSupITMModel_woinit(ITMModel_copy):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # 定义网络
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
        
        # 加载预训练模型
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
        # 如果是训练模式，则初始化训练设置
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        
        # (###### [BF16 CHANGE]) 检查是否使用 BF16
        self.use_bf16 = train_opt.get('use_bf16', False)  # 在配置文件里加一个 use_bf16 = true/false
        if self.use_bf16:
            logger = get_root_logger()
            logger.info('Using bfloat16 mixed precision training ...')  # 修正日志信息
        
        # (###### [BF16 CHANGE]) 初始化 GradScaler
        self.scaler = GradScaler() if self.use_bf16 else None  # BF16 通常需要 GradScaler
        
        # 定义EMA衰减
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
            
            # if load_path is None:
            #     # 没有预训练，就执行自定义初始化
            #     if hasattr(self.net_g, 'init_weights'):
            #         self.net_g.init_weights()
            # else:
            #     logger.info(f'Will load pretrained from {load_path}, skip init_weights')
    
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()
            # if load_path_feat is not None:
            #     self.load_network(self.feat_extractor_ema, load_path_feat, self.opt['path'].get('strict_load_g', True), 'feat_params_ema')
            # else:
            #     self.contrast_model_ema(0)  # copy net_g weight
            # self.feat_extractor_ema.eval()
        
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
        # 原图
        self.lq = data['lq'].to(self.device)  # [B, 3, H, W]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        
        # im_q, im_k 也上GPU
        self.im_q = data['im_q'].to(self.device)
        self.im_k = data['im_k'].to(self.device)

        # 先对原图做一次 SDR->ICTCP (放到GPU计算)
        # 直接传递 [B, 3, H, W]，无需 unsqueeze 和 squeeze
        dim = 1  # 通道维度
        sdr_ictcp_base = SDR_to_YCbCr(self.lq, dim=dim)  # shape [B, 3, H, W], GPU

        # --------------- 亮度负样本 ---------------
        self.im_negs_I = None
        if data['im_negs_I_raw'] is not None:
            im_negs_I_list = []  # 存放替换后张量
            for i, neg_i_cpu in enumerate(data['im_negs_I_raw']):
                # 负样本上 GPU
                neg_i_gpu = neg_i_cpu.to(self.device)  # [B, 3, H, W] 或 [3, H, W]
                
                # 负样本转换 SDR->ICTCP
                neg_ictcp = SDR_to_YCbCr(neg_i_gpu, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                
                # 克隆
                sdr_ictcp_i = sdr_ictcp_base.clone()
                
                # 替换 I 通道 (index=0)
                sdr_ictcp_i[:, 0, :, :] = neg_ictcp[:, 0, :, :]
                
                # 回到 SDR
                replaced_sdr_i = YCbCr_to_SDR(sdr_ictcp_i, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                im_negs_I_list.append(replaced_sdr_i)
            
            if len(im_negs_I_list) > 0:
                self.im_negs_I = torch.stack(im_negs_I_list, dim=1)  # [N, B, 3, H, W] 或 [N, 3, H, W]
                # 如果 N=1 且 B=1，可以简化维度
                if self.im_negs_I.size(1) == 1:
                    self.im_negs_I = self.im_negs_I.squeeze(1)  # [N, 3, H, W] 或 [N, 3, H, W]
                self.im_negs_I = apply_random_flip(self.im_negs_I)
        
        # --------------- 色度负样本 ---------------
        self.im_negs_TP = None
        if data['im_negs_TP_raw'] is not None:
            im_negs_TP_list = []
            for i, neg_tp_cpu in enumerate(data['im_negs_TP_raw']):
                neg_tp_gpu = neg_tp_cpu.to(self.device)  # [B, 3, H, W] 或 [3, H, W]
                
                # 负样本转换 SDR->ICTCP
                neg_ictcp = SDR_to_YCbCr(neg_tp_gpu, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                
                # 克隆原图 ictcp
                sdr_ictcp_tp = sdr_ictcp_base.clone()
                
                # 替换 T,P 通道 (index=1,2)
                sdr_ictcp_tp[:, 1:3, :, :] = neg_ictcp[:, 1:3, :, :]
                
                # 回到 SDR
                replaced_sdr_tp = YCbCr_to_SDR(sdr_ictcp_tp, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                im_negs_TP_list.append(replaced_sdr_tp)
            
            if len(im_negs_TP_list) > 0:
                self.im_negs_TP = torch.stack(im_negs_TP_list, dim=1)  # [N, B, 3, H, W] 或 [N, 3, H, W]
                # 如果 N=1 且 B=1，可以简化维度
                if self.im_negs_TP.size(1) == 1:
                    self.im_negs_TP = self.im_negs_TP.squeeze(1)  # [N, 3, H, W] 或 [N, 3, H, W]
                self.im_negs_TP = apply_random_flip(self.im_negs_TP)

        # 这样 self.im_negs_I 和 self.im_negs_TP 就是 GPU 上处理好的负样本

    
    def feed_data_test(self, data):
        # 获取输入数据并放入设备
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        # 生成对比学习的正样本和负样本
        self.im_q = data['im_q'].to(self.device)
        # self.im_k = data['im_k'].to(self.device)
        # self.im_negs_I = data['im_negs_I'].to(self.device)
        # self.im_negs_TP = data['im_negs_TP'].to(self.device)
    
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
        
        # 定义 net_g 参数
        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} in net_g will not be optimized.')

        # 定义 feat_extractor 参数
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
        
        # 保存优化器到优化器列表
        self.optimizers.append(self.optimizer_g)
        self.optimizers.append(self.optimizer_feat_I)
        self.optimizers.append(self.optimizer_feat_TP)
    
    def save_visualizations(self, current_iter, save_dir='saved_images'):
        """
        保存 im_q、lq、mid_result、output 和 gt 作为图像。

        参数:
            current_iter (int): 当前迭代次数，用于文件命名。
            save_dir (str): 图像保存目录。
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 定义迭代后缀
        iter_suffix = f'iter_{current_iter}'
        
        # 选择要保存的样本索引（例如第一个样本）
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
        # 清除梯度
        self.optimizer_g.zero_grad()
        self.optimizer_feat_I.zero_grad()
        self.optimizer_feat_TP.zero_grad()
        
        # 初始化总损失
        l_total = 0
        loss_dict = OrderedDict()

        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            if current_iter <= train_opt['stage1_iters']:  # 阶段 1：仅更新主模型，不引入对比学习
                # 使用对比学习网络的 UNet 部分 (unet_q)
                z_I, y_I = self.feat_extractor_I.unet_q(self.im_q)   # in:(B, 1, H, W)
                z_TP, y_TP = self.feat_extractor_TP.unet_q(self.im_q) # in:(B, 2, H, W)

                # 超分辨率模型前向传播
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
                
                # # 可视化部分：按log_interval频率保存图像
                # log_interval = train_opt.get('log_interval', 1)  # 默认每100次迭代保存一次
                # if current_iter % log_interval == 0:
                #     self.save_visualizations(current_iter, save_dir='saved_images_stage1')
                
                self.feat_extractor_I._momentum_update_key_encoder()
                self.feat_extractor_TP._momentum_update_key_encoder()
                
                # self.feat_extractor_I.reset_unet_k()
                # self.feat_extractor_TP.reset_unet_k()

            else:  # 阶段 2：引入对比学习
                # 对比学习网络前向传播
                # save_negatives(self.im_negs_I, self.im_negs_TP, batch_index=0)
                logits_global_I, labels_global_I, logits_local_I, labels_local_I, z_I, y_I = self.feat_extractor_I(self.im_q, self.im_k, self.im_negs_I)
                
                logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP, z_TP, y_TP = self.feat_extractor_TP(self.im_q, self.im_k, self.im_negs_TP)

                # 全局对比学习损失
                global_contra_loss_I = self.criterion_contra(logits_global_I, labels_global_I)
                loss_dict['l_global_contra_I'] = global_contra_loss_I
                l_total += global_contra_loss_I * self.cri_contra_weight_global * self.cri_contra_weight
                
                global_contra_loss_TP = self.criterion_contra(logits_global_TP, labels_global_TP)
                loss_dict['l_global_contra_TP'] = global_contra_loss_TP
                l_total += global_contra_loss_TP * self.cri_contra_weight_global * self.cri_contra_weight

                # 局部对比学习损失
                local_contra_loss_I = self.criterion_contra(logits_local_I, labels_local_I)
                loss_dict['l_local_contra_I'] = local_contra_loss_I
                l_total += local_contra_loss_I * self.cri_contra_weight_local * self.cri_contra_weight
                
                local_contra_loss_TP = self.criterion_contra(logits_local_TP, labels_local_TP)
                loss_dict['l_local_contra_TP'] = local_contra_loss_TP
                l_total += local_contra_loss_TP * self.cri_contra_weight_local * self.cri_contra_weight


                # 超分辨率模型前向传播
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
                    
                # # 可视化部分：按log_interval频率保存图像
                # log_interval = train_opt.get('log_interval', 100)  # 默认每100次迭代保存一次
                # if current_iter % log_interval == 0:
                #     self.save_visualizations(current_iter, save_dir='saved_images_stage2')

                
        # 反向传播和优化
        if self.scaler is not None:
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.step(self.optimizer_feat_I)
            self.scaler.step(self.optimizer_feat_TP)
            self.scaler.update()
        else:
            # 否则直接普通 backward
            l_total.backward()
            self.optimizer_g.step()
            self.optimizer_feat_I.step()
            self.optimizer_feat_TP.step()

        # # 计算度量指标（metrics）
        # if self.opt['val'].get('metrics') is not None:
        #     if not hasattr(self, 'metric_results'):  # only execute in the first run
        #         self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        #     # 初始化 metric_data 并进行指标计算
        #     metric_data = {'img': self.output.detach().cpu().numpy(), 'img2': self.gt.detach().cpu().numpy()}
        #     for name, opt_ in self.opt['val']['metrics'].items():
        #         metric_val = calculate_metric(metric_data, opt_)
        #         self.metric_results[name] = metric_val / self.opt['datasets']['train']['batch_size_per_gpu']
        #         # 打印或记录每个度量指标
        #         loss_dict[f'{name}'] = metric_val
        
        # # 创建保存路径
        # save_dir = self.opt['path'].get('visualization', './visualization')
        # os.makedirs(save_dir, exist_ok=True)

        # # 遍历 batch 并保存每张图片
        # for i in range(self.output.size(0)):
        #     output_img = tensor2img([self.output[i]])  # 将 Tensor 转换为图像
        #     gt_img = tensor2img([self.gt[i]])  # 目标图像

        #     # 保存图像路径
        #     save_output_path = os.path.join(save_dir, f'output_{current_iter}_img_{i}.png')
        #     save_gt_path = os.path.join(save_dir, f'gt_{current_iter}_img_{i}.png')
            
        #     cv2.imwrite(save_output_path, output_img)
        #     cv2.imwrite(save_gt_path, gt_img)
                
        # # 反向传播和优化
        # l_total.backward()
        # self.optimizer_g.step()
        # self.optimizer_feat.step()
        # 使用 scaler 缩放损失，进行反向传播和优化
            # self.scaler.scale(l_total).backward()
            # self.scaler.step(self.optimizer_g)
            # self.scaler.step(self.optimizer_feat)
            # self.scaler.update()

        # 保存日志
        self.log_dict = self.reduce_loss_dict(loss_dict)

        # 如果启用了EMA，则更新EMA模型权重
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
            # self.contrast_model_ema(decay=self.ema_decay)

    def test(self):
        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):  # (###### [FP16 CHANGE])
            # 使用EMA模型进行测试
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.feat_extractor_I.eval()
                self.feat_extractor_TP.eval()
                with torch.no_grad():
                    # 获取特征编码
                    z_I, y_I = self.feat_extractor_I.unet_k(self.im_q)
                    z_TP, y_TP = self.feat_extractor_TP.unet_k(self.im_q)
                    # 使用特征编码进行推理
                    self.output = self.net_g_ema(self.lq, z_I, z_TP, y_I, y_TP)
                    self.output_mid = self.net_g_ema.mid_result
            else:
                # 使用原始模型和feat_extractor进行测试
                self.net_g.eval()
                self.feat_extractor_I.eval()
                self.feat_extractor_TP.eval()
                with torch.no_grad():
                    # 获取特征编码
                    z_I, y_I = self.feat_extractor_I.unet_k(self.im_q)
                    z_TP, y_TP = self.feat_extractor_TP.unet_k(self.im_q)
                    # 使用特征编码进行推理
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
        # 构建要保存的网络和对应的键名列表
        nets_to_save = [self.net_g]
        param_keys = ['params']

        if hasattr(self, 'net_g_ema'):
            nets_to_save.append(self.net_g_ema)
            param_keys.append('params_ema')

        nets_to_save.append(self.feat_extractor_I)
        param_keys.append('feat_params_I')
        
        nets_to_save.append(self.feat_extractor_TP)
        param_keys.append('feat_params_TP')

        # if hasattr(self, 'feat_extractor_ema'):
        #     nets_to_save.append(self.feat_extractor_ema)
        #     param_keys.append('feat_params_ema')

        # 调用 save_network 并确保 nets_to_save 和 param_keys 长度匹配
        self.save_network(
            net=nets_to_save,
            net_label='net_g',
            current_iter=current_iter,
            param_key=param_keys
        )

        # 保存训练状态
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
class ContrastYCBCRMidSupITMModel_copy(ITMModel_copy):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # 定义网络
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
        
        # 加载预训练模型
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
        # 如果是训练模式，则初始化训练设置
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        
        # (###### [BF16 CHANGE]) 检查是否使用 BF16
        self.use_bf16 = train_opt.get('use_bf16', False)  # 在配置文件里加一个 use_bf16 = true/false
        if self.use_bf16:
            logger = get_root_logger()
            logger.info('Using bfloat16 mixed precision training ...')  # 修正日志信息
        
        # (###### [BF16 CHANGE]) 初始化 GradScaler
        self.scaler = GradScaler() if self.use_bf16 else None  # BF16 通常需要 GradScaler
        
        # 定义EMA衰减
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
            
            # if load_path is None:
            #     # 没有预训练，就执行自定义初始化
            #     if hasattr(self.net_g, 'init_weights'):
            #         self.net_g.init_weights()
            # else:
            #     logger.info(f'Will load pretrained from {load_path}, skip init_weights')
    
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()
            # if load_path_feat is not None:
            #     self.load_network(self.feat_extractor_ema, load_path_feat, self.opt['path'].get('strict_load_g', True), 'feat_params_ema')
            # else:
            #     self.contrast_model_ema(0)  # copy net_g weight
            # self.feat_extractor_ema.eval()
        
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
        # 原图
        self.lq = data['lq'].to(self.device)  # [B, 3, H, W]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        
        # im_q, im_k 也上GPU
        self.im_q = data['im_q'].to(self.device)
        self.im_k = data['im_k'].to(self.device)

        # 先对原图做一次 SDR->ICTCP (放到GPU计算)
        # 直接传递 [B, 3, H, W]，无需 unsqueeze 和 squeeze
        dim = 1  # 通道维度
        sdr_ictcp_base = SDR_to_YCbCr(self.lq, dim=dim)  # shape [B, 3, H, W], GPU

        # --------------- 亮度负样本 ---------------
        self.im_negs_I = None
        if data['im_negs_I_raw'] is not None:
            im_negs_I_list = []  # 存放替换后张量
            for i, neg_i_cpu in enumerate(data['im_negs_I_raw']):
                # 负样本上 GPU
                neg_i_gpu = neg_i_cpu.to(self.device)  # [B, 3, H, W] 或 [3, H, W]
                
                # 负样本转换 SDR->ICTCP
                neg_ictcp = SDR_to_YCbCr(neg_i_gpu, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                
                # 克隆
                sdr_ictcp_i = sdr_ictcp_base.clone()
                
                # 替换 I 通道 (index=0)
                sdr_ictcp_i[:, 0, :, :] = neg_ictcp[:, 0, :, :]
                
                # 回到 SDR
                replaced_sdr_i = YCbCr_to_SDR(sdr_ictcp_i, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                im_negs_I_list.append(replaced_sdr_i)
            
            if len(im_negs_I_list) > 0:
                self.im_negs_I = torch.stack(im_negs_I_list, dim=1)  # [N, B, 3, H, W] 或 [N, 3, H, W]
                # 如果 N=1 且 B=1，可以简化维度
                if self.im_negs_I.size(1) == 1:
                    self.im_negs_I = self.im_negs_I.squeeze(1)  # [N, 3, H, W] 或 [N, 3, H, W]
                self.im_negs_I = apply_random_flip(self.im_negs_I)
        
        # --------------- 色度负样本 ---------------
        self.im_negs_TP = None
        if data['im_negs_TP_raw'] is not None:
            im_negs_TP_list = []
            for i, neg_tp_cpu in enumerate(data['im_negs_TP_raw']):
                neg_tp_gpu = neg_tp_cpu.to(self.device)  # [B, 3, H, W] 或 [3, H, W]
                
                # 负样本转换 SDR->ICTCP
                neg_ictcp = SDR_to_YCbCr(neg_tp_gpu, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                
                # 克隆原图 ictcp
                sdr_ictcp_tp = sdr_ictcp_base.clone()
                
                # 替换 T,P 通道 (index=1,2)
                sdr_ictcp_tp[:, 1:3, :, :] = neg_ictcp[:, 1:3, :, :]
                
                # 回到 SDR
                replaced_sdr_tp = YCbCr_to_SDR(sdr_ictcp_tp, dim=dim)  # [B, 3, H, W] 或 [3, H, W]
                im_negs_TP_list.append(replaced_sdr_tp)
            
            if len(im_negs_TP_list) > 0:
                self.im_negs_TP = torch.stack(im_negs_TP_list, dim=1)  # [N, B, 3, H, W] 或 [N, 3, H, W]
                # 如果 N=1 且 B=1，可以简化维度
                if self.im_negs_TP.size(1) == 1:
                    self.im_negs_TP = self.im_negs_TP.squeeze(1)  # [N, 3, H, W] 或 [N, 3, H, W]
                self.im_negs_TP = apply_random_flip(self.im_negs_TP)

        # 这样 self.im_negs_I 和 self.im_negs_TP 就是 GPU 上处理好的负样本

    
    def feed_data_test(self, data):
        # 获取输入数据并放入设备
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        # 生成对比学习的正样本和负样本
        self.im_q = data['im_q'].to(self.device)
        # self.im_k = data['im_k'].to(self.device)
        # self.im_negs_I = data['im_negs_I'].to(self.device)
        # self.im_negs_TP = data['im_negs_TP'].to(self.device)
    
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
        
        # 定义 net_g 参数
        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} in net_g will not be optimized.')

        # 定义 feat_extractor 参数
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
        
        # 保存优化器到优化器列表
        self.optimizers.append(self.optimizer_g)
        self.optimizers.append(self.optimizer_feat_I)
        self.optimizers.append(self.optimizer_feat_TP)
    
    def save_visualizations(self, current_iter, save_dir='saved_images'):
        """
        保存 im_q、lq、mid_result、output 和 gt 作为图像。

        参数:
            current_iter (int): 当前迭代次数，用于文件命名。
            save_dir (str): 图像保存目录。
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 定义迭代后缀
        iter_suffix = f'iter_{current_iter}'
        
        # 选择要保存的样本索引（例如第一个样本）
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
        # 清除梯度
        self.optimizer_g.zero_grad()
        self.optimizer_feat_I.zero_grad()
        self.optimizer_feat_TP.zero_grad()
        
        # 初始化总损失
        l_total = 0
        loss_dict = OrderedDict()

        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):
            if current_iter <= train_opt['stage1_iters']:  # 阶段 1：仅更新主模型，不引入对比学习
                # 使用对比学习网络的 UNet 部分 (unet_q)
                z_I, y_I = self.feat_extractor_I.unet_q(self.im_q)   # in:(B, 1, H, W)
                z_TP, y_TP = self.feat_extractor_TP.unet_q(self.im_q) # in:(B, 2, H, W)

                # 超分辨率模型前向传播
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
                
                # # 可视化部分：按log_interval频率保存图像
                # log_interval = train_opt.get('log_interval', 1)  # 默认每100次迭代保存一次
                # if current_iter % log_interval == 0:
                #     self.save_visualizations(current_iter, save_dir='saved_images_stage1')
                
                # self.feat_extractor_I._momentum_update_key_encoder()
                # self.feat_extractor_TP._momentum_update_key_encoder()
                
                # self.feat_extractor_I.reset_unet_k()
                # self.feat_extractor_TP.reset_unet_k()

            else:  # 阶段 2：引入对比学习
                # 对比学习网络前向传播
                # save_negatives(self.im_negs_I, self.im_negs_TP, batch_index=0)
                logits_global_I, labels_global_I, logits_local_I, labels_local_I, z_I, y_I = self.feat_extractor_I(self.im_q, self.im_k, self.im_negs_I)
                
                logits_global_TP, labels_global_TP, logits_local_TP, labels_local_TP, z_TP, y_TP = self.feat_extractor_TP(self.im_q, self.im_k, self.im_negs_TP)

                # 全局对比学习损失
                global_contra_loss_I = self.criterion_contra(logits_global_I, labels_global_I)
                loss_dict['l_global_contra_I'] = global_contra_loss_I
                l_total += global_contra_loss_I * self.cri_contra_weight_global * self.cri_contra_weight
                
                global_contra_loss_TP = self.criterion_contra(logits_global_TP, labels_global_TP)
                loss_dict['l_global_contra_TP'] = global_contra_loss_TP
                l_total += global_contra_loss_TP * self.cri_contra_weight_global * self.cri_contra_weight

                # 局部对比学习损失
                local_contra_loss_I = self.criterion_contra(logits_local_I, labels_local_I)
                loss_dict['l_local_contra_I'] = local_contra_loss_I
                l_total += local_contra_loss_I * self.cri_contra_weight_local * self.cri_contra_weight
                
                local_contra_loss_TP = self.criterion_contra(logits_local_TP, labels_local_TP)
                loss_dict['l_local_contra_TP'] = local_contra_loss_TP
                l_total += local_contra_loss_TP * self.cri_contra_weight_local * self.cri_contra_weight


                # 超分辨率模型前向传播
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
                    
                # # 可视化部分：按log_interval频率保存图像
                # log_interval = train_opt.get('log_interval', 100)  # 默认每100次迭代保存一次
                # if current_iter % log_interval == 0:
                #     self.save_visualizations(current_iter, save_dir='saved_images_stage2')

                
        # 反向传播和优化
        if self.scaler is not None:
            self.scaler.scale(l_total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.step(self.optimizer_feat_I)
            self.scaler.step(self.optimizer_feat_TP)
            self.scaler.update()
        else:
            # 否则直接普通 backward
            l_total.backward()
            self.optimizer_g.step()
            self.optimizer_feat_I.step()
            self.optimizer_feat_TP.step()

        # # 计算度量指标（metrics）
        # if self.opt['val'].get('metrics') is not None:
        #     if not hasattr(self, 'metric_results'):  # only execute in the first run
        #         self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        #     # 初始化 metric_data 并进行指标计算
        #     metric_data = {'img': self.output.detach().cpu().numpy(), 'img2': self.gt.detach().cpu().numpy()}
        #     for name, opt_ in self.opt['val']['metrics'].items():
        #         metric_val = calculate_metric(metric_data, opt_)
        #         self.metric_results[name] = metric_val / self.opt['datasets']['train']['batch_size_per_gpu']
        #         # 打印或记录每个度量指标
        #         loss_dict[f'{name}'] = metric_val
        
        # # 创建保存路径
        # save_dir = self.opt['path'].get('visualization', './visualization')
        # os.makedirs(save_dir, exist_ok=True)

        # # 遍历 batch 并保存每张图片
        # for i in range(self.output.size(0)):
        #     output_img = tensor2img([self.output[i]])  # 将 Tensor 转换为图像
        #     gt_img = tensor2img([self.gt[i]])  # 目标图像

        #     # 保存图像路径
        #     save_output_path = os.path.join(save_dir, f'output_{current_iter}_img_{i}.png')
        #     save_gt_path = os.path.join(save_dir, f'gt_{current_iter}_img_{i}.png')
            
        #     cv2.imwrite(save_output_path, output_img)
        #     cv2.imwrite(save_gt_path, gt_img)
                
        # # 反向传播和优化
        # l_total.backward()
        # self.optimizer_g.step()
        # self.optimizer_feat.step()
        # 使用 scaler 缩放损失，进行反向传播和优化
            # self.scaler.scale(l_total).backward()
            # self.scaler.step(self.optimizer_g)
            # self.scaler.step(self.optimizer_feat)
            # self.scaler.update()

        # 保存日志
        self.log_dict = self.reduce_loss_dict(loss_dict)

        # 如果启用了EMA，则更新EMA模型权重
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
            # self.contrast_model_ema(decay=self.ema_decay)

    def test(self):
        with autocast(enabled=self.use_bf16, dtype=torch.bfloat16):  # (###### [FP16 CHANGE])
            # 使用EMA模型进行测试
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.feat_extractor_I.eval()
                self.feat_extractor_TP.eval()
                with torch.no_grad():
                    # 获取特征编码
                    z_I, y_I = self.feat_extractor_I.unet_k(self.im_q)
                    z_TP, y_TP = self.feat_extractor_TP.unet_k(self.im_q)
                    # 使用特征编码进行推理
                    self.output = self.net_g_ema(self.lq, z_I, z_TP, y_I, y_TP)
                    self.output_mid = self.net_g_ema.mid_result
            else:
                # 使用原始模型和feat_extractor进行测试
                self.net_g.eval()
                self.feat_extractor_I.eval()
                self.feat_extractor_TP.eval()
                with torch.no_grad():
                    # 获取特征编码
                    z_I, y_I = self.feat_extractor_I.unet_k(self.im_q)
                    z_TP, y_TP = self.feat_extractor_TP.unet_k(self.im_q)
                    # 使用特征编码进行推理
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
        # 构建要保存的网络和对应的键名列表
        nets_to_save = [self.net_g]
        param_keys = ['params']

        if hasattr(self, 'net_g_ema'):
            nets_to_save.append(self.net_g_ema)
            param_keys.append('params_ema')

        nets_to_save.append(self.feat_extractor_I)
        param_keys.append('feat_params_I')
        
        nets_to_save.append(self.feat_extractor_TP)
        param_keys.append('feat_params_TP')

        # if hasattr(self, 'feat_extractor_ema'):
        #     nets_to_save.append(self.feat_extractor_ema)
        #     param_keys.append('feat_params_ema')

        # 调用 save_network 并确保 nets_to_save 和 param_keys 长度匹配
        self.save_network(
            net=nets_to_save,
            net_label='net_g',
            current_iter=current_iter,
            param_key=param_keys
        )

        # 保存训练状态
        self.save_training_state(epoch, current_iter)
    
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['mid_result'] = self.output_mid.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict