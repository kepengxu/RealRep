import lmdb,cv2
import numpy as np
import pickle,os
from basicsr.utils.registry import DATASET_REGISTRY
import torchvision.transforms as transforms
from basicsr.utils.ICTCP_convert import SDR_to_ICTCP, ICTCP_to_SDR

def testimage():
    testout = 'testout'

    env = lmdb.open('test.lmdb', map_size=1e12)
    txn = env.begin(write=False)

    with open('test.lmdb/names.pkl', 'rb') as f:
        loaded_data = pickle.load(f)


    def create_opencv_image_from_stringio(img_stream, cv2_img_flag=-1):

        img_array = np.asarray(bytearray(img_stream), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2_img_flag)

    for idx,keys in enumerate(loaded_data):
        if idx<1000:
            continue
        print(keys)
        for key in keys:
            image = create_opencv_image_from_stringio(txn.get(key.encode()))
            cv2.imwrite(os.path.join(testout,key+'.png'),image)
            print(key,image.shape)
    env.close()


import torch,cv2
import os
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from basicsr.utils import FileClient
import matplotlib.pyplot as plt

def save_lq_image_as_bgr(lq_tensor, output_path):
        # 将 `lq` Tensor 转换为 Numpy 数组并反转通道顺序
        lq_image = lq_tensor.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
        lq_image = (lq_image * 255).astype(np.uint8)
        
        # 使用 OpenCV 保存 BGR 格式的图像
        cv2.imwrite(output_path, lq_image)

def create_opencv_image_from_stringio(img_stream, cv2_img_flag=-1):

    img_array = np.asarray(bytearray(img_stream), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)

@DATASET_REGISTRY.register()
class HDRTVDMDATASET_RGBBGR_CONTRAST(Dataset):
    def __init__(self, opt):
        self.opt = opt
        with open(os.path.join(self.opt['root_dir'],'names.pkl'), 'rb') as f:
            self.namelist = pickle.load(f)
        self.sdrset = opt['sdrset']
        root_dir = self.opt['root_dir']
        self.io_backend_opt = opt['io_backend']
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = root_dir
            if not root_dir.endswith('lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
        self.file_client = None
        self.resize = transforms.Resize(size=(128, 128), antialias=True)

    def getmask(self,image,r = 0.75):
        mask = np.max(image, 2)
        mask = np.minimum(1.0, np.maximum(0, mask - r) / (1 - r))[:,:,np.newaxis]
        return torch.from_numpy(mask).permute(2,0,1).float()
    
    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, index):
        hname, sname = self.namelist[index][0],self.namelist[index][self.sdrset]
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        string_sdr = self.file_client.get(sname)
        string_hdr = self.file_client.get(hname)

        sdrimage=create_opencv_image_from_stringio(string_sdr)[:,:,::-1]/255.0
        hdrimage=create_opencv_image_from_stringio(string_hdr)[:,:,::-1]/65535.0
        # sdrimage=create_opencv_image_from_stringio(string_sdr)/255.0
        # hdrimage=create_opencv_image_from_stringio(string_hdr)/65535.0

        img_sdr = torch.from_numpy(np.ascontiguousarray(np.transpose(sdrimage, (2, 0, 1)))).float()
        img_hdr = torch.from_numpy(np.ascontiguousarray(np.transpose(hdrimage, (2, 0, 1)))).float()
        
        im_q = img_sdr #self.resize(img_sdr)
        
        # output_path = f"./visual/lq_bgr_image_{index}.png"
        # save_lq_image_as_bgr(img_sdr, output_path)

        return {
            'lq': img_sdr,
            'gt': img_hdr,
            # 'tiny': img_sdr,
            'im_q': im_q,
            # 'mask':self.getmask(hdrimage),
            # 'lqmask': self.getmask(sdrimage, 0.95),
            'gt_path':  self.opt['name']+'_{:0>8d}.png'.format(index),
            'lq_path':  self.opt['name']+'_{:0>8d}.png'.format(index)
        }

@DATASET_REGISTRY.register()
class HDRTVDMDATASET_BGRBGR_CONTRAST(Dataset):
    def __init__(self, opt):
        self.opt = opt
        with open(os.path.join(self.opt['root_dir'],'names.pkl'), 'rb') as f:
            self.namelist = pickle.load(f)
        self.sdrset = opt['sdrset']
        root_dir = self.opt['root_dir']
        self.io_backend_opt = opt['io_backend']
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = root_dir
            if not root_dir.endswith('lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
        self.file_client = None
        self.resize = transforms.Resize(size=(128, 128), antialias=True)

    def getmask(self,image,r = 0.75):
        mask = np.max(image, 2)
        mask = np.minimum(1.0, np.maximum(0, mask - r) / (1 - r))[:,:,np.newaxis]
        return torch.from_numpy(mask).permute(2,0,1).float()
        
    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, index):
        hname, sname = self.namelist[index][0],self.namelist[index][self.sdrset]
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        string_sdr = self.file_client.get(sname)
        string_hdr = self.file_client.get(hname)

        # sdrimage=create_opencv_image_from_stringio(string_sdr)[:,:,::-1]/255.0
        # hdrimage=create_opencv_image_from_stringio(string_hdr)[:,:,::-1]/65535.0
        sdrimage=create_opencv_image_from_stringio(string_sdr)/255.0
        hdrimage=create_opencv_image_from_stringio(string_hdr)/65535.0

        img_sdr = torch.from_numpy(np.ascontiguousarray(np.transpose(sdrimage, (2, 0, 1)))).float()
        img_hdr = torch.from_numpy(np.ascontiguousarray(np.transpose(hdrimage, (2, 0, 1)))).float()
        
        im_q = img_sdr # self.resize(img_sdr)
        
        # output_path = f"./visual/lq_bgr_image.png"
        # save_lq_image_as_bgr(img_sdr, output_path)
        
        return {
            'lq': img_sdr,
            'gt': img_hdr,
            'tiny': img_sdr,
            'im_q': im_q,
            # 'mask':self.getmask(hdrimage),
            # 'lqmask': self.getmask(sdrimage, 0.95),
            'gt_path':  self.opt['name']+'_{:0>8d}.png'.format(index),
            'lq_path':  self.opt['name']+'_{:0>8d}.png'.format(index)
        }

@DATASET_REGISTRY.register()
class HDRTVDMDATASET_RGB_CONTRAST(Dataset):
    def __init__(self, opt):
        self.opt = opt
        with open(os.path.join(self.opt['root_dir'],'names.pkl'), 'rb') as f:
            self.namelist = pickle.load(f)
        self.sdrset = opt['sdrset']
        root_dir = self.opt['root_dir']
        self.io_backend_opt = opt['io_backend']
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = root_dir
            if not root_dir.endswith('lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
        self.file_client = None
        self.resize = transforms.Resize(size=(128, 128), antialias=True)

    def getmask(self,image,r = 0.75):
        mask = np.max(image, 2)
        mask = np.minimum(1.0, np.maximum(0, mask - r) / (1 - r))[:,:,np.newaxis]
        return torch.from_numpy(mask).permute(2,0,1).float()
    
    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, index):
        hname, sname = self.namelist[index][0],self.namelist[index][self.sdrset]
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        string_sdr = self.file_client.get(sname)
        string_hdr = self.file_client.get(hname)

        # sdrimage=create_opencv_image_from_stringio(string_sdr)[:,:,::-1]/255.0
        # hdrimage=create_opencv_image_from_stringio(string_hdr)[:,:,::-1]/65535.0
        sdrimage=create_opencv_image_from_stringio(string_sdr)/255.0
        hdrimage=create_opencv_image_from_stringio(string_hdr)/65535.0

        img_sdr = torch.from_numpy(np.ascontiguousarray(np.transpose(sdrimage, (2, 0, 1)))).float()
        img_hdr = torch.from_numpy(np.ascontiguousarray(np.transpose(hdrimage, (2, 0, 1)))).float()
        
        im_q = img_sdr #self.resize(img_sdr)
        
        # output_path = f"./visual/lq_bgr_image_{index}.png"
        # save_lq_image_as_bgr(img_sdr, output_path)

        return {
            'lq': img_sdr,
            'gt': img_hdr,
            # 'tiny': img_sdr,
            'im_q': im_q,
            # 'mask':self.getmask(hdrimage),
            # 'lqmask': self.getmask(sdrimage, 0.95),
            'gt_path':  self.opt['name']+'_{:0>8d}.png'.format(index),
            'lq_path':  self.opt['name']+'_{:0>8d}.png'.format(index)
        }

@DATASET_REGISTRY.register()
class HDRTVDMDATASET_CONTRAST(Dataset):
    def __init__(self, opt):
        self.opt = opt
        with open(os.path.join(self.opt['root_dir'],'names.pkl'), 'rb') as f:
            self.namelist = pickle.load(f)
        self.sdrset = opt['sdrset']
        root_dir = self.opt['root_dir']
        self.io_backend_opt = opt['io_backend']
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = root_dir
            if not root_dir.endswith('lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
        self.file_client = None
        self.resize = transforms.Resize(size=(128, 128), antialias=True)

    def getmask(self,image,r = 0.75):
        mask = np.max(image, 2)
        mask = np.minimum(1.0, np.maximum(0, mask - r) / (1 - r))[:,:,np.newaxis]
        return torch.from_numpy(mask).permute(2,0,1).float()
        
    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, index):
        hname, sname = self.namelist[index][0],self.namelist[index][self.sdrset]
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        string_sdr = self.file_client.get(sname)
        string_hdr = self.file_client.get(hname)

        sdrimage=create_opencv_image_from_stringio(string_sdr)[:,:,::-1]/255.0
        hdrimage=create_opencv_image_from_stringio(string_hdr)[:,:,::-1]/65535.0
        # sdrimage=create_opencv_image_from_stringio(string_sdr)/255.0
        # hdrimage=create_opencv_image_from_stringio(string_hdr)/65535.0

        img_sdr = torch.from_numpy(np.ascontiguousarray(np.transpose(sdrimage, (2, 0, 1)))).float()
        img_hdr = torch.from_numpy(np.ascontiguousarray(np.transpose(hdrimage, (2, 0, 1)))).float()
        
        im_q = img_sdr # self.resize(img_sdr)
        
        # output_path = f"./visual/lq_bgr_image.png"
        # save_lq_image_as_bgr(img_sdr, output_path)
        
        return {
            'lq': img_sdr,
            'gt': img_hdr,
            'tiny': img_sdr,
            'im_q': im_q,
            # 'mask':self.getmask(hdrimage),
            # 'lqmask': self.getmask(sdrimage, 0.95),
            'gt_path':  self.opt['name']+'_{:0>8d}.png'.format(index),
            'lq_path':  self.opt['name']+'_{:0>8d}.png'.format(index)
        }

def load_negative_samples(txt_path):
    negative_samples_dict = {}
    with open(txt_path, 'r') as f:
        for line in f:
            name, negatives = line.strip().split(':')
            negatives = negatives.strip().split(', ')
            negative_samples_dict[name] = negatives
    return negative_samples_dict

def tensor2bgr_image(tensor_3hw):
    """
    将 [3,h,w], float, [0,1] -> [h,w,3], BGR, [0,255]
    """
    np_img = tensor_3hw.cpu().numpy()  # (3,h,w)
    np_img = np.transpose(np_img, (1,2,0))  # (h,w,3)
    np_img = np.clip(np_img, 0.0, 1.0) * 255.0
    np_img_bgr = np_img[..., ::-1].astype(np.uint8)  # RGB->BGR
    return np_img_bgr

def save_debug_image(tensor_3hw, prefix, sname, neg_index, output_dir):
    """
    将替换后得到的 tensor_3hw (shape [3,h,w], [0,1]) 转为 BGR 并保存。
      prefix:  文件名前缀, 如 "Ireplace" 或 "TPreplace"
      sname:   原始图路径(可取 basename)
      neg_index: 第几个负样本
      output_dir: 保存目录
    """
    # 1) 转BGR图
    debug_bgr = tensor2bgr_image(tensor_3hw)
    
    # 2) 构造保存名
    base_name = os.path.basename(sname)  # e.g. "xxx.png"
    save_name = f"{prefix}_{base_name}_neg_{neg_index}.png"
    save_path = os.path.join(output_dir, save_name)
    
    # 3) 保存
    cv2.imwrite(save_path, debug_bgr)

def _pick_negs(cand_list, rank, num, window, mode='slice'):
    if cand_list is None or len(cand_list) == 0:
        return []
    r0 = max(rank - 1, 0)

    # ensure we can take num
    max_start = max(len(cand_list) - num, 0)

    if window and window > 0:
        if mode == 'slice':
            shift = np.random.randint(0, window + 1)
            start = min(r0 + shift, max_start)
            return cand_list[start:start + num]
        elif mode == 'random':
            lo = r0
            hi = min(r0 + window + 1, len(cand_list))
            pool = cand_list[lo:hi]
            if len(pool) <= num:
                return pool
            idx = np.random.choice(len(pool), size=num, replace=False)
            return [pool[i] for i in idx.tolist()]
        else:
            start = min(r0, max_start)
            return cand_list[start:start + num]
    else:
        start = min(r0, max_start)
        return cand_list[start:start + num]

def _pick_negs_with_rank(cand_list, rank, num, window, mode='slice'):
    """
    cand_list: fullrank list (strings)
    rank: 1-based semi-hard rank (2 means start from index 1)
    returns: (picked_paths, picked_ranks) where picked_ranks are 0-based indices in cand_list
    """
    if cand_list is None or len(cand_list) == 0:
        return [], []

    r0 = max(rank - 1, 0)  # convert to 0-based start
    max_start = max(len(cand_list) - num, 0)

    if window and window > 0:
        if mode == 'slice':
            shift = np.random.randint(0, window + 1)
            start = min(r0 + shift, max_start)
            picks = cand_list[start:start + num]
            ranks = list(range(start, start + len(picks)))
            return picks, ranks

        elif mode == 'random':
            lo = r0
            hi = min(r0 + window + 1, len(cand_list))
            pool = cand_list[lo:hi]
            pool_r = list(range(lo, hi))
            if len(pool) <= num:
                return pool, pool_r
            idx = np.random.choice(len(pool), size=num, replace=False)
            picks = [pool[i] for i in idx.tolist()]
            ranks = [pool_r[i] for i in idx.tolist()]
            return picks, ranks

    # no window
    start = min(r0, max_start)
    picks = cand_list[start:start + num]
    ranks = list(range(start, start + len(picks)))
    return picks, ranks
    
def _dedup_negs_by_path(*neg_groups):
    seen = set()
    deduped_groups = []
    for paths, ranks in neg_groups:
        deduped_paths = []
        deduped_ranks = []
        for path, rank in zip(paths, ranks):
            if path in seen:
                continue
            seen.add(path)
            deduped_paths.append(path)
            deduped_ranks.append(rank)
        deduped_groups.append((deduped_paths, deduped_ranks))
    return deduped_groups


def collate_ranked_contrast_batch(batch):
    if len(batch) == 0:
        return batch

    out = {}
    keys = batch[0].keys()
    neg_keys = {'im_negs_I_raw', 'im_negs_TP_raw'}
    rank_keys = {'im_negs_I_rank', 'im_negs_TP_rank'}

    for key in keys:
        values = [sample[key] for sample in batch]
        if key in neg_keys:
            min_len = min(len(v) if v is not None else 0 for v in values)
            out[key] = [torch.stack([v[i] for v in values], dim=0) for i in range(min_len)]
        elif key in rank_keys:
            min_len = min(v.numel() if v is not None else 0 for v in values)
            if min_len == 0:
                out[key] = torch.empty(len(values), 0, dtype=torch.long)
            else:
                out[key] = torch.stack([v[:min_len] for v in values], dim=0)
        else:
            out[key] = torch.utils.data.default_collate(values)
    return out


@DATASET_REGISTRY.register()
class HDRTVDMDATASET1_RGB_CONTRAST_RANK(Dataset):
    def __init__(self, opt):
        self.opt = opt
        with open(os.path.join(self.opt['root_dir'],'names.pkl'), 'rb') as f:
            self.namelist = pickle.load(f)
        self.sdrset = opt['sdrset']
        root_dir = self.opt['root_dir']
        self.io_backend_opt = opt['io_backend']
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = root_dir
            if not root_dir.endswith('lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
        self.file_client = None
        self.resize = transforms.Resize(size=(128, 128), antialias=True)
        self.k_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        luma_path = os.path.join(self.opt['root_dir'], 'negatives_luma_fullrank.txt')
        if not os.path.isfile(luma_path):
            luma_path = os.path.join(self.opt['root_dir'], 'negatives_luma.txt')

        chroma_path = os.path.join(self.opt['root_dir'], 'negatives_chroma_fullrank.txt')
        if not os.path.isfile(chroma_path):
            chroma_path = os.path.join(self.opt['root_dir'], 'negatives_chroma.txt')

        self.negative_samples_dict_I = load_negative_samples(luma_path)
        self.negative_samples_dict_TP = load_negative_samples(chroma_path)

        self.neg_rank_luma = opt.get('neg_rank_luma', 2)
        self.neg_num_luma = opt.get('neg_num_luma', 4)
        self.neg_rank_window_luma = opt.get('neg_rank_window_luma', 0)

        self.neg_rank_chroma = opt.get('neg_rank_chroma', 2)
        self.neg_num_chroma = opt.get('neg_num_chroma', 3)
        self.neg_rank_window_chroma = opt.get('neg_rank_window_chroma', 2)

        self.neg_sample_mode = opt.get('neg_sample_mode', 'slice')
        self.dedup_negatives = opt.get('dedup_negatives', False)

        self.debug_vis_dir = os.path.join(self.opt['root_dir'], "debug_vis")
        os.makedirs(self.debug_vis_dir, exist_ok=True)

    def save_images(self, img_array, title, output_dir, index, sdrset):
        if not isinstance(img_array, np.ndarray):
            img_array = img_array.numpy()
        img_array = img_array.transpose(1, 2, 0)
        img_array = img_array[..., [2, 1, 0]]
        img_array = np.clip(img_array, 0, 1)
        filename = f"{title}_Index_{index}_SDRSet_{sdrset}.png"
        plt.figure(figsize=(12, 4))
        plt.imshow(img_array)
        plt.axis('off')
        plt.suptitle(f"{title} - Index {index} - SDRSet {sdrset}")
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    def getmask(self,image,r = 0.75):
        mask = np.max(image, 2)
        mask = np.minimum(1.0, np.maximum(0, mask - r) / (1 - r))[:,:,np.newaxis]
        return torch.from_numpy(mask).permute(2,0,1).float()

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, index):
        hname, sname, tiny_name = self.namelist[index][0], self.namelist[index][self.sdrset], self.namelist[index][-1]
        if 'SDR' not in tiny_name or not ('TINY' in tiny_name or 'Tiny' in tiny_name):
            raise ValueError(f"The tiny_name '{tiny_name}' does not contain required substrings 'SDR' and 'TINY'/'Tiny'. Program stopped.")
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        string_sdr = self.file_client.get(sname)
        string_hdr = self.file_client.get(hname)
        sdrimage = create_opencv_image_from_stringio(string_sdr) / 255.0
        hdrimage = create_opencv_image_from_stringio(string_hdr) / 65535.0
        img_sdr = torch.from_numpy(np.ascontiguousarray(np.transpose(sdrimage, (2, 0, 1)))).float()
        img_hdr = torch.from_numpy(np.ascontiguousarray(np.transpose(hdrimage, (2, 0, 1)))).float()

        im_q = img_sdr
        im_k = self.k_transform(im_q)

        negative_names_I = []
        ranks_I = []
        if sname in self.negative_samples_dict_I:
            cand_I = self.negative_samples_dict_I[sname]
            negative_names_I, ranks_I = _pick_negs_with_rank(
                cand_I, self.neg_rank_luma, self.neg_num_luma,
                self.neg_rank_window_luma, self.neg_sample_mode
            )

        negative_names_TP = []
        ranks_TP = []
        if sname in self.negative_samples_dict_TP:
            cand_TP = self.negative_samples_dict_TP[sname]
            negative_names_TP, ranks_TP = _pick_negs_with_rank(
                cand_TP, self.neg_rank_chroma, self.neg_num_chroma,
                self.neg_rank_window_chroma, self.neg_sample_mode
            )

        if self.dedup_negatives:
            (negative_names_I, ranks_I), (negative_names_TP, ranks_TP) = _dedup_negs_by_path(
                (negative_names_I, ranks_I),
                (negative_names_TP, ranks_TP),
            )

        im_negs_I_raw = []
        for p in negative_names_I:
            neg_bytes = self.file_client.get(p)
            neg_rgb = create_opencv_image_from_stringio(neg_bytes) / 255.0
            neg_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(neg_rgb, (2, 0, 1)))).float()
            im_negs_I_raw.append(neg_tensor)
        if len(im_negs_I_raw) == 0:
            im_negs_I_raw = None
            im_negs_I_rank = None
        else:
            im_negs_I_rank = torch.tensor(ranks_I, dtype=torch.long)

        im_negs_TP_raw = []
        for p in negative_names_TP:
            neg_bytes = self.file_client.get(p)
            neg_rgb = create_opencv_image_from_stringio(neg_bytes) / 255.0
            neg_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(neg_rgb, (2, 0, 1)))).float()
            im_negs_TP_raw.append(neg_tensor)
        if len(im_negs_TP_raw) == 0:
            im_negs_TP_raw = None
            im_negs_TP_rank = None
        else:
            im_negs_TP_rank = torch.tensor(ranks_TP, dtype=torch.long)

        return {
            'lq': img_sdr,
            'gt': img_hdr,
            'im_q': im_q,
            'im_k': im_k,
            'im_negs_I_raw': im_negs_I_raw,
            'im_negs_TP_raw': im_negs_TP_raw,
            'im_negs_I_rank': im_negs_I_rank,
            'im_negs_TP_rank': im_negs_TP_rank,
            'gt_path': self.opt['name'] + '_{:0>8d}.png'.format(index),
            'lq_path': self.opt['name'] + '_{:0>8d}.png'.format(index)
        }


@DATASET_REGISTRY.register()
class HDRTVDMDATASET1_RGB_CONTRAST(Dataset):
    def __init__(self, opt):
        self.opt = opt
        with open(os.path.join(self.opt['root_dir'],'names.pkl'), 'rb') as f:
            self.namelist = pickle.load(f)
        self.sdrset = opt['sdrset']
        root_dir = self.opt['root_dir']
        self.io_backend_opt = opt['io_backend']
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = root_dir
            if not root_dir.endswith('lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
        self.file_client = None
        self.resize = transforms.Resize(size=(128, 128), antialias=True)
        self.k_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        self.negative_samples_dict_I = load_negative_samples(os.path.join(self.opt['root_dir'],'negatives_luma.txt'))
        self.negative_samples_dict_TP = load_negative_samples(os.path.join(self.opt['root_dir'],'negatives_chroma.txt'))
        self.neg_num_luma = opt.get('neg_num_luma', 5)
        self.neg_num_chroma = opt.get('neg_num_chroma', 3)

        self.debug_vis_dir = os.path.join(self.opt['root_dir'], "debug_vis")
        os.makedirs(self.debug_vis_dir, exist_ok=True)

    def save_images(self, img_array, title, output_dir, index, sdrset):
        if not isinstance(img_array, np.ndarray):
            img_array = img_array.numpy()
        img_array = img_array.transpose(1, 2, 0)
        img_array = img_array[..., [2, 1, 0]]
        img_array = np.clip(img_array, 0, 1)
        filename = f"{title}_Index_{index}_SDRSet_{sdrset}.png"
        plt.figure(figsize=(12, 4))
        plt.imshow(img_array)
        plt.axis('off')
        plt.suptitle(f"{title} - Index {index} - SDRSet {sdrset}")
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    def getmask(self,image,r = 0.75):
        mask = np.max(image, 2)
        mask = np.minimum(1.0, np.maximum(0, mask - r) / (1 - r))[:,:,np.newaxis]
        return torch.from_numpy(mask).permute(2,0,1).float()

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, index):
        hname, sname, tiny_name = self.namelist[index][0], self.namelist[index][self.sdrset], self.namelist[index][-1]
        if 'SDR' not in tiny_name or not ('TINY' in tiny_name or 'Tiny' in tiny_name):
            raise ValueError(f"The tiny_name '{tiny_name}' does not contain required substrings 'SDR' and 'TINY'/'Tiny'. Program stopped.")
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        string_sdr = self.file_client.get(sname)
        string_hdr = self.file_client.get(hname)
        sdrimage = create_opencv_image_from_stringio(string_sdr) / 255.0
        hdrimage = create_opencv_image_from_stringio(string_hdr) / 65535.0
        img_sdr = torch.from_numpy(np.ascontiguousarray(np.transpose(sdrimage, (2, 0, 1)))).float()
        img_hdr = torch.from_numpy(np.ascontiguousarray(np.transpose(hdrimage, (2, 0, 1)))).float()

        im_q = img_sdr
        im_k = self.k_transform(im_q)

        im_negs_I_raw = []
        if sname in self.negative_samples_dict_I:
            negative_names_I = self.negative_samples_dict_I[sname][:self.neg_num_luma]
            for neg_i_path in negative_names_I:
                neg_i_bytes = self.file_client.get(neg_i_path)
                neg_i_rgb = create_opencv_image_from_stringio(neg_i_bytes)/255.0
                neg_i_tensor = torch.from_numpy(
                    np.ascontiguousarray(np.transpose(neg_i_rgb,(2,0,1)))
                ).float()
                im_negs_I_raw.append(neg_i_tensor)
        if len(im_negs_I_raw) == 0:
            im_negs_I_raw = None

        im_negs_TP_raw = []
        if sname in self.negative_samples_dict_TP:
            negative_names_TP = self.negative_samples_dict_TP[sname][:self.neg_num_chroma]
            for neg_tp_path in negative_names_TP:
                neg_tp_bytes = self.file_client.get(neg_tp_path)
                neg_tp_rgb = create_opencv_image_from_stringio(neg_tp_bytes)/255.0
                neg_tp_tensor = torch.from_numpy(
                    np.ascontiguousarray(np.transpose(neg_tp_rgb,(2,0,1)))
                ).float()
                im_negs_TP_raw.append(neg_tp_tensor)
        if len(im_negs_TP_raw) == 0:
            im_negs_TP_raw = None

        return {
            'lq': img_sdr,
            'gt': img_hdr,
            'im_q': im_q,
            'im_k': im_k,
            'im_negs_I_raw': im_negs_I_raw,
            'im_negs_TP_raw': im_negs_TP_raw,
            'gt_path': self.opt['name'] + '_{:0>8d}.png'.format(index),
            'lq_path': self.opt['name'] + '_{:0>8d}.png'.format(index)
        }

    

@DATASET_REGISTRY.register()
class HDRTVDMDATASET1_CONTRAST(Dataset):
    def __init__(self, opt):
        self.opt = opt
        with open(os.path.join(self.opt['root_dir'],'names.pkl'), 'rb') as f:
            self.namelist = pickle.load(f)
        self.sdrset = opt['sdrset']
        root_dir = self.opt['root_dir']
        self.io_backend_opt = opt['io_backend']
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = root_dir
            if not root_dir.endswith('lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
        self.file_client = None

    def save_images(self, img_array, title, output_dir, index):
        if not isinstance(img_array, np.ndarray):
            img_array = img_array.numpy()  # Convert tensor to numpy array
        
        # Convert to HWC format
        img_array = img_array.transpose(1, 2, 0)  # [c, h, w] to [h, w, c]

        # Clip values to [0, 1]
        img_array = np.clip(img_array, 0, 1)
        
        plt.figure(figsize=(12, 4))
        plt.imshow(img_array)
        plt.axis('off')
        plt.suptitle(f"{title} - Index {index}")
        plt.savefig(f"{output_dir}/{title}_Index_{index}.png")
        plt.close()

                
    def getmask(self,image,r = 0.75):
        mask = np.max(image, 2)
        mask = np.minimum(1.0, np.maximum(0, mask - r) / (1 - r))[:,:,np.newaxis]
        return torch.from_numpy(mask).permute(2,0,1).float()

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, index):        
        hname, sname, tiny_name = self.namelist[index][0], self.namelist[index][self.sdrset], self.namelist[index][-1]
        if 'SDR' not in tiny_name or not ('TINY' in tiny_name or 'Tiny' in tiny_name):
            raise ValueError(f"The tiny_name '{tiny_name}' does not contain required substrings 'SDR' and 'TINY'/'Tiny'. Program stopped.")
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        string_sdr = self.file_client.get(sname)
        string_hdr = self.file_client.get(hname)
        string_tiny = self.file_client.get(tiny_name)

        sdrimage = create_opencv_image_from_stringio(string_sdr) / 255.0
        hdrimage = create_opencv_image_from_stringio(string_hdr) / 65535.0
        tinyimage = create_opencv_image_from_stringio(string_tiny) / 255.0

        img_sdr = torch.from_numpy(np.ascontiguousarray(np.transpose(sdrimage, (2, 0, 1)))).float()
        img_hdr = torch.from_numpy(np.ascontiguousarray(np.transpose(hdrimage, (2, 0, 1)))).float()
        img_tiny = torch.from_numpy(np.ascontiguousarray(np.transpose(tinyimage, (2, 0, 1)))).float()
        
        output_dir = "train_view"  # Replace with your desired output directory
        # self.save_images(img_sdr, "Cropped_SDR", output_dir, index)
        # self.save_images(img_hdr, "Cropped_HDR", output_dir, index)
        # self.save_images(img_tiny, "tiny_SDR", output_dir, index)
        # output_path = f"./visual/lq_bgr_image.png"
        # save_lq_image_as_bgr(img_sdr, output_path)
        return {
            'lq': img_sdr,
            'gt': img_hdr,
            # 'mask': self.getmask(hdrimage),
            # 'lqmask': self.getmask(sdrimage, 0.95),
            # 'tiny': img_tiny,
            'gt_path': self.opt['name'] + '_{:0>8d}.png'.format(index),
            'lq_path': self.opt['name'] + '_{:0>8d}.png'.format(index)
        }



@DATASET_REGISTRY.register()
class HDRTVDMDATASETMIX_CONTRAST(Dataset):
    def __init__(self, opt):
        self.datasets = []
        self.opt = opt
        self.lent = 0
        self.numdata = 0
        self.namelistd = []  # 存储每个HDRTVDMDATASET1的样本数量
        for key in opt['dataset'].keys():
            opt1 = opt['dataset'][key]
            if 'hdrtvdm' in opt1['name']:
                self.datasets.append(HDRTVDMDATASET1_RGB_CONTRAST(opt1))
            else:
                self.datasets.append(HDRTVDMDATASET1_CONTRAST(opt1))
            self.lent += self.datasets[-1].__len__()
            for step, ndn in enumerate(self.datasets[-1].namelist):
                self.namelistd.append([self.numdata, step])
            print(f"Dataset: {key}")  # 打印数据集名称
            self.numdata += 1
        self.num_datasets = len(self.datasets)  # 总数据集数量

    def __len__(self):
        return self.lent

    def __getitem__(self, index):
        namelistdd = self.namelistd[index]
        dataset_index = namelistdd[0]
        sample_index = namelistdd[1]

        # 获取对应的数据集和样本
        sample = self.datasets[dataset_index].__getitem__(sample_index)

        return sample


@DATASET_REGISTRY.register()
class HDRTVDMDATASET1_UNIFIEDYXY_RGB_CONTRAST(HDRTVDMDATASET1_RGB_CONTRAST_RANK):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        return {
            'lq': sample['lq'],
            'gt': sample['gt'],
            'im_q': sample['im_q'],
            'im_k': sample['im_k'],
            'im_negs_I_raw': sample.get('im_negs_I_raw', None),
            'im_negs_TP_raw': sample.get('im_negs_TP_raw', None),
            'im_negs_I_rank': sample.get('im_negs_I_rank', None),
            'im_negs_TP_rank': sample.get('im_negs_TP_rank', None),
            'gt_path': sample['gt_path'],
            'lq_path': sample['lq_path']
        }


@DATASET_REGISTRY.register()
class HDRTVDMDATASETMIX_UNIFIEDYXY_CONTRAST(Dataset):
    def __init__(self, opt):
        self.datasets = []
        self.opt = opt
        self.collate_fn = collate_ranked_contrast_batch if opt.get('collate_ranked_contrast', False) else None
        self.lent = 0
        self.numdata = 0
        self.namelistd = []
        for key in opt['dataset'].keys():
            opt1 = opt['dataset'][key]
            self.datasets.append(HDRTVDMDATASET1_UNIFIEDYXY_RGB_CONTRAST(opt1))
            self.lent += self.datasets[-1].__len__()
            for step, _ in enumerate(self.datasets[-1].namelist):
                self.namelistd.append([self.numdata, step])
            print(f"Dataset: {key}")
            self.numdata += 1
        self.num_datasets = len(self.datasets)

    def __len__(self):
        return self.lent

    def __getitem__(self, index):
        dataset_index, sample_index = self.namelistd[index]
        return self.datasets[dataset_index].__getitem__(sample_index)


import os,torch,sys


sys.path.append(os.getcwd()+'/')



from collections import OrderedDict
def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper
