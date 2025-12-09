import lmdb,cv2
import numpy as np
import pickle,os
from basicsr.utils.registry import DATASET_REGISTRY

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


def create_opencv_image_from_stringio(img_stream, cv2_img_flag=-1):

    img_array = np.asarray(bytearray(img_stream), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)

@DATASET_REGISTRY.register()
class HDRTVDMDATASET_RGB(Dataset):
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

        img_sdr = torch.from_numpy(np.ascontiguousarray(np.transpose(sdrimage, (2, 0, 1)))).float()
        img_hdr = torch.from_numpy(np.ascontiguousarray(np.transpose(hdrimage, (2, 0, 1)))).float()

        return {
            'lq': img_sdr,
            'gt': img_hdr,
            # 'tiny': img_sdr,
            # 'mask':self.getmask(hdrimage),
            # 'lqmask': self.getmask(sdrimage, 0.95),
            'gt_path':  self.opt['name']+'_{:0>8d}.png'.format(index),
            'lq_path':  self.opt['name']+'_{:0>8d}.png'.format(index)
        }

@DATASET_REGISTRY.register()
class HDRTVDMDATASET(Dataset):
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

        sdrimage=create_opencv_image_from_stringio(string_sdr)/255.0
        hdrimage=create_opencv_image_from_stringio(string_hdr)/65535.0

        img_sdr = torch.from_numpy(np.ascontiguousarray(np.transpose(sdrimage, (2, 0, 1)))).float()
        img_hdr = torch.from_numpy(np.ascontiguousarray(np.transpose(hdrimage, (2, 0, 1)))).float()

        return {
            'lq': img_sdr,
            'gt': img_hdr,
            # 'tiny': img_sdr,
            # 'mask':self.getmask(hdrimage),
            # 'lqmask': self.getmask(sdrimage, 0.95),
            'gt_path':  self.opt['name']+'_{:0>8d}.png'.format(index),
            'lq_path':  self.opt['name']+'_{:0>8d}.png'.format(index)
        }

from basicsr.data.transforms import augment, paired_random_crop
@DATASET_REGISTRY.register()
class HDRTVDMDATASET1_RGB(Dataset):
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
        
        # Convert RGB to BGR
        img_array = img_array[..., [2, 1, 0]]
        
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

        sdrimage = create_opencv_image_from_stringio(string_sdr)[:, :, ::-1] / 255.0
        hdrimage = create_opencv_image_from_stringio(string_hdr)[:, :, ::-1] / 65535.0
        tinyimage = create_opencv_image_from_stringio(string_tiny)[:, :, ::-1] / 255.0
        
        # augmentation for training

        # gt_size = self.opt['gt_size']
        # # random crop
        # if gt_size:
        #     img_hdr, img_sdr = paired_random_crop(img_hdr, img_sdr, gt_size, 1)

        img_sdr = torch.from_numpy(np.ascontiguousarray(np.transpose(sdrimage, (2, 0, 1)))).float()
        img_hdr = torch.from_numpy(np.ascontiguousarray(np.transpose(hdrimage, (2, 0, 1)))).float()
        # img_tiny = torch.from_numpy(np.ascontiguousarray(np.transpose(tinyimage, (2, 0, 1)))).float()

        # output_dir = "train_view"  # Replace with your desired output directory
        # self.save_images(img_sdr, "Cropped_SDR", output_dir, index)
        # self.save_images(img_hdr, "Cropped_HDR", output_dir, index)
        # self.save_images(img_tiny, "tiny_SDR", output_dir, index)
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
class HDRTVDMDATASET1(Dataset):
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

        # output_dir = "train_view"  # Replace with your desired output directory
        # self.save_images(img_sdr, "Cropped_SDR", output_dir, index)
        # self.save_images(img_hdr, "Cropped_HDR", output_dir, index)
        # self.save_images(img_tiny, "tiny_SDR", output_dir, index)
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
class HDRTVDMDATASETMIX(Dataset):
    def __init__(self, opt):
        self.datasets = []
        self.opt = opt
        self.lent = 0
        self.numdata = 0
        self.namelistd = []
        for key in opt['dataset'].keys():
            opt1 = opt['dataset'][key]
            if 'hdrtvdm' in opt1['name']:
                self.datasets.append(HDRTVDMDATASET1_RGB(opt1))
            else:
                self.datasets.append(HDRTVDMDATASET1(opt1))
            self.lent += self.datasets[-1].__len__()
            for step, ndn in enumerate(self.datasets[-1].namelist):
                self.namelistd.append([self.numdata, step])
            print(f"Dataset: {key}")  # 打印数据集名称
            self.numdata += 1

    def __len__(self):
        return self.lent

    def __getitem__(self, index):
        namelistdd = self.namelistd[index]
        dataset_index = namelistdd[0]
        sample_index = namelistdd[1]
        
        sample = self.datasets[dataset_index].__getitem__(sample_index)

        return sample


class HDRTVDMDATASET1_RGB_SIMCONTRAST(Dataset):
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
        
        self.debug_vis_dir = os.path.join(self.opt['root_dir'], "debug_vis")
        os.makedirs(self.debug_vis_dir, exist_ok=True)
    
    def save_images(self, img_array, title, output_dir, index, sdrset):
        if not isinstance(img_array, np.ndarray):
            img_array = img_array.numpy()  # Convert tensor to numpy array
        
        # Convert to HWC format
        img_array = img_array.transpose(1, 2, 0)  # [c, h, w] to [h, w, c]
        
        # Convert RGB to BGR
        img_array = img_array[..., [2, 1, 0]]
        
        # Clip values to [0, 1]
        img_array = np.clip(img_array, 0, 1)
        
        # Create unique filename by adding sdrset attribute
        filename = f"{title}_Index_{index}_SDRSet_{sdrset}.png"
        
        # Save the image
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

        next_index = (index + 1) % len(self.namelist)  
        next_sname = self.namelist[next_index][self.sdrset]

        next_string_sdr = self.file_client.get(next_sname)
        next_sdrimage = create_opencv_image_from_stringio(next_string_sdr) / 255.0
        im_k = torch.from_numpy(np.ascontiguousarray(np.transpose(next_sdrimage, (2, 0, 1)))).float()  # 作为 im_k


        return {
            'lq': img_sdr,          # CPU tensor
            'gt': img_hdr,          # CPU tensor
            'im_k': im_k,           # CPU tensor
            'gt_path': self.opt['name'] + '_{:0>8d}.png'.format(index),
            'lq_path': self.opt['name'] + '_{:0>8d}.png'.format(index)
        }


@DATASET_REGISTRY.register()
class HDRTVDMDATASETMIX_SIMCONTRAST(Dataset):
    def __init__(self, opt):
        self.datasets = []
        self.opt = opt
        self.lent = 0
        self.numdata = 0
        self.namelistd = []  
        for key in opt['dataset'].keys():
            opt1 = opt['dataset'][key]
            if 'hdrtvdm' in opt1['name']:
                self.datasets.append(HDRTVDMDATASET1_RGB_SIMCONTRAST(opt1))
            else:
                self.datasets.append(HDRTVDMDATASET1(opt1))
            self.lent += self.datasets[-1].__len__()
            for step, ndn in enumerate(self.datasets[-1].namelist):
                self.namelistd.append([self.numdata, step])
            print(f"Dataset: {key}")  
            self.numdata += 1
        self.num_datasets = len(self.datasets)  
    def __len__(self):
        return self.lent

    def __getitem__(self, index):
        namelistdd = self.namelistd[index]
        dataset_index = namelistdd[0]
        sample_index = namelistdd[1]
        
        sample = self.datasets[dataset_index].__getitem__(sample_index)

        return sample



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



if __name__ == '__main__':
    import yaml
    with open('/data/disk1/xukepeng/project/diffhdr/options/trainmix/HDRFormer/train_s1_hdrrqvae_mix_100_maskmaev2.yml', mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    print(opt)
    opt1 = opt['datasets']['train']
    dataset = HDRTVDMDATASETMIX(opt1)
    for i in range(40000):
        if i%1000==0:
            data = dataset.__getitem__(i)
            print(i,data['gt'].min(),data['gt'].max(),data['lq'].min(),data['lq'].max())
            cv2.imwrite(os.path.join('/data/disk1/xukepeng/project/diffhdr/tests/testimg/outs',str(i)+'lq.png'),np.array(data['lq'].permute(1,2,0).numpy()*65535,np.uint16))
            cv2.imwrite(os.path.join('/data/disk1/xukepeng/project/diffhdr/tests/testimg/outs',str(i)+'gt.png'),np.array(data['gt'].permute(1,2,0).numpy()*65535,np.uint16))
