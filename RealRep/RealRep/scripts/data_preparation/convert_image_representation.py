import os
import cv2
import numpy as np

from basicsr.utils.img_util import imread, imwrite
from basicsr.utils.hdr_util import SdrHdrPairRepresentationConversion

dataset_root = './datasets/HDRTV1K/test'  # './dataset/train' | './dataset/test'
hdr_dir = 'test_hdr'  # 'train_hdr' | 'test_hdr'
sdr_dir = 'test_sdr'  # 'train_sdr' | 'test_sdr'
opt_domain = 'ictcp'

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tif', 'npy']


def is_image_file(filename):
    return os.path.splitext(filename)[-1] in IMG_EXTENSIONS


def get_img_list_from_folder(dataroot):
    assert os.path.isdir(dataroot), f"{dataroot} is not a valid directory."

    img_list = []
    for dirpath, _, fnames in sorted(os.walk(dataroot)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                img_list.append(img_path)

    assert img_list, f'{dataroot} has no valid image file'
    return img_list


if __name__ == '__main__':
    sdr_path = os.path.join(dataset_root, sdr_dir)
    hdr_path = os.path.join(dataset_root, hdr_dir)

    lq_img_list = get_img_list_from_folder(sdr_path)
    gt_img_list = get_img_list_from_folder(hdr_path)

    assert len(lq_img_list) == len(gt_img_list), \
        'SDR and HDR data folders have different number of images - {}, {}.'.format(
            len(lq_img_list), len(gt_img_list))

    n_imgs = len(gt_img_list)

    print("Found {} pairs of SDR-HDR images.".format(n_imgs))

    if opt_domain and opt_domain != 'original':
        new_sdr_dir = os.path.join(dataset_root, "{}_{}".format(sdr_dir, opt_domain))
        new_hdr_dir = os.path.join(dataset_root, "{}_{}".format(hdr_dir, opt_domain))

        if not os.path.exists(new_sdr_dir):
            os.makedirs(new_sdr_dir)
        if not os.path.exists(new_hdr_dir):
            os.makedirs(new_hdr_dir)

        convert = SdrHdrPairRepresentationConversion(src='rgb', dst=opt_domain,
                                                     zero_one_norm=True)

        progress_milestone = range(0, n_imgs, n_imgs // 20)

        for idx in range(n_imgs):
            if idx in progress_milestone:
                print("{}%".format(idx / n_imgs * 100))

            lq_path = lq_img_list[idx]
            gt_path = gt_img_list[idx]
            fname = os.path.split(lq_path)[-1]
            new_lq_path = os.path.join(new_sdr_dir, fname)
            new_gt_path = os.path.join(new_hdr_dir, fname)

            img_lq_orig = imread(lq_path, float32=True)
            img_gt_orig = imread(gt_path, float32=True)

            img_lq, img_gt = convert(img_lq_orig, img_gt_orig)

            img_lq = (img_lq * 255.).round().astype(np.uint8)
            img_gt = (img_gt * 65535.).round().astype(np.uint16)

            imwrite(cv2.cvtColor(img_lq, cv2.COLOR_RGB2BGR), new_lq_path)
            imwrite(cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR), new_gt_path)

