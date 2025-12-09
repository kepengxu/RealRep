import cv2
import numpy as np

from basicsr.metrics.metric_util import reorder_image
from basicsr.utils.hdr_util import ConvertToICtCp
from basicsr.utils.registry import METRIC_REGISTRY

bt2100_pq_rgb_to_ictcp = ConvertToICtCp('bt2100_pq')


@METRIC_REGISTRY.register()
def calculate_delta_e_itp(img, img2, crop_border=0, input_order='HWC', uint2float=True, bgr2rgb=True, relative=False):
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')

    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if uint2float:
        img = img.astype(np.float32) / 65535.
        img2 = img2.astype(np.float32) / 65535.

    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # convert RGB to ICtCp
    img_ictcp = bt2100_pq_rgb_to_ictcp(img.astype(np.float64))
    img2_ictcp = bt2100_pq_rgb_to_ictcp(img2.astype(np.float64))

    # scale ICtCp to create ITP
    scale_coeff = np.array([1, 0.5 * 1.823698, 1.887755]) if relative \
        else np.array([1, 0.5, 1.])
    img_itp = img_ictcp * scale_coeff
    img2_itp = img2_ictcp * scale_coeff

    # calculate DeltaE_ITP
    delta_e = 720 * np.sqrt(np.sum((img_itp - img2_itp) ** 2, axis=-1))

    return np.mean(delta_e)

