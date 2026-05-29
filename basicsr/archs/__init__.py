from copy import deepcopy

from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY
from .RealRep_fuse_zero_norm_arch import RealRep_fuse_norm_single
from .contrastive import MoCoWithUNetDualNegLatentMix

__all__ = ['build_network']


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net
