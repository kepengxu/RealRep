# RealRep

Code for paper *[RealRep: Generalized SDR-to-HDR Conversion via Attribute-Disentangled Representation Learning](https://arxiv.org/abs/2505.07322)*.


## Introduction

This project is developed based on [BasicSR](https://github.com/xinntao/BasicSR).

For matters not covered in this document, please refer to the detailed documentation of the original project.


### Structure of directories
```shell
RealRep/                        # Project root directory
  basicsr/                      # Core source code
    archs/                      # Network architecture modules
      RealRep_fuse_norm_contra.py         # Main network architecture
    data/                       # Dataset-related modules
      hdrtvmix_contrast_dataset.py   # dataset class
    losses/                     # Loss function modules
    metrics/                    # Evaluation metrics
    models/
      itm_model.py              # Model definition
    ops/                        
    utils/
      hdr_util.py               # Utility functions for representation conversion, etc.
  options/                      # Experiment configurations
    test/                       # Test configurations
      RealRep
    train/                      # Training configurations
      RealRep
  scripts/                      # Helper scripts
  requirements.txt              # Dependencies list
  test.py                       # Testing entry point
  train.py                      # Training entry point
```


## Dataset

HDRTV1K dataset \[[GitHub](https://github.com/chxy95/HDRTVNet)\]

HDRTV4K dataset \[[GitHub](https://github.com/AndreGuo/HDRTVDM)]

## How To Runs

```
cd RealRep
pip install -r requirements.txt
```

Train

```shell
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/train_realrep_mix_prompt_contra_feat_noema_ITPContrast_UNet_onlyycbcr_flip_norm.yml
```

Test

```shell
CUDA_VISIBLE_DEVICES=0 python test.py -opt options/test/test_realrep_mix_tipcontra_2stage_unet_onlyycbcr_fuse_norm_zero.yml
```
