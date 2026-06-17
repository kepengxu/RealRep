# Dataset Organization for DeCoMix-HDR

This document describes how the released DeCoMix-HDR code expects HDRTV4K/HDRTVDM and HDRTV1K data to be organized. It is intended to make the training, validation, and full testing configs reproducible without editing dataset code.

## 1. Datasets used by the released configs

DeCoMix-HDR uses two prepared LMDB datasets:

| Purpose | Dataset | Default path in configs | Dataset class |
|---|---|---|---|
| Training | HDRTV4K / HDRTVDM train split with tiny SDR entries | `./datasets/HDRTV4K/trainwtiny.lmdb` | `HDRTVDMDATASETMIX_UNIFIEDYXY_CONTRAST` |
| In-training known-degradation validation | HDRTV4K / HDRTVDM sample test split | `./datasets/HDRTV4K/test/sample/test_sample.lmdb` | `HDRTVDMDATASET_RGB_CONTRAST` |
| Full known-degradation evaluation | HDRTV4K / HDRTVDM full test split | `./datasets/HDRTV4K/test/test.lmdb` | `HDRTVDMDATASET_RGB_CONTRAST` |
| Unknown-degradation evaluation | HDRTV1K test split | `./datasets/HDRTV1K/test_set_ori.lmdb` | `HDRTVDMDATASET_CONTRAST` |

The released training config is:

```text
options/train/train_decomix_hdr.yml
```

The released full-evaluation config is:

```text
options/test/test_decomix_hdr_fullval.yml
```

## 2. Recommended directory tree

The simplest setup is to keep the configs unchanged and place or symlink the prepared datasets under `datasets/`:

```text
DeCoMix-HDR-open-source/
├── datasets/
│   ├── HDRTV4K/
│   │   ├── trainwtiny.lmdb/
│   │   │   ├── data.mdb
│   │   │   ├── lock.mdb
│   │   │   ├── names.pkl
│   │   │   ├── negatives_luma_fullrank.txt
│   │   │   ├── negatives_chroma_fullrank.txt
│   │   │   ├── negatives_luma.txt                  # fallback if fullrank file is absent
│   │   │   └── negatives_chroma.txt                # fallback if fullrank file is absent
│   │   └── test/
│   │       ├── test.lmdb/
│   │       │   ├── data.mdb
│   │       │   ├── lock.mdb
│   │       │   └── names.pkl
│   │       └── sample/
│   │           └── test_sample.lmdb/
│   │               ├── data.mdb
│   │               ├── lock.mdb
│   │               └── names.pkl
│   └── HDRTV1K/
│       └── test_set_ori.lmdb/
│           ├── data.mdb
│           ├── lock.mdb
│           ├── names.pkl
│           └── names.txt                           # optional helper file; not required by code
├── pretrained_models/
│   └── decomix_hdr_net_g_400000.pth
├── options/
└── train.py / test.py
```

A storage-efficient setup is to keep large LMDBs outside the code repository and create symlinks:

```bash
mkdir -p datasets
ln -s /path/to/HDRTV4K datasets/HDRTV4K
ln -s /path/to/HDRTV1K datasets/HDRTV1K
```

The helper script `scripts/configure_decomix_data.py` can create these symlinks and verify the required files automatically.

## 3. Current prepared LMDB format

The code expects BasicSR LMDB folders. Each LMDB folder must contain:

```text
data.mdb
lock.mdb
names.pkl
```

The prepared datasets used for the released configs have the following metadata structure:

| LMDB | Number of entries | `names.pkl` row length | First row structure |
|---|---:|---:|---|
| `HDRTV4K/trainwtiny.lmdb` | 106,960 | 18 | `HDR`, 8 SDR styles, `TINY_HDR`, 8 tiny SDR styles |
| `HDRTV4K/test/test.lmdb` | 400 | 9 | `HDR`, 8 SDR styles |
| `HDRTV4K/test/sample/test_sample.lmdb` | 100 | 9 | `HDR`, 8 SDR styles |
| `HDRTV1K/test_set_ori.lmdb` | 117 | 2 | `HDR`, `SDR` |

Example `names.pkl` rows:

```python
# HDRTV4K trainwtiny.lmdb
['HDR_0', 'SDR1_0', 'SDR2_0', 'SDR3_0', 'SDR4_0', 'SDR5_0', 'SDR6_0', 'SDR7_0', 'SDR8_0',
 'TINY_HDR_0', 'TINY_SDR1_0', 'TINY_SDR2_0', 'TINY_SDR3_0', 'TINY_SDR4_0',
 'TINY_SDR5_0', 'TINY_SDR6_0', 'TINY_SDR7_0', 'TINY_SDR8_0']

# HDRTV4K test/test.lmdb and test/sample/test_sample.lmdb
['HDR_0', 'SDR1_0', 'SDR2_0', 'SDR3_0', 'SDR4_0', 'SDR5_0', 'SDR6_0', 'SDR7_0', 'SDR8_0']

# HDRTV1K test_set_ori.lmdb
('hdr-094.png', 'sdr-094.png')
```

For HDRTV4K/HDRTVDM, `sdrset` in the YAML config selects which SDR style to use:

```text
sdrset: 1 -> SDR1_* key
sdrset: 2 -> SDR2_* key
...
sdrset: 8 -> SDR8_* key
```

The HDR key is always index `0`.

## 4. Training LMDB requirements

Training uses the mixed dataset wrapper `HDRTVDMDATASETMIX_UNIFIEDYXY_CONTRAST` in `basicsr/data/hdrtvmix_contrast_dataset.py`. The wrapper instantiates eight HDRTV4K training subsets, all pointing to the same `trainwtiny.lmdb` but with `sdrset` from 1 to 8:

```yaml
datasets:
  train:
    type: HDRTVDMDATASETMIX_UNIFIEDYXY_CONTRAST
    dataset:
      hdrtvdm1:
        root_dir: ./datasets/HDRTV4K/trainwtiny.lmdb
        sdrset: 1
      ...
      hdrtvdm8:
        root_dir: ./datasets/HDRTV4K/trainwtiny.lmdb
        sdrset: 8
    collate_ranked_contrast: true
```

The training dataset returns the following core fields:

| Field | Meaning |
|---|---|
| `lq` | SDR input image selected by `sdrset` |
| `gt` | HDR ground-truth image |
| `im_q` | query view for contrastive representation learning |
| `im_k` | key view, loaded from the tiny SDR entry |
| `im_negs_I_raw` | luma-dominant negative samples |
| `im_negs_TP_raw` | chroma-dominant negative samples |
| `im_negs_I_rank` | luma negative ranks |
| `im_negs_TP_rank` | chroma negative ranks |

The final entry in each training `names.pkl` row must be a tiny SDR key whose name contains `SDR` and `TINY`/`Tiny`. The code checks this explicitly before loading contrastive key views.

## 5. Ranked negative files

Luma-/chroma-aware contrastive mining requires ranked negative-list files inside the training LMDB folder:

```text
HDRTV4K/trainwtiny.lmdb/
├── negatives_luma_fullrank.txt
└── negatives_chroma_fullrank.txt
```

If the `*_fullrank.txt` files are absent, the dataset class falls back to:

```text
negatives_luma.txt
negatives_chroma.txt
```

At least one luma file and one chroma file must exist. Without these files, training cannot reproduce the DeCoMix-HDR ranked contrastive mining behavior.

The expected text format is one query key per line:

```text
QUERY_KEY:NEG_KEY_1,NEG_KEY_2,NEG_KEY_3,...
```

The loader parses each line with `name, negatives = line.strip().split(':')` and then splits the negative list by comma. Keep the key strings consistent with the keys stored in the LMDB.

## 6. Validation and test LMDB requirements

Validation and testing use simpler paired datasets. They only need `data.mdb`, `lock.mdb`, and `names.pkl`.

### HDRTV4K known-degradation evaluation

`options/test/test_decomix_hdr_fullval.yml` defines eight HDRTV4K test datasets:

```yaml
test_2:
  name: hdrtvdm1
  type: HDRTVDMDATASET_RGB_CONTRAST
  root_dir: ./datasets/HDRTV4K/test/test.lmdb
  sdrset: 1
...
test_9:
  name: hdrtvdm8
  type: HDRTVDMDATASET_RGB_CONTRAST
  root_dir: ./datasets/HDRTV4K/test/test.lmdb
  sdrset: 8
```

Each dataset uses the same HDRTV4K test LMDB but selects a different SDR style through `sdrset`.

### HDRTV1K unknown-degradation evaluation

`test_1` uses HDRTV1K:

```yaml
test_1:
  name: hdrtv1k
  type: HDRTVDMDATASET_CONTRAST
  root_dir: ./datasets/HDRTV1K/test_set_ori.lmdb
  sdrset: 1
```

HDRTV1K rows contain only HDR and SDR keys, so `sdrset: 1` selects the SDR input.

## 7. Quick configuration

Use the provided helper to create dataset symlinks, verify LMDB files, and generate local configs:

```bash
python scripts/configure_decomix_data.py \
  --hdrtv4k-root /path/to/HDRTV4K \
  --hdrtv1k-root /path/to/HDRTV1K
```

By default, it will:

1. Create `datasets/HDRTV4K -> /path/to/HDRTV4K`.
2. Create `datasets/HDRTV1K -> /path/to/HDRTV1K`.
3. Verify the required LMDB folders, `names.pkl` files, and ranked negative files.
4. Verify `pretrained_models/decomix_hdr_net_g_400000.pth` if present or requested.
5. Write local config copies under `options/local/`:
   - `options/local/train_decomix_hdr_local.yml`
   - `options/local/test_decomix_hdr_fullval_local.yml`

Run training with:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/local/train_decomix_hdr_local.yml
```

Run evaluation with:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py -opt options/local/test_decomix_hdr_fullval_local.yml
```

If symlinks are not desired, generate local configs with absolute dataset paths:

```bash
python scripts/configure_decomix_data.py \
  --hdrtv4k-root /path/to/HDRTV4K \
  --hdrtv1k-root /path/to/HDRTV1K \
  --absolute-config-paths \
  --no-symlink
```

## 8. Checkpoint path

The released checkpoint is expected at:

```text
pretrained_models/decomix_hdr_net_g_400000.pth
```

The evaluation config should load it with:

```yaml
path:
  pretrain_network_g: ./pretrained_models/decomix_hdr_net_g_400000.pth
  strict_load_g: true
  param_key_g: params_ema
  param_key_feat_single_moco_base: feat_params_single_moco_base
```

The checkpoint file contains both networks required at inference time:

| Key | Used for |
|---|---|
| `params_ema` | EMA HDR mapping network (`network_g`) |
| `feat_params_single_moco_base` | contrastive feature encoder (`network_feat`) |
| `params` | non-EMA generator weights |

## 9. Sanity checks before running

Run this before training or testing:

```bash
python scripts/configure_decomix_data.py \
  --hdrtv4k-root /path/to/HDRTV4K \
  --hdrtv1k-root /path/to/HDRTV1K \
  --check-only
```

The script checks for:

```text
HDRTV4K/trainwtiny.lmdb/data.mdb
HDRTV4K/trainwtiny.lmdb/lock.mdb
HDRTV4K/trainwtiny.lmdb/names.pkl
HDRTV4K/trainwtiny.lmdb/negatives_luma_fullrank.txt or negatives_luma.txt
HDRTV4K/trainwtiny.lmdb/negatives_chroma_fullrank.txt or negatives_chroma.txt
HDRTV4K/test/test.lmdb/data.mdb
HDRTV4K/test/test.lmdb/lock.mdb
HDRTV4K/test/test.lmdb/names.pkl
HDRTV4K/test/sample/test_sample.lmdb/data.mdb
HDRTV4K/test/sample/test_sample.lmdb/lock.mdb
HDRTV4K/test/sample/test_sample.lmdb/names.pkl
HDRTV1K/test_set_ori.lmdb/data.mdb
HDRTV1K/test_set_ori.lmdb/lock.mdb
HDRTV1K/test_set_ori.lmdb/names.pkl
```

Common problems:

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: names.pkl` | The YAML `root_dir` points to the wrong LMDB folder | Fix the path or recreate the symlink |
| missing `negatives_luma*` or `negatives_chroma*` | Training LMDB does not include ranked negative lists | Add the ranked negative files to `trainwtiny.lmdb` |
| tiny-name assertion error | The training `names.pkl` row does not include tiny SDR keys | Use the `trainwtiny.lmdb` format with tiny SDR entries |
| checkpoint loading error | `pretrain_network_g` points to a missing or incompatible checkpoint | Use `pretrained_models/decomix_hdr_net_g_400000.pth` or update the path |
| evaluation only works for HDRTV1K | HDRTV4K test LMDB does not contain SDR1--SDR8 keys | Rebuild or use the prepared HDRTV4K test LMDB |

## 10. Notes

- Do not commit `datasets/`, `experiments/`, `results/`, or generated local configs.
- Keep `collate_ranked_contrast: true` for the main training config.
- Keep `sdrset` values aligned with the number of SDR keys in `names.pkl`.
- The historical `RealRep` class and registry names in code are retained for checkpoint compatibility.
