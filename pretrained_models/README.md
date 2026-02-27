# Pretrained models

This directory contains pretrained weights used for validation and evaluation.

## Layout

```
pretrained_models/
├── sex/
│   └── resnet-all/
│       └── epoch=13-step=7125.ckpt   # Sex classifier for validation metrics
├── fid_radnet/
│   ├── RadImageNet-ResNet50_notop.pth
│   └── radimagenet-models-main/      # RadImageNet models (for FID / RadImageNet metrics)
└── README.md
```

## Sex model

- **Path:** `sex/resnet-all/epoch=13-step=7125.ckpt`
- **Use:** Validation monitor uses this checkpoint to predict sex from generated images and compute subgroup metrics. Set `validation_sex_model_path` in your config to this path (default in the provided configs).

## FID / RadImageNet

- **Path:** `fid_radnet/` — RadImageNet ResNet50 weights (`RadImageNet-ResNet50_notop.pth`) and the `radimagenet-models` package (in `radimagenet-models-main/radimagenet-models-main/`).
- **Where it’s used:** When `compute_fid_radimagenet: true` in your training/validation config, the validation monitor in `gen_source/run_validation_monitor_debug.py` computes:
  - **Overall** `val/fid_radimagenet` (FID between real and generated images using RadImageNet embeddings).
  - **Per-subgroup** FID RadImageNet (per sex, ethnicity, age group) when `compute_subgroup_metrics: true`.
  - **Per-intersectional** FID RadImageNet (age × ethnicity × sex) when subgroup metrics are enabled.
- **Loading:** `gen_source/validation_metrics.py` loads RadImageNet from this directory when the weights and package are present; otherwise it falls back to `torch.hub` (`Warvito/radimagenet-models`).

## Our trained HCN checkpoints

Our trained best checkpoints for the HCN model (chest X-ray and fundus) will be made available online for reproduction and downstream evaluation. Links will be added here once released.
