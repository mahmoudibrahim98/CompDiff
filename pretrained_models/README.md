# Pretrained models

This directory contains pretrained weights used for validation and evaluation.

## Layout

```
pretrained_models/
├── fid_radnet/
│   ├── RadImageNet-ResNet50_notop.pth
│   └── radimagenet-models-main/      # RadImageNet models (for FID / RadImageNet metrics)
└── README.md
```

## Sex model

No checkpoint is needed. Sex accuracy on generated chest X-rays uses the MIRA sex model from
[torchxrayvision](https://github.com/mlmed/torchxrayvision) (`xrv.baseline_models.mira.SexModel`),
whose weights are downloaded automatically on first use. The `validation_sex_model_path` config
key is kept for compatibility and is ignored.

## FID / RadImageNet

- **Path:** `fid_radnet/` — RadImageNet ResNet50 weights (`RadImageNet-ResNet50_notop.pth`) and the `radimagenet-models` package (in `radimagenet-models-main/radimagenet-models-main/`).
- **Where it’s used:** When `compute_fid_radimagenet: true` in your training/validation config, the validation monitor in `gen_source/run_validation_monitor_debug.py` computes:
  - **Overall** `val/fid_radimagenet` (FID between real and generated images using RadImageNet embeddings).
  - **Per-subgroup** FID RadImageNet (per sex, ethnicity, age group) when `compute_subgroup_metrics: true`.
  - **Per-intersectional** FID RadImageNet (age × ethnicity × sex) when subgroup metrics are enabled.
- **Loading:** `gen_source/validation_metrics.py` loads RadImageNet from this directory when the weights and package are present; otherwise it falls back to `torch.hub` (`Warvito/radimagenet-models`).

## Our trained HCN checkpoints

Our trained best checkpoints for the HCN model (chest X-ray and fundus) will be made available online for reproduction and downstream evaluation. Links will be added here once released.
