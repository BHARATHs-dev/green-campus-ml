# Datasets

This directory contains data for three separate AGB estimation pipelines. These datasets are **not mixed** during training.

## Directory Structure

```
ai/datasets/
├── image/
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── image_labels.csv
├── satellite/
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── satellite_labels.csv
└── ...
```

## Dataset Separation Policy

| Dataset | Model | Input | Target |
|---------|-------|-------|--------|
| Field data | CatBoostRegressor | diameter, height, year, group, site | AGB (kg) |
| Tree images | EfficientNet-B0 + Regression Head | RGB image (224×224) | AGB (kg) |
| Satellite patches | Satellite CNN | Multi-band patches (B2, B3, B4, B8, B11, B12) | AGB (kg) |

**Do not combine these datasets for joint training.** Each model is trained independently on its matched modality.

Future multimodal fusion (Field + Image + Satellite → Multimodal Model) requires properly matched paired data and is a separate enhancement phase.

## Ground Truth Requirement

Every sample must be linked to a real measured AGB value:

- **Image dataset:** `image_id,agb_kg,tree_id` — each tree photo maps to field-measured AGB
- **Satellite dataset:** `patch_id,plot_id,agb_kg` — each satellite patch maps geographically to a field plot with measured AGB

## Preparation Scripts

- `ai/training/prepare_image_dataset.py` — splits raw images into train/val/test
- `ai/training/prepare_satellite_dataset.py` — splits raw satellite bands into train/val/test
