# Sentinel-2 + GEDI AGBD Dataset

## Structure

```
ai/datasets/satellite_agbd/
    sentinel2/        # Sentinel-2 GeoTIFF files
    gedi/             # GEDI AGBD data (CSV/Parquet)
    patches/          # Generated patch metadata
    metadata/         # Preprocessing artifacts (normalizer JSON, etc.)
    train.csv         # Training split manifest
    val.csv           # Validation split manifest
    test.csv          # Test split manifest
    README.md         # This file
```

## Data Sources

- **Sentinel-2 MSI:** Level-2A surface reflectance products
- **GEDI:** Level-4A Above Ground Biomass Density (AGBD)

## Target

AGBD in Mg/ha (metric tons per hectare)

Carbon = AGBD × 0.47

## Usage

```python
from ai.datasets.satellite_agbd.dataset import SatelliteAGBDataset
from ai.datasets.satellite_agbd.splits import create_splits

# Create splits
create_splits(
    manifest_path="ai/datasets/satellite_agbd/train.csv",
    group_col="tile_id",
    train_frac=0.8,
    val_frac=0.1,
    random_state=42,
)

# Load dataset
train_ds = SatelliteAGBDataset(
    root_dir="ai/datasets/satellite_agbd",
    split="train",
    normalizer=normalizer,
)
```

## Notes

- Do not place synthetic or placeholder data in this directory for production training.
- All spatial matching must use real coordinates and CRS-aware operations.
- Temporal matching requires actual acquisition timestamps.
