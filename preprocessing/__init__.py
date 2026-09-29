"""
Preprocessing utilities for satellite imagery and field data.

Modules:
- sentinel2_preprocessor: Sentinel-2 band reading, resampling, and feature generation
- satellite_dataset: PyTorch Dataset for satellite patch training
- gedi_reader: GEDI AGBD data reader
- spatial_matcher: Spatial matching between Sentinel-2 and GEDI
- temporal_matcher: Temporal matching utilities
- quality_filter: Quality filtering for Sentinel-2 + GEDI pairs
- patch_generator: Patch generation around GEDI reference points
"""

from .sentinel2_preprocessor import (
    Sentinel2Normalizer,
    build_sentinel2_feature_stack,
    calculate_sentinel2_indices,
    preprocess_sentinel2,
)
from .satellite_dataset import SatelliteDataset
from .gedi_reader import read_gedi_csv, filter_gedi_records
from .spatial_matcher import (
    get_sentinel2_bounds,
    point_in_bounds,
    match_gedi_to_sentinel2,
    matches_to_dataframe,
)
from .temporal_matcher import (
    compute_temporal_difference,
    filter_by_temporal_diff,
)
from .quality_filter import (
    apply_quality_filters,
    filter_gedi_quality,
    filter_sentinel2_pixels,
    validate_sentinel2_bands,
)
from .patch_generator import (
    PatchMetadata,
    generate_patch,
    generate_patch_dataset,
    extract_patch,
)

__all__ = [
    "Sentinel2Normalizer",
    "build_sentinel2_feature_stack",
    "calculate_sentinel2_indices",
    "preprocess_sentinel2",
    "SatelliteDataset",
    "read_gedi_csv",
    "filter_gedi_records",
    "get_sentinel2_bounds",
    "point_in_bounds",
    "match_gedi_to_sentinel2",
    "matches_to_dataframe",
    "compute_temporal_difference",
    "filter_by_temporal_diff",
    "apply_quality_filters",
    "filter_gedi_quality",
    "filter_sentinel2_pixels",
    "validate_sentinel2_bands",
    "PatchMetadata",
    "generate_patch",
    "generate_patch_dataset",
    "extract_patch",
]
