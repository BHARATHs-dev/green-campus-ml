"""
Quality filtering for Sentinel-2 + GEDI pairs.

Applies configurable filters and records statistics.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .sentinel2_preprocessor import (
    SENTINEL2_BANDS,
    REQUIRED_BANDS,
    _compute_valid_pixel_ratio,
)

logger = logging.getLogger(__name__)


@dataclass
class QualityFilterResult:
    """Result of quality filtering."""
    original_count: int
    filtered_count: int
    final_count: int
    stats: Dict[str, Any] = field(default_factory=dict)


def filter_gedi_quality(
    df: pd.DataFrame,
    min_agbd: float = 0.0,
    max_agbd: float = 500.0,
    require_quality_flag: bool = False,
    valid_quality_values: Optional[List[int]] = None,
    max_agbd_uncertainty: Optional[float] = None,
) -> QualityFilterResult:
    """
    Filter GEDI records by quality criteria.
    """
    from .gedi_reader import filter_gedi_records

    original_count = len(df)
    df, stats = filter_gedi_records(
        df,
        min_agbd=min_agbd,
        max_agbd=max_agbd,
        require_quality_flag=require_quality_flag,
        valid_quality_values=valid_quality_values,
        max_agbd_uncertainty=max_agbd_uncertainty,
    )

    return QualityFilterResult(
        original_count=original_count,
        filtered_count=original_count - len(df),
        final_count=len(df),
        stats=stats,
    )


def filter_sentinel2_pixels(
    filepath: Union[str, Path],
    min_valid_ratio: float = 0.5,
) -> Dict[str, Any]:
    """
    Check Sentinel-2 pixel quality.

    Returns:
        Dict with valid_pixel_ratio, passed, errors
    """
    filepath = Path(filepath)
    import rasterio

    result = {
        "file": str(filepath),
        "valid_pixel_ratio": None,
        "passed": False,
        "errors": [],
    }

    try:
        import rasterio
        with rasterio.open(filepath) as src:
            # Read all bands
            bands = {}
            for band_name, band_idx in zip(SENTINEL2_BANDS.keys(), range(1, src.count + 1)):
                bands[band_name] = src.read(band_idx)

        valid_ratio = _compute_valid_pixel_ratio(bands)
        result["valid_pixel_ratio"] = valid_ratio
        result["passed"] = valid_ratio >= min_valid_ratio

        if not result["passed"]:
            result["errors"].append(
                f"Valid pixel ratio {valid_ratio:.2f} below threshold {min_valid_ratio}"
            )
    except Exception as e:
        result["errors"].append(f"Failed to read Sentinel-2 file: {e}")

    return result


def validate_sentinel2_bands(
    filepath: Union[str, Path],
) -> Dict[str, Any]:
    """
    Validate that a Sentinel-2 file contains required bands.
    """
    filepath = Path(filepath)
    import rasterio

    result = {
        "file": str(filepath),
        "valid": False,
        "identified_bands": [],
        "missing_required": [],
        "errors": [],
    }

    try:
        with rasterio.open(filepath) as src:
            descriptions = src.descriptions
            if descriptions:
                identified = []
                for idx, desc in enumerate(descriptions, start=1):
                    if desc:
                        desc_upper = str(desc).upper().strip()
                        for band_name in SENTINEL2_BANDS:
                            if band_name in desc_upper:
                                identified.append(band_name)
                                break
                result["identified_bands"] = identified
            else:
                # Fallback: assume sequential bands
                for i in range(1, min(src.count, len(REQUIRED_BANDS)) + 1):
                    pass

            missing = [b for b in REQUIRED_BANDS if b not in result["identified_bands"]]
            result["missing_required"] = missing
            result["valid"] = len(missing) == 0

            if not result["valid"]:
                result["errors"].append(f"Missing required bands: {missing}")
    except Exception as e:
        result["errors"].append(f"Failed to validate bands: {e}")

    return result


def apply_quality_filters(
    manifest_df: pd.DataFrame,
    sentinel2_dir: Union[str, Path],
    min_valid_pixel_ratio: float = 0.5,
    require_valid_bands: bool = True,
) -> pd.DataFrame:
    """
    Apply all quality filters to a manifest DataFrame.

    Args:
        manifest_df: DataFrame with sample_id, sentinel2_path, etc.
        sentinel2_dir: Base directory for Sentinel-2 files
        min_valid_pixel_ratio: Minimum ratio of valid pixels
        require_valid_bands: If True, filter out records with missing bands

    Returns:
        Filtered DataFrame
    """
    original_count = len(manifest_df)
    stats = {"original_count": original_count}

    # Filter by valid pixel ratio
    if min_valid_pixel_ratio > 0:
        valid_mask = []
        for _, row in manifest_df.iterrows():
            s2_path = row.get("sentinel2_path")
            if not s2_path or pd.isna(s2_path):
                valid_mask.append(False)
                continue
            s2_file = Path(sentinel2_dir) / s2_path
            if not s2_file.exists():
                valid_mask.append(False)
                continue
            quality = filter_sentinel2_pixels(s2_file, min_valid_pixel_ratio)
            valid_mask.append(quality["passed"])

        valid_mask = np.array(valid_mask)
        stats["invalid_pixels_filtered"] = int((~valid_mask).sum())
        manifest_df = manifest_df[valid_mask].copy()

    # Filter by band validity
    if require_valid_bands:
        valid_mask = []
        for _, row in manifest_df.iterrows():
            s2_path = row.get("sentinel2_path")
            if not s2_path or pd.isna(s2_path):
                valid_mask.append(False)
                continue
            s2_file = Path(sentinel2_dir) / s2_path
            if not s2_file.exists():
                valid_mask.append(False)
                continue
            validation = validate_sentinel2_bands(s2_file)
            valid_mask.append(validation["valid"])

        valid_mask = np.array(valid_mask)
        stats["invalid_bands_filtered"] = int((~valid_mask).sum())
        manifest_df = manifest_df[valid_mask].copy()

    stats["final_count"] = len(manifest_df)
    stats["filtered_count"] = original_count - len(manifest_df)

    logger.info(
        "Quality filtering: %d -> %d records",
        original_count,
        len(manifest_df),
    )

    return manifest_df
