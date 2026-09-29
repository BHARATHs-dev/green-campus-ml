"""
Patch generation for Sentinel-2 + GEDI training data.

Extracts configurable-size patches around GEDI reference points
and stores them as GeoTIFF files with associated metadata.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds

from .sentinel2_preprocessor import (
    SENTINEL2_BANDS,
    _compute_valid_pixel_ratio,
)

logger = logging.getLogger(__name__)


@dataclass
class PatchMetadata:
    """Metadata for a generated patch."""
    sample_id: str
    sentinel2_source: str
    gedi_source: str
    latitude: float
    longitude: float
    crs: str
    patch_size_pixels: int
    patch_size_meters: float
    valid_pixel_ratio: float
    agbd: float
    agbd_carbon: float
    sentinel2_acquisition_time: Optional[str] = None
    gedi_acquisition_time: Optional[str] = None
    time_diff_days: Optional[float] = None
    transform: Optional[List[float]] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def get_pixel_coordinates(
    lat: float,
    lon: float,
    transform: rasterio.Affine,
    crs: str = "EPSG:4326",
    tile_crs: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Convert lat/lon to pixel coordinates in a Sentinel-2 tile.

    If CRS differ, transform coordinates first.
    """
    if tile_crs and tile_crs != crs:
        try:
            from rasterio.warp import transform
            x, y = transform(crs, tile_crs, [lon], [lat])
            lon = float(x[0])
            lat = float(y[0])
        except Exception as e:
            logger.warning("CRS transform failed: %s", e)

    col = (lon - transform.c) / transform.a
    row = (lat - transform.f) / transform.e
    return float(col), float(row)


def extract_patch(
    sentinel2_path: Union[str, Path],
    lat: float,
    lon: float,
    patch_size_pixels: int = 64,
    band_names: Optional[List[str]] = None,
) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Extract a patch around a GEDI point from a Sentinel-2 tile.

    Args:
        sentinel2_path: Path to Sentinel-2 GeoTIFF
        lat: Latitude of GEDI point
        lon: Longitude of GEDI point
        patch_size_pixels: Patch size in pixels (at 10m resolution)
        band_names: List of band names to extract (default: all S2 bands)

    Returns:
        (patch_array, metadata) or (None, metadata) on failure
    """
    sentinel2_path = Path(sentinel2_path)
    metadata = {"file": str(sentinel2_path), "errors": []}

    if not sentinel2_path.exists():
        metadata["errors"].append(f"File not found: {sentinel2_path}")
        return None, metadata

    if band_names is None:
        band_names = list(SENTINEL2_BANDS.keys())

    try:
        with rasterio.open(sentinel2_path) as src:
            transform = src.transform
            tile_crs = src.crs.to_string() if src.crs else "EPSG:4326"

            col, row = get_pixel_coordinates(lat, lon, transform, tile_crs=tile_crs)

            half = patch_size_pixels // 2
            window = Window(
                col - half,
                row - half,
                patch_size_pixels,
                patch_size_pixels,
            )

            bands_data = {}
            band_map = {}
            descriptions = src.descriptions

            if descriptions and any(descriptions):
                for idx, desc in enumerate(descriptions, start=1):
                    if desc:
                        desc_upper = str(desc).upper().strip()
                        for band_name in band_names:
                            if band_name in desc_upper:
                                band_map[band_name] = idx
                                break

            for band_name in band_names:
                if band_name not in band_map:
                    continue
                band_idx = band_map[band_name]
                arr = src.read(band_idx, window=window)
                if arr.shape != (patch_size_pixels, patch_size_pixels):
                    padded = np.full(
                        (patch_size_pixels, patch_size_pixels),
                        fill_value=src.nodata if src.nodata is not None else 0,
                        dtype=src.dtypes[0],
                    )
                    h, w = arr.shape
                    padded[:h, :w] = arr
                    arr = padded
                bands_data[band_name] = arr

            if not bands_data:
                metadata["errors"].append("No bands extracted")
                return None, metadata

            metadata["transform"] = list(transform)[:6]
            metadata["crs"] = tile_crs
            metadata["valid_pixel_ratio"] = _compute_valid_pixel_ratio(bands_data)

    except Exception as e:
        metadata["errors"].append(f"Patch extraction failed: {e}")
        return None, metadata

    return bands_data, metadata


def generate_patch(
    sample_id: str,
    sentinel2_path: Union[str, Path],
    gedi_row: pd.Series,
    patch_size_pixels: int = 64,
    output_dir: Union[str, Path] = "ai/datasets/satellite_agbd/patches",
    band_names: Optional[List[str]] = None,
    save_patch: bool = True,
) -> Optional[PatchMetadata]:
    """
    Generate a single patch for a GEDI point.

    Args:
        sample_id: Unique sample identifier
        sentinel2_path: Path to Sentinel-2 tile
        gedi_row: GEDI observation as pandas Series
        patch_size_pixels: Patch size in pixels
        output_dir: Directory to save patch files
        band_names: Bands to include
        save_patch: If True, save patch as GeoTIFF

    Returns:
        PatchMetadata or None on failure
    """
    if band_names is None:
        band_names = list(SENTINEL2_BANDS.keys())

    lat = float(gedi_row["latitude"])
    lon = float(gedi_row["longitude"])
    agbd = float(gedi_row["agbd"])
    agbd_carbon = agbd * 0.47

    bands_data, meta = extract_patch(
        sentinel2_path,
        lat,
        lon,
        patch_size_pixels,
        band_names,
    )

    if bands_data is None:
        logger.warning("Failed to extract patch %s: %s", sample_id, meta["errors"])
        return None

    patch_meta = PatchMetadata(
        sample_id=sample_id,
        sentinel2_source=str(sentinel2_path),
        gedi_source=str(gedi_row.get("sample_id", sample_id)),
        latitude=lat,
        longitude=lon,
        crs=meta.get("crs", "EPSG:4326"),
        patch_size_pixels=patch_size_pixels,
        patch_size_meters=patch_size_pixels * 10.0,
        valid_pixel_ratio=meta.get("valid_pixel_ratio", 0.0),
        agbd=agbd,
        agbd_carbon=agbd_carbon,
        transform=meta.get("transform"),
        errors=meta.get("errors", []),
    )

    if "acquisition_time" in gedi_row:
        patch_meta.sentinel2_acquisition_time = str(gedi_row["acquisition_time"]) if pd.notna(gedi_row["acquisition_time"]) else None
    if "time" in gedi_row:
        patch_meta.gedi_acquisition_time = str(gedi_row["time"]) if pd.notna(gedi_row["time"]) else None

    if save_patch:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        patch_file = output_path / f"{sample_id}.tif"

        try:
            with rasterio.open(
                patch_file,
                "w",
                driver="GTiff",
                height=patch_size_pixels,
                width=patch_size_pixels,
                count=len(bands_data),
                dtype=rasterio.float32,
                crs=patch_meta.crs,
                transform=rasterio.Affine(*meta["transform"]) if meta.get("transform") else rasterio.Affine.identity(),
            ) as dst:
                for i, (band_name, arr) in enumerate(bands_data.items(), start=1):
                    dst.write(arr.astype(np.float32), i)
                    dst.set_band_description(i, band_name)
        except Exception as e:
            logger.warning("Failed to save patch %s: %s", sample_id, e)
            patch_meta.errors.append(f"Save failed: {e}")

    return patch_meta


def generate_patch_dataset(
    manifest_df: pd.DataFrame,
    patch_size_pixels: int = 64,
    output_dir: Union[str, Path] = "ai/datasets/satellite_agbd/patches",
    band_names: Optional[List[str]] = None,
    max_errors: int = 100,
) -> pd.DataFrame:
    """
    Generate patches for all records in a manifest.

    Args:
        manifest_df: DataFrame with sample_id, sentinel2_path, latitude, longitude, agbd
        patch_size_pixels: Patch size in pixels
        output_dir: Directory to save patches
        band_names: Bands to include
        max_errors: Maximum number of errors before aborting

    Returns:
        DataFrame with patch metadata
    """
    if band_names is None:
        band_names = list(SENTINEL2_BANDS.keys())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    errors = 0

    for _, row in manifest_df.iterrows():
        sample_id = str(row["sample_id"])
        s2_path = row.get("sentinel2_path")
        if not s2_path or pd.isna(s2_path):
            errors += 1
            continue

        meta = generate_patch(
            sample_id=sample_id,
            sentinel2_path=s2_path,
            gedi_row=row,
            patch_size_pixels=patch_size_pixels,
            output_dir=output_dir,
            band_names=band_names,
            save_patch=True,
        )

        if meta is None:
            errors += 1
            continue

        records.append({
            "sample_id": meta.sample_id,
            "sentinel2_source": meta.sentinel2_source,
            "gedi_source": meta.gedi_source,
            "latitude": meta.latitude,
            "longitude": meta.longitude,
            "crs": meta.crs,
            "patch_size_pixels": meta.patch_size_pixels,
            "patch_size_meters": meta.patch_size_meters,
            "valid_pixel_ratio": meta.valid_pixel_ratio,
            "agbd": meta.agbd,
            "agbd_carbon": meta.agbd_carbon,
            "sentinel2_acquisition_time": meta.sentinel2_acquisition_time,
            "gedi_acquisition_time": meta.gedi_acquisition_time,
            "time_diff_days": meta.time_diff_days,
            "transform": str(meta.transform) if meta.transform else None,
            "patch_path": str(output_dir / f"{sample_id}.tif"),
            "errors": "; ".join(meta.errors) if meta.errors else None,
        })

        if errors >= max_errors:
            logger.error("Aborting patch generation: too many errors (%d)", errors)
            break

    result_df = pd.DataFrame(records)
    logger.info(
        "Generated %d patches (%d errors)",
        len(result_df),
        errors,
    )

    return result_df
