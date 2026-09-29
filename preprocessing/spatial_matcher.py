"""
Spatial matching between Sentinel-2 tiles and GEDI AGBD points.

Responsibilities:
- Read Sentinel-2 GeoTIFF spatial metadata
- Transform GEDI coordinates to Sentinel-2 CRS
- Determine point-in-polygon / bounding-box containment
- Associate GEDI observations with Sentinel-2 tiles/patches
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

logger = logging.getLogger(__name__)


@dataclass
class SpatialMatch:
    """Result of matching a GEDI point to a Sentinel-2 tile."""
    gedi_sample_id: str
    sentinel2_path: str
    latitude: float
    longitude: float
    agbd: float
    pixel_x: Optional[float] = None
    pixel_y: Optional[float] = None
    distance_to_center: Optional[float] = None
    matched: bool = True


def get_sentinel2_bounds(filepath: Union[str, Path]) -> Dict:
    """
    Read spatial metadata from a Sentinel-2 GeoTIFF.

    Returns:
        Dict with bounds, crs, transform, width, height, resolution
    """
    filepath = Path(filepath)
    with rasterio.open(filepath) as src:
        bounds = src.bounds
        return {
            "path": str(filepath),
            "crs": src.crs.to_string() if src.crs else None,
            "bounds": {
                "left": bounds.left,
                "bottom": bounds.bottom,
                "right": bounds.right,
                "top": bounds.top,
            },
            "transform": list(src.transform)[:6],
            "width": src.width,
            "height": src.height,
            "resolution": src.res,
        }


def transform_coordinates(
    lat: float,
    lon: float,
    src_crs: str = "EPSG:4326",
    dst_crs: str = "EPSG:4326",
) -> Tuple[float, float]:
    """
    Transform coordinates between CRS.

    Args:
        lat: Latitude in source CRS
        lon: Longitude in source CRS
        src_crs: Source CRS (default WGS84)
        dst_crs: Destination CRS

    Returns:
        (x, y) in destination CRS
    """
    if src_crs == dst_crs:
        return lon, lat

    x, y = transform(src_crs, dst_crs, [lon], [lat])
    return float(x[0]), float(y[0])


def point_in_bounds(
    lat: float,
    lon: float,
    bounds: Dict,
    src_crs: str = "EPSG:4326",
    dst_crs: Optional[str] = None,
) -> bool:
    """
    Check if a point falls within Sentinel-2 tile bounds.

    Args:
        lat: Latitude
        lon: Longitude
        bounds: Dict from get_sentinel2_bounds
        src_crs: Source CRS of the point
        dst_crs: Destination CRS (defaults to tile CRS)

    Returns:
        True if point is within bounds
    """
    dst_crs = dst_crs or bounds.get("crs", "EPSG:4326")

    try:
        x, y = transform_coordinates(lat, lon, src_crs, dst_crs)
    except Exception as e:
        logger.warning("Coordinate transform failed: %s", e)
        return False

    b = bounds["bounds"]
    return b["left"] <= x <= b["right"] and b["bottom"] <= y <= b["top"]


def match_gedi_to_sentinel2(
    gedi_df: pd.DataFrame,
    sentinel2_files: List[Union[str, Path]],
    src_crs: str = "EPSG:4326",
) -> List[SpatialMatch]:
    """
    Match GEDI points to Sentinel-2 tiles.

    For each GEDI record, finds the first Sentinel-2 tile whose bounds
    contain the point.

    Args:
        gedi_df: DataFrame with latitude, longitude, agbd, sample_id
        sentinel2_files: List of Sentinel-2 GeoTIFF paths
        src_crs: CRS of GEDI coordinates

    Returns:
        List of SpatialMatch objects
    """
    if gedi_df.empty:
        return []

    if not sentinel2_files:
        raise ValueError("sentinel2_files list is empty")

    tile_metadata = []
    for f in sentinel2_files:
        try:
            tile_metadata.append(get_sentinel2_bounds(f))
        except Exception as e:
            logger.warning("Failed to read Sentinel-2 metadata for %s: %s", f, e)

    if not tile_metadata:
        raise ValueError("No valid Sentinel-2 tiles found")

    matches = []
    for _, row in gedi_df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        matched = False

        for tile in tile_metadata:
            if point_in_bounds(lat, lon, tile, src_crs=src_crs, dst_crs=tile.get("crs")):
                pixel_x, pixel_y = transform_coordinates(
                    lat, lon, src_crs=src_crs, dst_crs=tile.get("crs", src_crs)
                )
                transform = rasterio.Affine(*tile["transform"])
                col = int((pixel_x - transform.c) / transform.a)
                row = int((pixel_y - transform.f) / transform.e)

                matches.append(SpatialMatch(
                    gedi_sample_id=str(row["sample_id"]),
                    sentinel2_path=tile["path"],
                    latitude=lat,
                    longitude=lon,
                    agbd=float(row["agbd"]),
                    pixel_x=float(col),
                    pixel_y=float(row),
                    matched=True,
                ))
                matched = True
                break

        if not matched:
            matches.append(SpatialMatch(
                gedi_sample_id=str(row["sample_id"]),
                sentinel2_path="",
                latitude=lat,
                longitude=lon,
                agbd=float(row["agbd"]),
                matched=False,
            ))

    matched_count = sum(1 for m in matches if m.matched)
    logger.info("Spatial matching: %d/%d GEDI points matched to Sentinel-2 tiles", matched_count, len(matches))

    return matches


def matches_to_dataframe(matches: List[SpatialMatch]) -> pd.DataFrame:
    """Convert SpatialMatch list to DataFrame."""
    if not matches:
        return pd.DataFrame(columns=[
            "sample_id", "sentinel2_path", "latitude", "longitude",
            "agbd", "pixel_x", "pixel_y", "matched"
        ])

    return pd.DataFrame([{
        "sample_id": m.gedi_sample_id,
        "sentinel2_path": m.sentinel2_path,
        "latitude": m.latitude,
        "longitude": m.longitude,
        "agbd": m.agbd,
        "pixel_x": m.pixel_x,
        "pixel_y": m.pixel_y,
        "matched": m.matched,
    } for m in matches])
