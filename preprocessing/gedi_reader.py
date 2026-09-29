"""
GEDI AGBD data reader.

Supports:
- CSV files with GEDI Level-4A AGBD data
- Flexible column name mapping
- Quality flag filtering
- Temporal parsing
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GEDIRecord:
    """Single GEDI AGBD observation."""
    sample_id: str
    latitude: float
    longitude: float
    agbd: float
    agbd_uncertainty: Optional[float] = None
    quality_flag: Optional[int] = None
    acquisition_time: Optional[str] = None
    beam: Optional[str] = None
    crs: str = "EPSG:4326"


def read_gedi_csv(
    filepath: Union[str, Path],
    lat_col: str = "lat_lowestmode",
    lon_col: str = "lon_lowestmode",
    agbd_col: str = "agbd",
    quality_col: Optional[str] = "quality_class",
    time_col: Optional[str] = "time_start",
    beam_col: Optional[str] = "beam",
    uncertainty_col: Optional[str] = "agbd_se",
) -> pd.DataFrame:
    """
    Read GEDI AGBD data from a CSV file.

    Args:
        filepath: Path to CSV file
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        agbd_col: Column name for AGBD (Mg/ha)
        quality_col: Column name for quality flag (optional)
        time_col: Column name for acquisition time (optional)
        beam_col: Column name for beam identifier (optional)
        uncertainty_col: Column name for AGBD standard error (optional)

    Returns:
        DataFrame with standardized column names
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"GEDI CSV not found: {filepath}")

    df = pd.read_csv(filepath)

    required = [lat_col, lon_col, agbd_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in GEDI data. Found: {list(df.columns)}")

    result = pd.DataFrame({
        "sample_id": df.index.astype(str),
        "latitude": pd.to_numeric(df[lat_col], errors="coerce"),
        "longitude": pd.to_numeric(df[lon_col], errors="coerce"),
        "agbd": pd.to_numeric(df[agbd_col], errors="coerce"),
        "agbd_uncertainty": pd.to_numeric(df[uncertainty_col], errors="coerce") if uncertainty_col and uncertainty_col in df.columns else None,
        "quality_flag": pd.to_numeric(df[quality_col], errors="coerce").astype("Int64") if quality_col and quality_col in df.columns else None,
        "acquisition_time": df[time_col] if time_col and time_col in df.columns else None,
        "beam": df[beam_col] if beam_col and beam_col in df.columns else None,
        "crs": "EPSG:4326",
    })

    return result


def filter_gedi_records(
    df: pd.DataFrame,
    min_agbd: float = 0.0,
    max_agbd: float = 500.0,
    require_quality_flag: bool = False,
    valid_quality_values: Optional[List[int]] = None,
    max_agbd_uncertainty: Optional[float] = None,
) -> pd.DataFrame:
    """
    Apply quality filters to GEDI records.

    Args:
        df: DataFrame from read_gedi_csv
        min_agbd: Minimum valid AGBD (Mg/ha)
        max_agbd: Maximum valid AGBD (Mg/ha)
        require_quality_flag: If True, drop records with missing quality flag
        valid_quality_values: Allowed quality flag values (if None, all values accepted)
        max_agbd_uncertainty: Maximum allowed standard error (Mg/ha)

    Returns:
        Filtered DataFrame
    """
    original_count = len(df)
    stats = {"original_count": original_count}

    # Filter by AGBD range
    mask = (df["agbd"] >= min_agbd) & (df["agbd"] <= max_agbd)
    stats["agbd_range_filtered"] = int((~mask).sum())
    df = df[mask].copy()

    # Filter by quality flag
    if require_quality_flag:
        mask = df["quality_flag"].notna()
        stats["missing_quality_filtered"] = int((~mask).sum())
        df = df[mask].copy()

    if valid_quality_values is not None and "quality_flag" in df.columns:
        mask = df["quality_flag"].isin(valid_quality_values)
        stats["invalid_quality_filtered"] = int((~mask).sum())
        df = df[mask].copy()

    # Filter by uncertainty
    if max_agbd_uncertainty is not None and "agbd_uncertainty" in df.columns:
        mask = (df["agbd_uncertainty"].isna()) | (df["agbd_uncertainty"] <= max_agbd_uncertainty)
        stats["uncertainty_filtered"] = int((~mask).sum())
        df = df[mask].copy()

    # Remove records with missing coordinates
    mask = df["latitude"].notna() & df["longitude"].notna()
    stats["missing_coords_filtered"] = int((~mask).sum())
    df = df[mask].copy()

    stats["final_count"] = len(df)
    stats["filtered_count"] = original_count - len(df)

    logger.info(
        "GEDI filtering: %d -> %d records (filtered %d)",
        original_count,
        len(df),
        stats["filtered_count"],
    )

    return df, stats
