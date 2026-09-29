"""
Temporal matching between Sentinel-2 and GEDI observations.

Ensures that paired observations are within a configurable time window.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TemporalMatch:
    """Result of temporal matching."""
    sample_id: str
    sentinel2_time: Optional[str]
    gedi_time: Optional[str]
    time_diff_days: Optional[float]
    matched: bool


def parse_timestamp(ts: Optional[Union[str, datetime]]) -> Optional[datetime]:
    """Parse timestamp from string or datetime."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if pd.isna(ts):
        return None
    ts = str(ts)
    for fmt in [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y%m%d",
    ]:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def compute_temporal_difference(
    s2_time: Optional[Union[str, datetime]],
    gedi_time: Optional[Union[str, datetime]],
) -> Optional[float]:
    """Compute absolute temporal difference in days."""
    s2_dt = parse_timestamp(s2_time)
    gedi_dt = parse_timestamp(gedi_time)
    if s2_dt is None or gedi_dt is None:
        return None
    return abs((s2_dt - gedi_dt).total_seconds()) / 86400.0


def filter_by_temporal_diff(
    df: pd.DataFrame,
    s2_time_col: str = "sentinel2_time",
    gedi_time_col: str = "gedi_time",
    max_days: float = 30.0,
) -> pd.DataFrame:
    """
    Filter records by maximum temporal difference.

    Args:
        df: DataFrame with Sentinel-2 and GEDI timestamps
        s2_time_col: Column name for Sentinel-2 acquisition time
        gedi_time_col: Column name for GEDI acquisition time
        max_days: Maximum allowed temporal difference in days

    Returns:
        Filtered DataFrame with added 'time_diff_days' column
    """
    if s2_time_col not in df.columns or gedi_time_col not in df.columns:
        logger.warning("Missing timestamp columns. Skipping temporal filter.")
        df["time_diff_days"] = None
        return df

    df = df.copy()
    df["time_diff_days"] = df.apply(
        lambda row: compute_temporal_difference(
            row.get(s2_time_col), row.get(gedi_time_col)
        ),
        axis=1,
    )

    before = len(df)
    mask = df["time_diff_days"].isna() | (df["time_diff_days"] <= max_days)
    df = df[mask].copy()
    after = len(df)

    logger.info(
        "Temporal filter: %d -> %d records (max_diff=%.1f days)",
        before, after, max_days,
    )

    return df
