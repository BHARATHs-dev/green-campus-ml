"""
Train/validation/test splitting for Sentinel-2 + GEDI dataset.

Uses group-aware splitting based on spatial tiles/sites to avoid
spatial leakage between splits.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

logger = logging.getLogger(__name__)


def create_splits(
    manifest_df: pd.DataFrame,
    output_dir: Union[str, Path],
    group_col: str = "tile_id",
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    random_state: int = 42,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation/test splits with group awareness.

    Uses GroupShuffleSplit when group_col is present and has multiple
    unique values. Falls back to random split otherwise.

    Args:
        manifest_df: DataFrame with samples
        output_dir: Directory to save split CSVs
        group_col: Column to group by for spatial splitting
        train_frac: Fraction for training
        val_frac: Fraction for validation
        random_state: Random seed
        save: If True, save splits to CSV

    Returns:
        (train_df, val_df, test_df)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if group_col in manifest_df.columns and manifest_df[group_col].nunique() > 1:
        logger.info("Using group-aware splitting on '%s'", group_col)
        gss = GroupShuffleSplit(
            n_splits=1,
            train_size=train_frac + val_frac,
            random_state=random_state,
        )
        train_val_idx, test_idx = next(gss.split(manifest_df, groups=manifest_df[group_col]))
        train_val_df = manifest_df.iloc[train_val_idx].reset_index(drop=True)
        test_df = manifest_df.iloc[test_idx].reset_index(drop=True)

        val_frac_of_train = val_frac / (train_frac + val_frac)
        gss_val = GroupShuffleSplit(
            n_splits=1,
            train_size=1 - val_frac_of_train,
            random_state=random_state,
        )
        train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df[group_col]))
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    else:
        logger.info("Using random splitting")
        train_df, temp_df = train_test_split(
            manifest_df, test_size=1 - train_frac, random_state=random_state
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=random_state
        )

    if save:
        train_df.to_csv(output_dir / "train.csv", index=False)
        val_df.to_csv(output_dir / "val.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)
        logger.info(
            "Saved splits: train=%d, val=%d, test=%d",
            len(train_df), len(val_df), len(test_df),
        )

    return train_df, val_df, test_df


def load_split(
    root_dir: Union[str, Path],
    split: str = "train",
) -> pd.DataFrame:
    """
    Load a split CSV.

    Args:
        root_dir: Dataset root directory
        split: One of 'train', 'val', 'test'

    Returns:
        DataFrame with split samples
    """
    root_dir = Path(root_dir)
    csv_path = root_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")
    return pd.read_csv(csv_path)
