"""
Satellite dataset utilities for AGBD regression.

Supports:
- Multi-band GeoTIFF / TIFF patches
- Labels CSV with patch_id and agbd_mgha
- Optional metadata (plot_id, latitude, longitude, date)
- Group-based splitting by plot_id
- Band selection and spectral index computation
- Train / validation / test splits
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils.spectral import apply_spectral_indices
from .sentinel2_preprocessor import (
    preprocess_sentinel2,
    DEFAULT_FEATURE_BANDS,
    REQUIRED_BANDS,
    SENTINEL2_BANDS,
    Sentinel2Normalizer,
)

REQUIRED_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
BAND_ALIASES = {
    "B2": "B2",
    "B3": "B3",
    "B4": "B4",
    "B5": "B5",
    "B6": "B6",
    "B7": "B7",
    "B8": "B8",
    "B8A": "B8A",
    "B11": "B11",
    "B12": "B12",
}
INPUT_SIZE = 224
CARBON_FACTOR = 0.47


def _read_tiff_bands(filepath: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Read a Sentinel-2 GeoTIFF using the Phase 2 preprocessor.

    Returns:
        tensor: (H, W, C) array
        bands_dict: raw band arrays
        metadata: preprocessing metadata
    """
    tensor, metadata = preprocess_sentinel2(
        filepath,
        selected_bands=DEFAULT_FEATURE_BANDS,
        include_indices=True,
        target_resolution=10,
    )
    bands_dict = {}
    for i, name in enumerate(metadata.get("channel_names", [])):
        if name in SENTINEL2_BANDS:
            bands_dict[name] = tensor[:, :, i]
    return tensor, bands_dict, metadata


def detect_bands(bands: Dict[str, np.ndarray], required: List[str]) -> List[str]:
    """
    Heuristically map raw band arrays to Sentinel-2 band names.
    For now, returns whatever band keys are present.
    """
    available = list(bands.keys())
    return available


def validate_bands(bands: Dict[str, np.ndarray], required: List[str]) -> Tuple[bool, List[str]]:
    """
    Check whether all required bands are present.

    Returns (is_valid, missing_bands).
    """
    available = set(bands.keys())
    required_set = set(required)
    missing = sorted(required_set - available)
    return len(missing) == 0, missing


def build_band_tensor(
    bands: Dict[str, np.ndarray],
    selected_bands: List[str],
    include_indices: bool = True,
    normalizer: Optional[Sentinel2Normalizer] = None,
) -> np.ndarray:
    """
    Build a (H, W, C) tensor from selected raw bands and optionally
    append computed spectral indices.

    If a selected band is missing, it is skipped with a warning.
    """
    h = w = None
    arrays = []

    for band_name in selected_bands:
        if band_name not in bands:
            continue
        arr = bands[band_name].astype(np.float32)
        if h is None:
            h, w = arr.shape
        if arr.shape != (h, w):
            raise ValueError(
                f"Band {band_name} has shape {arr.shape}, expected ({h}, {w})"
            )
        arrays.append(arr)

    if not arrays:
        raise ValueError("No valid bands available for tensor construction.")

    if include_indices:
        index_dict = apply_spectral_indices(bands)
        for idx_name, idx_arr in index_dict.items():
            if idx_arr.shape != (h, w):
                continue
            arrays.append(idx_arr)

    tensor = np.stack(arrays, axis=-1)

    if normalizer is not None:
        tensor = normalizer.apply(tensor)
        tensor = np.clip(tensor, -3.0, 3.0)
        tensor = (tensor + 3.0) / 6.0
    else:
        raise RuntimeError(
            "Per-image min-max normalization is not supported in production. "
            "Provide a fitted Sentinel2Normalizer."
        )

    return tensor


class SatelliteDataset(Dataset):
    """
    PyTorch Dataset for satellite patch -> AGBD regression.

    Expects:
      - root_dir: directory containing patch TIFF files
      - labels_csv: path to CSV with at least patch_id and agbd_mgha
      - transform: optional torchvision transform
      - bands: list of band names to use
      - normalizer: optional Sentinel2Normalizer for deterministic normalization
      - return_dict: if True, return dict with 'image', 'target', 'metadata'
    """

    def __init__(
        self,
        root_dir: str,
        labels_csv: str,
        transform=None,
        bands: Optional[List[str]] = None,
        normalizer: Optional[Sentinel2Normalizer] = None,
        return_dict: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.bands = bands if bands is not None else REQUIRED_BANDS.copy()
        self.normalizer = normalizer
        self.return_dict = return_dict

        if isinstance(labels_csv, (str, Path)):
            self.labels_df = pd.read_csv(labels_csv)
        else:
            self.labels_df = labels_csv.copy()

        if "patch_id" not in self.labels_df.columns:
            raise ValueError("labels CSV must contain a 'patch_id' column")

        if "agbd_mgha" not in self.labels_df.columns:
            if "agb_kg" in self.labels_df.columns:
                self.labels_df["agbd_mgha"] = self.labels_df["agb_kg"] / CARBON_FACTOR
            else:
                raise ValueError("labels CSV must contain 'agbd_mgha' or 'agb_kg'")

        self.labels_df = self.labels_df.reset_index(drop=True)
        self.transform = transform
        self.plot_id_col = "plot_id" if "plot_id" in self.labels_df.columns else None

    @classmethod
    def fit_normalizer(
        cls,
        root_dir: str,
        labels_csv: str,
        bands: Optional[List[str]] = None,
        sample_size: Optional[int] = None,
        random_state: int = 42,
    ) -> Sentinel2Normalizer:
        """
        Fit normalization statistics from training data.

        Args:
            root_dir: directory containing patch TIFF files
            labels_csv: path to CSV with patch_id
            bands: list of band names to use
            sample_size: maximum number of samples to use for fitting
            random_state: random seed for sampling

        Returns:
            Fitted Sentinel2Normalizer
        """
        if bands is None:
            bands = DEFAULT_FEATURE_BANDS.copy()

        df = pd.read_csv(labels_csv) if isinstance(labels_csv, (str, Path)) else labels_csv.copy()
        if sample_size is not None and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

        normalizer = Sentinel2Normalizer(channel_names=bands)

        tensors = []
        for _, row in df.iterrows():
            patch_id = str(row["patch_id"])
            patch_path = Path(root_dir) / f"{patch_id}.tif"
            if not patch_path.exists():
                continue
            try:
                tensor, _ = preprocess_sentinel2(
                    patch_path,
                    selected_bands=bands,
                    include_indices=True,
                    target_resolution=10,
                    normalizer=None,
                )
                tensors.append(tensor)
            except Exception:
                continue

        if not tensors:
            raise RuntimeError("No valid tensors found for normalizer fitting.")

        combined = np.concatenate(tensors, axis=0)
        normalizer.fit(combined)
        return normalizer

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int):
        row = self.labels_df.iloc[idx]
        patch_id = str(row["patch_id"])
        patch_path = self.root_dir / f"{patch_id}.tif"

        if not patch_path.exists():
            raise FileNotFoundError(f"Patch file not found: {patch_path}")

        arr, bands_dict, preprocessing_meta = _read_tiff_bands(patch_path)

        try:
            tensor = build_band_tensor(
                bands_dict, self.bands, include_indices=True, normalizer=self.normalizer
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build tensor for {patch_id}: {e}")

        if tensor.ndim == 2:
            tensor = tensor[:, :, np.newaxis]

        label = float(row["agbd_mgha"])
        plot_id = str(row[self.plot_id_col]) if self.plot_id_col else None

        if self.transform:
            tensor = self.transform(tensor)

        if self.return_dict:
            metadata = {
                "sample_id": patch_id,
                "plot_id": plot_id,
                "agbd": label,
                "agbd_carbon": label * CARBON_FACTOR,
                "patch_path": str(patch_path),
                "preprocessing": preprocessing_meta,
            }
            return {
                "image": tensor,
                "target": torch.tensor(label, dtype=torch.float32),
                "metadata": metadata,
            }

        return tensor, torch.tensor(label, dtype=torch.float32), plot_id


def get_satellite_transforms(input_size: int = INPUT_SIZE, in_channels: int = 15):
    """
    Return train/val/test transforms.

    Train: resize + random crop + flip + rotation + colour jitter + normalisation
    Val/Test: deterministic resize + normalisation
    """
    normalize = transforms.Normalize(mean=[0.5] * in_channels, std=[0.5] * in_channels)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((input_size + 32, input_size + 32)),
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((input_size, input_size)),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((input_size, input_size)),
        normalize,
    ])

    return {
        "train": train_transform,
        "val": val_transform,
        "test": test_transform,
    }


def create_splits(
    labels_csv: str,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split labels into train / validation / test.

    If plot_id is available, uses GroupShuffleSplit to avoid spatial leakage.
    Otherwise falls back to random split.
    """
    df = pd.read_csv(labels_csv) if isinstance(labels_csv, str) else labels_csv.copy()

    if "plot_id" in df.columns and df["plot_id"].nunique() > 1:
        gss = GroupShuffleSplit(
            n_splits=1,
            train_size=train_frac + val_frac,
            random_state=random_state,
        )
        train_val_idx, test_idx = next(gss.split(df, groups=df["plot_id"]))
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        val_frac_of_train = val_frac / (train_frac + val_frac)
        gss_val = GroupShuffleSplit(
            n_splits=1,
            train_size=1 - val_frac_of_train,
            random_state=random_state,
        )
        train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df["plot_id"]))
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    else:
        train_df, temp_df = train_test_split(df, test_size=1 - train_frac, random_state=random_state)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state)

    return train_df, val_df, test_df


def validate_dataset(labels_csv: str, root_dir: str, bands: List[str]) -> Dict:
    """
    Run data validation checks on the satellite dataset.

    Returns a validation report dict.
    """
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "summary": {},
    }

    root = Path(root_dir)
    if not root.exists():
        report["valid"] = False
        report["errors"].append(f"Dataset directory not found: {root_dir}")
        return report

    labels_path = Path(labels_csv)
    if not labels_path.exists():
        report["valid"] = False
        report["errors"].append(f"Labels file not found: {labels_csv}")
        return report

    try:
        df = pd.read_csv(labels_path)
    except Exception as e:
        report["valid"] = False
        report["errors"].append(f"Failed to read labels CSV: {e}")
        return report

    if "patch_id" not in df.columns:
        report["valid"] = False
        report["errors"].append("labels CSV missing 'patch_id' column")
        return report

    if "agbd_mgha" not in df.columns and "agb_kg" not in df.columns:
        report["valid"] = False
        report["errors"].append("labels CSV missing 'agbd_mgha' or 'agb_kg' column")
        return report

    report["summary"]["total_patches"] = len(df)

    missing_files = []
    corrupt_files = []
    nodata_warnings = []

    for patch_id in df["patch_id"]:
        patch_path = root / f"{patch_id}.tif"
        if not patch_path.exists():
            missing_files.append(str(patch_id))
            continue
        try:
            with rasterio.open(patch_path) as src:
                src.read(1)
        except Exception:
            corrupt_files.append(str(patch_id))

    if missing_files:
        report["errors"].append(f"{len(missing_files)} patch files missing")
        report["valid"] = False

    if corrupt_files:
        report["errors"].append(f"{len(corrupt_files)} corrupt TIFF files")
        report["valid"] = False

    if df["patch_id"].duplicated().any():
        dupes = df[df["patch_id"].duplicated()]["patch_id"].tolist()
        report["errors"].append(f"Duplicate patch IDs: {dupes}")
        report["valid"] = False

    agb_col = "agbd_mgha" if "agbd_mgha" in df.columns else "agb_kg"
    if pd.isnull(df[agb_col]).any():
        report["errors"].append("Missing AGB values in labels")
        report["valid"] = False

    if (df[agb_col] < 0).any():
        report["warnings"].append("Negative AGB values detected")

    report["summary"]["missing_files"] = len(missing_files)
    report["summary"]["corrupt_files"] = len(corrupt_files)

    return report
