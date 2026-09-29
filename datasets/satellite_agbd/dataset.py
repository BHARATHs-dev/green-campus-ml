"""
Satellite AGBD dataset for PyTorch training.

Loads Sentinel-2 patches and GEDI AGBD targets.
Returns dict with 'image', 'target', and 'metadata'.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ai.preprocessing.sentinel2_preprocessor import (
    Sentinel2Normalizer,
    SENTINEL2_BANDS,
    calculate_sentinel2_indices,
)

logger = logging.getLogger(__name__)


class SatelliteAGBDataset(Dataset):
    """
    PyTorch Dataset for Sentinel-2 -> AGBD regression.

    Expects:
        - root_dir: ai/datasets/satellite_agbd/
        - split: 'train', 'val', or 'test'
        - normalizer: Sentinel2Normalizer fitted on training data
        - transform: optional torchvision transform
        - patch_size: size of patches in pixels (default 64)
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        normalizer: Optional[Sentinel2Normalizer] = None,
        transform=None,
        patch_size: int = 64,
        max_temp_diff_days: Optional[float] = 30.0,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.patch_size = patch_size
        self.max_temp_diff_days = max_temp_diff_days
        self.normalizer = normalizer
        self.transform = transform

        # Load split manifest
        split_path = self.root_dir / f"{split}.csv"
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {split_path}. "
                "Run ai.datasets.satellite_agbd.splits.create_splits() first."
            )
        self.manifest = pd.read_csv(split_path)

        if "patch_path" not in self.manifest.columns:
            raise ValueError("Manifest missing 'patch_path' column")

        if "agbd" not in self.manifest.columns:
            raise ValueError("Manifest missing 'agbd' column")

        # Apply temporal filter if configured
        if max_temp_diff_days is not None and "time_diff_days" in self.manifest.columns:
            before = len(self.manifest)
            mask = (
                self.manifest["time_diff_days"].isna() |
                (self.manifest["time_diff_days"] <= max_temp_diff_days)
            )
            self.manifest = self.manifest[mask].reset_index(drop=True)
            logger.info(
                "Temporal filter: %d -> %d samples (max_diff=%.1f days)",
                before, len(self.manifest), max_temp_diff_days,
            )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.manifest.iloc[idx]
        patch_path = Path(row["patch_path"])

        if not patch_path.exists():
            raise FileNotFoundError(f"Patch file not found: {patch_path}")

        try:
            with rasterio.open(patch_path) as src:
                band_names = src.descriptions
                bands = {}
                for i, desc in enumerate(band_names, start=1):
                    if desc:
                        bands[desc] = src.read(i)
        except Exception as e:
            raise RuntimeError(f"Failed to read patch {patch_path}: {e}")

        # Calculate indices
        indices = calculate_sentinel2_indices(bands)

        # Build feature stack
        selected_bands = [b for b in SENTINEL2_BANDS.keys() if b in bands]
        if not selected_bands:
            raise ValueError(f"No valid bands in patch {patch_path}")

        h = w = None
        arrays = []
        channel_names = []

        for band_name in selected_bands:
            arr = bands[band_name].astype(np.float32)
            if h is None:
                h, w = arr.shape
            if arr.shape != (h, w):
                continue
            arrays.append(arr)
            channel_names.append(band_name)

        for idx_name in ["ndvi", "evi", "savi", "ndwi", "msi"]:
            if idx_name in indices:
                idx_arr = indices[idx_name]
                if idx_arr.shape != (h, w):
                    continue
                arrays.append(idx_arr)
                channel_names.append(idx_name)

        tensor = np.stack(arrays, axis=-1)

        # Apply normalizer if provided
        if self.normalizer is not None:
            tensor = self.normalizer.apply(tensor)
            tensor = np.clip(tensor, -3.0, 3.0)
            tensor = (tensor + 3.0) / 6.0

        # Convert to tensor
        if tensor.ndim == 2:
            tensor = tensor[:, :, np.newaxis]

        image = torch.from_numpy(tensor).permute(2, 0, 1).float()
        target = torch.tensor(float(row["agbd"]), dtype=torch.float32)

        metadata = {
            "sample_id": str(row.get("sample_id", idx)),
            "latitude": float(row["latitude"]) if "latitude" in row and pd.notna(row["latitude"]) else None,
            "longitude": float(row["longitude"]) if "longitude" in row and pd.notna(row["longitude"]) else None,
            "crs": str(row.get("crs", "EPSG:4326")) if "crs" in row else "EPSG:4326",
            "valid_pixel_ratio": float(row["valid_pixel_ratio"]) if "valid_pixel_ratio" in row and pd.notna(row["valid_pixel_ratio"]) else None,
            "agbd_carbon": float(row["agbd_carbon"]) if "agbd_carbon" in row and pd.notna(row["agbd_carbon"]) else None,
            "channel_names": channel_names,
        }

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "target": target,
            "metadata": metadata,
        }

    @classmethod
    def fit_normalizer(
        cls,
        root_dir: Union[str, Path],
        split: str = "train",
        max_samples: Optional[int] = None,
        random_state: int = 42,
    ) -> Sentinel2Normalizer:
        """
        Fit normalization statistics on training data.

        Args:
            root_dir: Dataset root directory
            split: Split to use for fitting ('train' only recommended)
            max_samples: Maximum samples to use
            random_state: Random seed

        Returns:
            Fitted Sentinel2Normalizer
        """
        root_dir = Path(root_dir)
        split_path = root_dir / f"{split}.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {split_path}")

        df = pd.read_csv(split_path)
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)

        channel_names = list(SENTINEL2_BANDS.keys()) + ["ndvi", "evi", "savi", "ndwi", "msi"]
        normalizer = Sentinel2Normalizer(channel_names=channel_names)

        tensors = []
        for _, row in df.iterrows():
            patch_path = Path(row["patch_path"])
            if not patch_path.exists():
                continue
            try:
                with rasterio.open(patch_path) as src:
                    band_names = src.descriptions
                    bands = {}
                    for i, desc in enumerate(band_names, start=1):
                        if desc:
                            bands[desc] = src.read(i)
                indices = calculate_sentinel2_indices(bands)
                selected = [b for b in SENTINEL2_BANDS if b in bands]
                arrays = [bands[b].astype(np.float32) for b in selected]
                for idx_name in ["ndvi", "evi", "savi", "ndwi", "msi"]:
                    if idx_name in indices:
                        arrays.append(indices[idx_name].astype(np.float32))
                if arrays:
                    tensors.append(np.stack(arrays, axis=-1))
            except Exception:
                continue

        if not tensors:
            raise RuntimeError("No valid tensors found for normalizer fitting.")

        combined = np.concatenate(tensors, axis=0)
        normalizer.fit(combined)
        return normalizer
