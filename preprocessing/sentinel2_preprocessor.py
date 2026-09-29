"""
Sentinel-2 preprocessing and feature generation.

Phase 2 implementation:
- Harmonize bands to common 10 m spatial resolution
- Apply reflectance scaling from product/raster metadata
- Handle NoData, NaN, and infinite values safely
- Calculate NDVI, EVI, SAVI, NDWI, MSI
- Build configurable 15-channel feature stack
- Return preprocessing metadata
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio import Affine

from utils.spectral import (
    apply_spectral_indices,
)

logger = logging.getLogger(__name__)


SENTINEL2_BANDS = {
    "B2": {"native_resolution": 10},
    "B3": {"native_resolution": 10},
    "B4": {"native_resolution": 10},
    "B5": {"native_resolution": 20},
    "B6": {"native_resolution": 20},
    "B7": {"native_resolution": 20},
    "B8": {"native_resolution": 10},
    "B8A": {"native_resolution": 20},
    "B11": {"native_resolution": 20},
    "B12": {"native_resolution": 20},
}

DEFAULT_FEATURE_BANDS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
]
REQUIRED_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]

ALL_BAND_NAMES = list(SENTINEL2_BANDS.keys())
INDEX_NAMES = ["ndvi", "evi", "savi", "ndwi", "msi"]


def _identify_bands(src: rasterio.DatasetReader, filepath: Path) -> Dict[str, int]:
    """Map rasterio band indices to Sentinel-2 band names."""
    band_map: Dict[str, int] = {}
    descriptions = src.descriptions

    # Sort band names by descending length so B8A matches before B8
    sorted_band_names = sorted(ALL_BAND_NAMES, key=len, reverse=True)

    if descriptions and any(descriptions):
        for idx, desc in enumerate(descriptions, start=1):
            if desc:
                desc_upper = str(desc).upper().strip()
                for band_name in sorted_band_names:
                    if band_name in desc_upper:
                        band_map[band_name] = idx
                        break

    if not band_map:
        tags = src.tags()
        for key, value in tags.items():
            key_upper = str(key).upper()
            for band_name in sorted_band_names:
                if band_name in key_upper:
                    try:
                        band_map[band_name] = int(value)
                    except (ValueError, TypeError):
                        pass
                    break

    if not band_map:
        for i in range(1, src.count + 1):
            band_map[f"band_{i}"] = i

    # Fallback for standard 3-band RGB imagery stored in GeoTIFF
    if len([b for b in band_map.keys() if b in SENTINEL2_BANDS]) < 3 and src.count >= 3:
        band_map["B4"] = 1  # Red
        band_map["B3"] = 2  # Green
        band_map["B2"] = 3  # Blue

    return band_map


def _detect_reflectance_scaling(
    metadata: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float]]:
    """Detect reflectance scaling factor and offset from metadata."""
    tags = metadata.get("tags", {})

    quantification = None
    for key in [
        "QUANTIFICATION_VALUE",
        "QUANTIFICATIONVALUE",
        "QUANTIFICATION",
    ]:
        if key in tags:
            try:
                quantification = float(tags[key])
                break
            except (ValueError, TypeError):
                pass

    offset = None
    for key in ["BOA_ADD_OFFSET", "BOAADDOFFSET", "OFFSET"]:
        if key in tags:
            try:
                offset = float(tags[key])
                break
            except (ValueError, TypeError):
                pass

    return quantification, offset


def _apply_reflectance_scaling(
    bands_dict: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Apply reflectance scaling based on metadata."""
    quantification, offset = _detect_reflectance_scaling(metadata)

    if quantification is None:
        logger.info(
            "No reflectance scaling metadata found. "
            "Returning raw pixel values without scaling."
        )
        metadata["reflectance_scaling_applied"] = False
        metadata["reflectance_quantification"] = None
        metadata["reflectance_offset"] = None
        return bands_dict, metadata

    offset = offset if offset is not None else 0.0

    scaled_bands = {}
    for band_name, arr in bands_dict.items():
        scaled = (arr.astype(np.float32) - offset) / quantification
        scaled = np.clip(scaled, 0.0, 1.0)
        scaled_bands[band_name] = scaled

    metadata["reflectance_scaling_applied"] = True
    metadata["reflectance_quantification"] = quantification
    metadata["reflectance_offset"] = offset

    logger.info(
        "Applied reflectance scaling: factor=%s, offset=%s",
        quantification,
        offset,
    )

    return scaled_bands, metadata


def _mask_invalid_pixels(
    bands_dict: Dict[str, np.ndarray],
    nodata_value: Optional[float],
) -> Dict[str, np.ndarray]:
    """Mask NoData, NaN, and infinite values."""
    masked = {}
    for band_name, arr in bands_dict.items():
        arr = arr.copy()
        if nodata_value is not None:
            mask = np.isclose(arr, nodata_value, equal_nan=True)
            arr[mask] = np.nan

        mask = ~np.isfinite(arr)
        arr[mask] = np.nan
        masked[band_name] = arr
    return masked


def _resample_to_target(
    src: rasterio.DatasetReader,
    band_map: Dict[str, int],
    target_resolution: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Resample bands to target resolution, ensuring all bands share the exact same spatial dimensions."""
    target_bands: Dict[str, np.ndarray] = {}
    resampling_info: Dict[str, Any] = {}

    target_height = src.height
    target_width = src.width

    for band_name, band_idx in band_map.items():
        if band_name not in SENTINEL2_BANDS:
            continue

        native_res = SENTINEL2_BANDS[band_name]["native_resolution"]
        arr = src.read(band_idx).astype(np.float32)

        if arr.shape != (target_height, target_width):
            import torch
            import torch.nn.functional as F
            tensor_arr = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
            resized = F.interpolate(tensor_arr, size=(target_height, target_width), mode='bilinear', align_corners=False)
            arr = resized.squeeze(0).squeeze(0).numpy()
            resampled = True
        else:
            resampled = False

        target_bands[band_name] = arr
        resampling_info[band_name] = {
            "native_resolution": native_res,
            "resampled": resampled,
        }

    return target_bands, resampling_info


def _compute_valid_pixel_ratio(bands_dict: Dict[str, np.ndarray]) -> float:
    """Compute ratio of valid (non-NaN) pixels."""
    total_pixels = 0
    valid_pixels = 0

    for arr in bands_dict.values():
        total_pixels += arr.size
        valid_pixels += int(np.sum(np.isfinite(arr)))

    if total_pixels == 0:
        return 0.0
    return float(valid_pixels) / float(total_pixels)


def calculate_sentinel2_indices(bands_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Calculate Sentinel-2 spectral indices."""
    return apply_spectral_indices(bands_dict)


def build_sentinel2_feature_stack(
    bands_dict: Dict[str, np.ndarray],
    indices_dict: Dict[str, np.ndarray],
    selected_bands: List[str],
    include_indices: bool = True,
    normalizer: Optional["Sentinel2Normalizer"] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Build (H, W, C) feature stack from bands and indices."""
    h = w = None
    arrays = []
    channel_names = []

    for band_name in selected_bands:
        if band_name not in bands_dict:
            continue
        arr = bands_dict[band_name].astype(np.float32)
        if h is None:
            h, w = arr.shape
        if arr.shape != (h, w):
            raise ValueError(
                f"Band {band_name} has shape {arr.shape}, expected ({h}, {w})"
            )
        arrays.append(arr)
        channel_names.append(band_name)

    if not arrays:
        raise ValueError("No valid bands available for feature stack.")

    if include_indices:
        for idx_name in INDEX_NAMES:
            if idx_name in indices_dict:
                idx_arr = indices_dict[idx_name]
                if idx_arr.shape != (h, w):
                    continue
                arrays.append(idx_arr)
                channel_names.append(idx_name)

    tensor = np.stack(arrays, axis=-1)

    if normalizer is not None:
        tensor = normalizer.apply(tensor)
        tensor = np.clip(tensor, -3.0, 3.0)
        tensor = (tensor + 3.0) / 6.0

    return tensor, channel_names


def preprocess_sentinel2(
    filepath: Union[str, Path],
    selected_bands: Optional[List[str]] = None,
    include_indices: bool = True,
    target_resolution: int = 10,
    normalizer: Optional["Sentinel2Normalizer"] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Full Sentinel-2 preprocessing pipeline.

    Returns:
        tensor: (H, W, C) numpy array
        metadata: preprocessing metadata dict
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {filepath}")

    if selected_bands is None:
        selected_bands = DEFAULT_FEATURE_BANDS.copy()

    metadata: Dict[str, Any] = {
        "file": filepath.name,
        "selected_bands": selected_bands,
        "include_indices": include_indices,
        "target_resolution": target_resolution,
        "original_resolution": {},
        "final_resolution": target_resolution,
        "resampling_info": {},
        "valid_pixel_ratio": None,
        "crs": None,
        "bounds": None,
        "transform": None,
        "shape": None,
        "reflectance_scaling_applied": False,
        "reflectance_quantification": None,
        "reflectance_offset": None,
        "channel_names": [],
        "errors": [],
        "warnings": [],
    }

    try:
        with rasterio.open(filepath) as src:
            metadata["crs"] = src.crs.to_string() if src.crs else None
            metadata["bounds"] = list(src.bounds)
            metadata["shape"] = [src.height, src.width]
            metadata["nodata"] = src.nodata
            metadata["tags"] = dict(src.tags())

            band_map = _identify_bands(src, filepath)
            if not band_map:
                raise ValueError(
                    "Could not identify any Sentinel-2 bands in the GeoTIFF."
                )

            missing = [b for b in REQUIRED_BANDS if b not in band_map]
            if missing:
                raise ValueError(
                    f"Missing required Sentinel-2 bands: {missing}. "
                    f"Found bands: {sorted(band_map.keys())}"
                )

            for band_name in selected_bands:
                if band_name not in band_map:
                    metadata["warnings"].append(
                        f"Selected band {band_name} not found in file."
                    )
                    continue
                native_res = SENTINEL2_BANDS.get(band_name, {}).get(
                    "native_resolution", 10
                )
                metadata["original_resolution"][band_name] = native_res

            bands_dict, resampling_info = _resample_to_target(
                src, band_map, target_resolution
            )
            metadata["resampling_info"] = resampling_info

            if target_resolution == 10:
                target_transform = Affine(
                    src.transform.a / 2,
                    src.transform.b,
                    src.transform.c,
                    src.transform.d,
                    src.transform.e / 2,
                    src.transform.f,
                )
            else:
                target_transform = src.transform
            metadata["transform"] = list(target_transform)[:6]

    except Exception as e:
        metadata["errors"].append(f"Failed to read GeoTIFF: {e}")
        raise ValueError(f"GeoTIFF reading failed: {e}")

    bands_dict = _mask_invalid_pixels(bands_dict, metadata.get("nodata"))
    bands_dict, metadata = _apply_reflectance_scaling(bands_dict, metadata)

    metadata["valid_pixel_ratio"] = _compute_valid_pixel_ratio(bands_dict)

    indices_dict = calculate_sentinel2_indices(bands_dict) if include_indices else {}

    valid_bands = [b for b in selected_bands if b in bands_dict]
    if not valid_bands:
        raise ValueError("No valid bands found after preprocessing.")

    tensor, channel_names = build_sentinel2_feature_stack(
        bands_dict, indices_dict, valid_bands, include_indices, normalizer=normalizer
    )
    metadata["channel_names"] = channel_names

    return tensor, metadata


class Sentinel2Normalizer:
    """Production-safe normalization for Sentinel-2 feature stacks."""

    def __init__(
        self,
        channel_names: List[str],
        normalization_version: str = "sentinel2-normalization-v1",
    ):
        self.channel_names = list(channel_names)
        self.normalization_version = normalization_version
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, tensor: np.ndarray) -> None:
        """Compute per-channel mean and std from training tensor."""
        if tensor.ndim != 3:
            raise ValueError(f"Expected (H, W, C) tensor, got shape {tensor.shape}")

        h, w, c = tensor.shape
        if c != len(self.channel_names):
            raise ValueError(
                f"Tensor channels {c} != channel_names count {len(self.channel_names)}"
            )

        pixels = tensor.reshape(-1, c).astype(np.float64)

        self.mean = np.zeros(c, dtype=np.float64)
        self.std = np.zeros(c, dtype=np.float64)

        for i in range(c):
            channel_data = pixels[:, i]
            valid = channel_data[np.isfinite(channel_data)]
            if valid.size == 0:
                self.mean[i] = 0.0
                self.std[i] = 1.0
            else:
                self.mean[i] = valid.mean()
                self.std[i] = valid.std()
                if self.std[i] < 1e-10:
                    self.std[i] = 1.0

    def apply(self, tensor: np.ndarray) -> np.ndarray:
        """Apply normalization using fitted statistics."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer has not been fitted. Call fit() first.")

        if tensor.ndim != 3:
            raise ValueError(f"Expected (H, W, C) tensor, got shape {tensor.shape}")

        h, w, c = tensor.shape
        if c != len(self.channel_names):
            raise ValueError(
                f"Tensor channels {c} != channel_names count {len(self.channel_names)}"
            )

        tensor = tensor.copy().astype(np.float64)
        for i in range(c):
            tensor[:, :, i] = (tensor[:, :, i] - self.mean[i]) / self.std[i]

        return tensor.astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize normalization metadata to a dict."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer has not been fitted.")

        return {
            "channel_names": self.channel_names,
            "normalization_version": self.normalization_version,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sentinel2Normalizer":
        """Load normalization metadata from a dict."""
        instance = cls(
            channel_names=data["channel_names"],
            normalization_version=data.get("normalization_version", "v1"),
        )
        instance.mean = np.array(data["mean"], dtype=np.float64)
        instance.std = np.array(data["std"], dtype=np.float64)
        return instance

    def save(self, path: Union[str, Path]) -> None:
        """Save normalization metadata to JSON."""
        import json

        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Sentinel2Normalizer":
        """Load normalization metadata from JSON."""
        import json

        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
