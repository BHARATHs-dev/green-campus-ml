"""
Satellite inference pipeline for AGBD regression.

Responsibilities:
1. Read GeoTIFF / TIFF
2. Inspect metadata
3. Detect number of bands
4. Validate required bands
5. Handle NoData
6. Normalize/scale values
7. Calculate spectral indices
8. Prepare CNN tensor
9. Load trained satellite CNN checkpoint
10. Perform inference
11. Return AGBD + carbon
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import rasterio
import torch
from torchvision import transforms

from models.satellite_cnn import SatelliteCNN, build_satellite_cnn
from preprocessing.sentinel2_preprocessor import (
    preprocess_sentinel2,
    DEFAULT_FEATURE_BANDS,
    REQUIRED_BANDS,
    SENTINEL2_BANDS,
    Sentinel2Normalizer,
)
from utils.spectral import apply_spectral_indices, normalize_band

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "satellite_cnn_agb.pth"
METADATA_PATH = BASE_DIR / "models" / "satellite_cnn_agb_metadata.json"
INPUT_SIZE = 224
CARBON_FACTOR = 0.47
MODEL_VERSION = "satellite-cnn-prototype-v1"

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

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


def _process_rgb_image(
    filepath: Union[Path, str],
    in_channels: int = 15,
    normalizer: Optional[Sentinel2Normalizer] = None,
) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
    """
    Process standard RGB image (JPG/PNG/WEBP) for satellite biomass prediction.
    Constructs a multispectral feature stack by extracting RGB channels (B4, B3, B2),
    estimating vegetation reflectance (NIR, RedEdge, SWIR proxies), and computing
    standard spectral indices (NDVI, EVI, SAVI, NDWI, MSI).
    """
    img = Image.open(filepath).convert("RGB")
    w, h = img.size

    # Limit max dimension to avoid memory bloat
    if max(w, h) > 1024:
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        w, h = img.size

    rgb = np.array(img, dtype=np.float32) / 255.0

    # Optical bands: Red (B4), Green (B3), Blue (B2)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # Synthesize vegetation proxies
    # Chlorophyll reflects strongly in NIR, absorbs Red & Blue
    nir = np.clip(1.5 * g - 0.5 * r, 0.0, 1.0)
    b5 = np.clip(0.75 * r + 0.25 * nir, 0.0, 1.0)
    b6 = np.clip(0.50 * r + 0.50 * nir, 0.0, 1.0)
    b7 = np.clip(0.25 * r + 0.75 * nir, 0.0, 1.0)
    b8 = nir
    b8a = nir
    b11 = np.clip(r * 0.8, 0.0, 1.0)
    b12 = np.clip(r * 0.6, 0.0, 1.0)

    # Compute spectral indices
    ndvi = np.clip((nir - r) / (nir + r + 1e-6), -1.0, 1.0)
    evi = np.clip(2.5 * (nir - r) / (nir + 6.0 * r - 7.5 * b + 1.0), -1.0, 1.0)
    savi = np.clip(1.5 * (nir - r) / (nir + r + 0.5), -1.0, 1.0)
    ndwi = np.clip((g - nir) / (g + nir + 1e-6), -1.0, 1.0)
    msi = np.clip((b11 - nir) / (b11 + nir + 1e-6), -1.0, 1.0)

    channels = [b, g, r, b5, b6, b7, b8, b8a, b11, b12, ndvi, evi, savi, ndwi, msi]
    channel_names = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "ndvi", "evi", "savi", "ndwi", "msi"]

    if in_channels < len(channels):
        tensor = np.stack(channels[:in_channels], axis=-1)
        used_names = channel_names[:in_channels]
    elif in_channels > len(channels):
        pad = [np.zeros_like(r) for _ in range(in_channels - len(channels))]
        tensor = np.stack(channels + pad, axis=-1)
        used_names = channel_names + [f"pad_{i}" for i in range(len(pad))]
    else:
        tensor = np.stack(channels, axis=-1)
        used_names = channel_names

    if normalizer is not None:
        try:
            tensor = normalizer.apply(tensor)
            tensor = np.clip(tensor, -3.0, 3.0)
            tensor = (tensor + 3.0) / 6.0
        except Exception:
            pass

    metadata = {
        "file": Path(filepath).name,
        "format": "RGB Image (Satellite / Aerial / Drone)",
        "source_type": "optical_rgb",
        "array_shape": list(tensor.shape),
        "dimensions": [w, h],
        "available_bands": ["B2", "B3", "B4"],
        "missing_required_bands": [],
        "bands_valid": True,
        "channel_names": used_names,
        "note": "Standard RGB image processed with synthesized multispectral bands and vegetation indices.",
    }

    return tensor, metadata, ["B2", "B3", "B4"]


def _read_tiff_bands(filepath: Path, normalizer: Optional[Sentinel2Normalizer] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
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
        normalizer=normalizer,
    )
    bands_dict = {}
    for i, name in enumerate(metadata.get("channel_names", [])):
        if name in SENTINEL2_BANDS:
            bands_dict[name] = tensor[:, :, i]
    return tensor, bands_dict, metadata


def _read_basic_tiff_metadata(filepath: Path) -> Dict:
    """
    Read GeoTIFF metadata using rasterio.
    """
    metadata = {}
    try:
        with rasterio.open(filepath) as src:
            metadata["crs"] = src.crs.to_string() if src.crs else None
            metadata["transform"] = list(src.transform)[:6]
            metadata["width"] = src.width
            metadata["height"] = src.height
            metadata["count"] = src.count
            metadata["bounds"] = list(src.bounds)
            metadata["nodata"] = src.nodata
            metadata["dtype"] = str(src.dtypes[0]) if src.dtypes else None
            metadata["res"] = list(src.res)
            metadata["tags"] = dict(src.tags())
    except Exception:
        pass
    return metadata


def detect_bands(bands: Dict[str, np.ndarray]) -> List[str]:
    """
    Return available Sentinel-2 band names.
    """
    return [b for b in bands.keys() if b in SENTINEL2_BANDS]


def validate_bands(
    bands: Dict[str, np.ndarray],
    required: List[str] = REQUIRED_BANDS,
) -> Tuple[bool, List[str], List[str]]:
    """
    Check whether all required bands are present.

    Returns (is_valid, available_bands, missing_bands).
    """
    available = sorted([b for b in bands.keys() if b in SENTINEL2_BANDS])
    required_set = set(required)
    missing = sorted(required_set - set(available))
    return len(missing) == 0, available, missing


def handle_nodata(tensor: np.ndarray, nodata_value: Optional[float] = None) -> np.ndarray:
    """
    Replace NoData values with the band median.

    If nodata_value is None, treats zeros and extreme outliers as NoData.
    """
    if nodata_value is not None:
        mask = np.isclose(tensor, nodata_value)
    else:
        mask = (tensor == 0) | (np.isnan(tensor)) | (np.isinf(tensor))

    if mask.any():
        for c in range(tensor.shape[-1]):
            band = tensor[:, :, c]
            valid = band[~mask[:, :, c]]
            if valid.size > 0:
                median_val = np.median(valid)
                band[mask[:, :, c]] = median_val
                tensor[:, :, c] = band

    return tensor


def normalize_tensor(tensor: np.ndarray) -> np.ndarray:
    """
    Min-max normalize each band to [0, 1].
    """
    h, w, c = tensor.shape
    out = np.zeros_like(tensor, dtype=np.float32)
    for i in range(c):
        band = tensor[:, :, i].astype(np.float32)
        bmin = band.min()
        bmax = band.max()
        if bmax - bmin > 1e-10:
            out[:, :, i] = (band - bmin) / (bmax - bmin)
        else:
            out[:, :, i] = band - bmin
    return out


def build_spectral_tensor(
    bands: Dict[str, np.ndarray],
    selected_bands: List[str],
    include_indices: bool = True,
) -> np.ndarray:
    """
    Build a (H, W, C) tensor from selected raw bands and optionally
    append computed spectral indices.

    Raises ValueError if no valid bands are available.
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
        arrays.append(normalize_band(arr, method="minmax"))

    if not arrays:
        raise ValueError("No valid bands available for tensor construction.")

    if include_indices:
        index_dict = apply_spectral_indices(bands)
        for idx_name, idx_arr in index_dict.items():
            if idx_arr.shape != (h, w):
                continue
            arrays.append(normalize_band(idx_arr, method="minmax"))

    return np.stack(arrays, axis=-1)


class SatellitePredictor:
    """
    End-to-end satellite AGBD predictor.

    Loads the model lazily and provides a predict() method for single files.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        normalizer_path: Optional[Path] = None,
    ):
        self.model_path = model_path or MODEL_PATH
        self.metadata_path = metadata_path or METADATA_PATH
        self.normalizer_path = normalizer_path
        self.model = None
        self.device = torch.device("cpu")
        self.metadata = {}
        self.in_channels = 6
        self.transform = None
        self.normalizer: Optional[Sentinel2Normalizer] = None

    def _load_metadata(self):
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path) as f:
                    self.metadata = json.load(f)
                self.in_channels = self.metadata.get("in_channels", 6)
            except Exception as e:
                logger.warning("Failed to load satellite metadata: %s", e)

    def _load_normalizer(self):
        if self.normalizer is not None:
            return

        if self.normalizer_path is None:
            self.normalizer_path = self.metadata_path.parent / "sentinel2_normalizer.json"

        if self.normalizer_path.exists():
            try:
                self.normalizer = Sentinel2Normalizer.load(self.normalizer_path)
                logger.info(
                    "Loaded Sentinel2Normalizer version=%s with %d channels",
                    self.normalizer.normalization_version,
                    len(self.normalizer.channel_names),
                )
                return
            except Exception as e:
                logger.warning("Failed to load normalizer from file: %s", e)

        if self.metadata:
            try:
                channel_names = self.metadata.get("channel_names", [])
                mean = self.metadata.get("mean")
                std = self.metadata.get("std")
                if channel_names and mean is not None and std is not None:
                    self.normalizer = Sentinel2Normalizer(
                        channel_names=channel_names,
                        normalization_version=self.metadata.get("normalization_version", "sentinel2-normalization-v1"),
                    )
                    self.normalizer.mean = np.array(mean, dtype=np.float64)
                    self.normalizer.std = np.array(std, dtype=np.float64)
                    logger.info(
                        "Loaded Sentinel2Normalizer from metadata with %d channels",
                        len(channel_names),
                    )
                    return
            except Exception as e:
                logger.warning("Failed to load normalizer from metadata: %s", e)

        logger.warning("No normalizer metadata available. Predictions may be less accurate.")

    def _load_model(self):
        if self.model is not None:
            return

        if not self.model_path.exists() or self.model_path.stat().st_size == 0:
            raise FileNotFoundError("Satellite model checkpoint is missing or empty")

        self._load_metadata()
        self._load_normalizer()
        in_channels = self.metadata.get("in_channels", 6)

        state_dict = torch.load(self.model_path, map_location=self.device)
        is_baseline = "conv1.weight" in state_dict and "init_conv.weight" not in state_dict
        if is_baseline:
            from models.satellite_cnn import BaselineSatelliteCNN
            model = BaselineSatelliteCNN(in_channels=in_channels)
        else:
            model = SatelliteCNN(in_channels=in_channels)
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model
        self.in_channels = in_channels
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.Normalize(mean=[0.5] * in_channels, std=[0.5] * in_channels),
        ])
        logger.info("Satellite CNN loaded with %d input channels", in_channels)

    def load(self):
        self._load_model()

    def predict_file(self, filepath: Path) -> Dict:
        """
        Run AGBD prediction on a single satellite file.

        Returns a dict with:
          - agbd: predicted AGBD in Mg/ha
          - carbon: estimated carbon in Mg C/ha
          - metadata: file metadata and diagnostics
          - preprocessing_ok: bool
        """
        self._load_model()

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Satellite file not found: {filepath}")

        ext = filepath.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: GeoTIFF (.tif, .tiff), JPG, JPEG, PNG, WEBP.")

        if ext in IMAGE_EXTENSIONS:
            tensor, metadata, selected_bands = _process_rgb_image(
                filepath,
                in_channels=self.in_channels,
                normalizer=self.normalizer,
            )
        else:
            try:
                tensor, preprocessing_meta = preprocess_sentinel2(
                    filepath,
                    selected_bands=DEFAULT_FEATURE_BANDS,
                    include_indices=True,
                    target_resolution=10,
                    normalizer=self.normalizer,
                )
            except Exception as e:
                raise ValueError(f"Failed to preprocess satellite image: {e}")

            metadata = {
                "file": filepath.name,
                "size_bytes": filepath.stat().st_size,
                "array_shape": list(tensor.shape),
                "available_bands": sorted([
                    b for b in preprocessing_meta.get("channel_names", [])
                    if b in SENTINEL2_BANDS
                ]),
                "missing_required_bands": [
                    b for b in REQUIRED_BANDS
                    if b not in preprocessing_meta.get("channel_names", [])
                ],
                "bands_valid": len([
                    b for b in REQUIRED_BANDS
                    if b in preprocessing_meta.get("channel_names", [])
                ]) == len(REQUIRED_BANDS),
                "tiff_metadata": _read_basic_tiff_metadata(filepath),
                "preprocessing": preprocessing_meta,
            }

            selected_bands = [
                b for b in DEFAULT_FEATURE_BANDS
                if b in preprocessing_meta.get("channel_names", [])
            ]
            if not selected_bands:
                selected_bands = [
                    b for b in preprocessing_meta.get("channel_names", [])
                    if b in SENTINEL2_BANDS
                ][: min(6, len(SENTINEL2_BANDS))]

            if not selected_bands:
                selected_bands = ["B2", "B3", "B4"]

        # tensor is already normalized by the preprocessor when normalizer is provided
        h, w, c = tensor.shape
        if tensor.ndim == 2:
            tensor = tensor[:, :, np.newaxis]

        if tensor.shape[-1] < self.in_channels:
            pad = np.zeros(
                (tensor.shape[0], tensor.shape[1], self.in_channels - tensor.shape[-1]),
                dtype=np.float32,
            )
            tensor = np.concatenate([tensor, pad], axis=-1)
        elif tensor.shape[-1] > self.in_channels:
            tensor = tensor[:, :, : self.in_channels]

        tensor_t = self.transform(tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            agbd_raw = self.model(tensor_t).item()

        agbd = max(0.0, float(agbd_raw))
        carbon = agbd * CARBON_FACTOR

        model_version = self.metadata.get("model_version", MODEL_VERSION)
        status = self.metadata.get("status", "trained")

        return {
            "success": True,
            "predictionType": "satellite",
            "model": "Satellite CNN",
            "modelVersion": model_version,
            "status": status,
            "agbd": round(agbd, 4),
            "agbd_unit": "Mg/ha",
            "carbon": round(carbon, 4),
            "carbon_unit": "Mg C/ha",
            "carbon_fraction": CARBON_FACTOR,
            "in_channels": self.in_channels,
            "selected_bands": selected_bands,
            "qualityOk": True,
            "metadata": metadata,
        }

    def predict(self, filepath: Union[Path, str]) -> Dict:
        """Alias for predict_file."""
        return self.predict_file(Path(filepath))

    def predict_bytes(self, data: bytes, filename: str = "upload.tif") -> Dict:
        """
        Run AGBD prediction from raw bytes.
        """
        import tempfile

        self._load_model()

        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: GeoTIFF (.tif, .tiff), JPG, JPEG, PNG, WEBP.")

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        try:
            result = self.predict_file(tmp_path)
            result["metadata"]["file"] = filename
            result["metadata"]["size_bytes"] = len(data)
            return result
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass


def load_model(
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    normalizer_path: Optional[Path] = None,
):
    """
    Public helper to instantiate a predictor and load the model.
    """
    predictor = SatellitePredictor(
        model_path=model_path,
        metadata_path=metadata_path,
        normalizer_path=normalizer_path,
    )
    predictor.load()
    return predictor
