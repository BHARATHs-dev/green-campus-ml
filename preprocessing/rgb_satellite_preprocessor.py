"""
RGB Satellite Preprocessor for Prototype Inference.

This module converts standard RGB images (JPG/PNG/WEBP) into the 15-channel
format expected by the prototype Satellite CNN.

IMPORTANT: This is a PROTOTYPE inference adapter. Normal RGB images do NOT contain
actual Sentinel-2 multispectral bands (B5, B6, B7, B8, B8A, B11, B12).
The conversion uses RGB channels as approximations for B4, B3, B2 respectively,
and derives the remaining channels using proxy relationships.

DO NOT claim this produces real Sentinel-2 multispectral data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
METADATA_PATH = BASE_DIR / "ai" / "models" / "satellite_cnn_agb_metadata.json"

INPUT_SIZE = 224
TARGET_CHANNELS = 15

SENTINEL2_BAND_NAMES = [
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"
]
INDEX_NAMES = ["ndvi", "evi", "savi", "ndwi", "msi"]
CHANNEL_NAMES = SENTINEL2_BAND_NAMES + INDEX_NAMES


class RGBSatellitePreprocessor:
    """
    Prototype preprocessor for converting RGB images to 15-channel satellite CNN input.
    
    Mapping:
    - R channel -> B4 approximation (Red band, 665 nm)
    - G channel -> B3 approximation (Green band, 560 nm)
    - B channel -> B2 approximation (Blue band, 490 nm)
    
    Derived channels (proxy approximations):
    - B8 (NIR) ~ enhanced from R
    - B5, B6, B7 (Red Edge) ~ interpolated between R and B8
    - B8A (Narrow NIR) ~ similar to B8
    - B11, B12 (SWIR) ~ derived from R/G ratios
    - Indices (NDVI, EVI, SAVI, NDWI, MSI) computed from proxy bands
    
    This is marked as "RGB-to-multispectral prototype inference" and should
    NOT be used for scientifically validated AGBD estimation.
    """
    
    def __init__(self, metadata_path: Optional[Path] = None):
        self.metadata_path = metadata_path or METADATA_PATH
        self.metadata = {}
        self.mean = None
        self.std = None
        self.channel_names = CHANNEL_NAMES
        self._load_metadata()
        self.transform = self._build_transform()
    
    def _load_metadata(self):
        """Load normalization statistics from satellite model metadata."""
        import json
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path) as f:
                    self.metadata = json.load(f)
                self.channel_names = self.metadata.get("channel_names", CHANNEL_NAMES)
                self.mean = np.array(self.metadata.get("mean", [0.5] * len(self.channel_names)), dtype=np.float32)
                self.std = np.array(self.metadata.get("std", [0.5] * len(self.channel_names)), dtype=np.float32)
                logger.info("Loaded RGB preprocessor metadata with %d channels", len(self.channel_names))
            except Exception as e:
                logger.warning("Failed to load metadata for RGB preprocessor: %s", e)
                self._use_defaults()
        else:
            self._use_defaults()
    
    def _use_defaults(self):
        """Use default normalization when metadata unavailable."""
        self.channel_names = CHANNEL_NAMES
        self.mean = np.array([0.5] * len(self.channel_names), dtype=np.float32)
        self.std = np.array([0.5] * len(self.channel_names), dtype=np.float32)
        logger.warning("Using default normalization for RGB prototype preprocessor")
    
    def _build_transform(self):
        """Build torchvision transform pipeline."""
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean.tolist(), std=self.std.tolist()),
        ])
    
    def load_rgb_image(self, image_data: Union[bytes, Path, Image.Image]) -> Image.Image:
        """
        Load and convert image to RGB.
        
        Args:
            image_data: Image bytes, file path, or PIL Image
            
        Returns:
            PIL Image in RGB mode
        """
        if isinstance(image_data, bytes):
            img = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, (str, Path)):
            img = Image.open(image_data)
        elif isinstance(image_data, Image.Image):
            img = image_data
        else:
            raise ValueError(f"Unsupported image input type: {type(image_data)}")
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    
    def rgb_to_proxy_bands(self, rgb_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert RGB (H, W, 3) array to proxy Sentinel-2 bands.
        
        Args:
            rgb_array: Normalized RGB array in [0, 1], shape (H, W, 3)
            
        Returns:
            Dictionary of proxy band arrays (H, W) each
        """
        h, w = rgb_array.shape[:2]
        R = rgb_array[:, :, 0].astype(np.float32)
        G = rgb_array[:, :, 1].astype(np.float32)
        B = rgb_array[:, :, 2].astype(np.float32)
        
        bands = {}
        
        # Direct RGB to B4, B3, B2 mapping
        bands["B4"] = R  # Red ~ B4 (665 nm)
        bands["B3"] = G  # Green ~ B3 (560 nm)
        bands["B2"] = B  # Blue ~ B2 (490 nm)
        
        # NIR proxy (B8): Enhanced vegetation signal from Red
        # NIR is typically higher than Red for vegetation
        bands["B8"] = np.clip(R * 1.3 + 0.1, 0, 1)
        
        # B8A (narrow NIR): Similar to B8
        bands["B8A"] = bands["B8"] * 0.95 + 0.02
        
        # Red Edge bands (B5, B6, B7): Interpolated between Red and NIR
        # B5 (705 nm), B6 (740 nm), B7 (783 nm)
        bands["B5"] = np.clip(R * 0.7 + bands["B8"] * 0.3, 0, 1)
        bands["B6"] = np.clip(R * 0.5 + bands["B8"] * 0.5, 0, 1)
        bands["B7"] = np.clip(R * 0.3 + bands["B8"] * 0.7, 0, 1)
        
        # SWIR bands (B11, B12): Derived from R/G/B ratios
        # SWIR responds to moisture and structure
        bands["B11"] = np.clip((R * 0.6 + G * 0.3 + B * 0.1) * 0.8, 0, 1)
        bands["B12"] = np.clip((R * 0.5 + G * 0.2 + B * 0.3) * 0.7, 0, 1)
        
        return bands
    
    def compute_proxy_indices(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute spectral indices from proxy bands.
        
        Args:
            bands: Dictionary of proxy band arrays
            
        Returns:
            Dictionary of index arrays
        """
        indices = {}
        eps = 1e-8
        
        B8 = bands.get("B8", np.zeros_like(bands["B4"]))
        B4 = bands["B4"]
        B3 = bands["B3"]
        B2 = bands["B2"]
        B11 = bands.get("B11", np.zeros_like(B4))
        B8A = bands.get("B8A", B8)
        
        # NDVI = (NIR - Red) / (NIR + Red)
        indices["ndvi"] = np.clip((B8 - B4) / (B8 + B4 + eps), -1, 1)
        
        # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        indices["evi"] = np.clip(2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1 + eps), -1, 1)
        
        # SAVI = (1 + L) * (NIR - Red) / (NIR + Red + L), L=0.5
        L = 0.5
        indices["savi"] = np.clip((1 + L) * (B8 - B4) / (B8 + B4 + L + eps), -1, 1)
        
        # NDWI = (NIR - SWIR) / (NIR + SWIR) - using B8A and B11
        indices["ndwi"] = np.clip((B8A - B11) / (B8A + B11 + eps), -1, 1)
        
        # MSI = SWIR / NIR
        indices["msi"] = np.clip(B11 / (B8A + eps), 0, 5)
        
        return indices
    
    def build_15_channel_tensor(
        self,
        bands: Dict[str, np.ndarray],
        indices: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Build 15-channel tensor from proxy bands and indices.
        
        Args:
            bands: Proxy band arrays
            indices: Computed index arrays
            
        Returns:
            Tensor of shape (H, W, 15) with channels in metadata order
        """
        h, w = bands["B4"].shape
        channels = []
        
        for name in self.channel_names:
            if name in bands:
                arr = bands[name]
            elif name in indices:
                arr = indices[name]
            else:
                arr = np.zeros((h, w), dtype=np.float32)
                logger.warning("Channel %s not found, using zeros", name)
            
            if arr.shape != (h, w):
                raise ValueError(f"Channel {name} has shape {arr.shape}, expected ({h}, {w})")
            
            channels.append(arr.astype(np.float32))
        
        tensor = np.stack(channels, axis=-1)
        return tensor
    
    def preprocess(self, image_data: Union[bytes, Path, Image.Image]) -> torch.Tensor:
        """
        Full preprocessing pipeline: RGB image -> 15-channel tensor.
        
        Args:
            image_data: Image bytes, path, or PIL Image
            
        Returns:
            Tensor of shape (1, 15, H, W) ready for model inference
        """
        # Load RGB image
        img = self.load_rgb_image(image_data)
        
        # Convert to numpy array [0, 1]
        rgb_array = np.array(img).astype(np.float32) / 255.0
        
        # Resize if needed
        if rgb_array.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
            img_resized = img.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS)
            rgb_array = np.array(img_resized).astype(np.float32) / 255.0
        
        # Convert RGB to proxy bands
        proxy_bands = self.rgb_to_proxy_bands(rgb_array)
        
        # Compute proxy indices
        proxy_indices = self.compute_proxy_indices(proxy_bands)
        
        # Build 15-channel tensor
        tensor = self.build_15_channel_tensor(proxy_bands, proxy_indices)
        
        # Apply transform (includes normalization)
        tensor_t = self.transform(tensor)
        
        # Add batch dimension
        tensor_t = tensor_t.unsqueeze(0)
        
        return tensor_t
    
    def preprocess_bytes(self, data: bytes) -> torch.Tensor:
        """Preprocess from raw bytes."""
        return self.preprocess(data)
    
    def preprocess_file(self, filepath: Union[str, Path]) -> torch.Tensor:
        """Preprocess from file path."""
        return self.preprocess(Path(filepath))


def create_rgb_preprocessor(metadata_path: Optional[Path] = None) -> RGBSatellitePreprocessor:
    """Factory function to create RGB satellite preprocessor."""
    return RGBSatellitePreprocessor(metadata_path=metadata_path)