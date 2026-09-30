"""
EfficientNet-B0 inference for image-based AGB prediction.

This module provides:
1. EfficientNetAGB model class (matching the trained architecture)
2. Metadata loading and validation
3. Image preprocessing (identical to training)
4. Prediction function with AGB + carbon output
5. Image quality diagnostics (for display, NOT fed into model)
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

EFFICIENTNET_MEAN = [0.485, 0.456, 0.406]
EFFICIENTNET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224
CARBON_FACTOR = 0.47
MODEL_VERSION = "efficientnet-b0-v1"


class EfficientNetAGB(nn.Module):
    """EfficientNet-B0 with regression head for AGB prediction."""

    def __init__(self, num_outputs=1):
        super().__init__()
        from torchvision.models import efficientnet_b0

        self.backbone = efficientnet_b0()
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_outputs),
        )

    def forward(self, x):
        return self.backbone(x)


def _get_paths():
    base = Path(__file__).resolve().parent.parent
    model_path = base / "models" / "efficientnet_agb.pth"
    metadata_path = base / "models" / "efficientnet_agb_metadata.json"
    return model_path, metadata_path


def load_model_and_metadata():
    """
    Load the trained EfficientNet-B0 checkpoint and its metadata using ModelManager.
    Cached as a singleton on CPU.
    """
    from models.model_manager import get_model_manager
    manager = get_model_manager()
    model, _ = manager.get_image_model_and_transform()
    metadata = manager.get_image_metadata() or {}
    device = torch.device("cpu")
    return model, device, metadata


def get_preprocessing_transform():
    """
    Return the deterministic preprocessing transform for inference.
    This must be IDENTICAL to the validation/test transform used during training.
    """
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=EFFICIENTNET_MEAN, std=EFFICIENTNET_STD),
    ])


def compute_image_diagnostics(image_path: str):
    """
    Compute interpretable image statistics for quality assessment (req. 22).
    These are NOT fed into the model — they are for display / warning only.

    Memory-optimized: uses thumbnail for luminance/contrast calculation.
    """
    with Image.open(image_path) as img:
        return compute_image_diagnostics_from_pil(img)



def predict_image(image_path: str):
    """
    Run AGB prediction on a single image.

    Args:
        image_path: Path to the image file.

    Returns a dict:
        agb (float): Predicted AGB in kg
        carbon (float): Estimated carbon = AGB * 0.47
        model_version (str)
        diagnostics (dict): Image quality statistics
        quality_warning (bool): Whether to warn about image quality
        preprocessing_ok (bool): Whether the image was successfully loaded
    """
    try:
        model, device, metadata = load_model_and_metadata()
    except FileNotFoundError as e:
        return {"error": str(e), "agb": None, "carbon": None}

    transform = get_preprocessing_transform()

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Cannot open image: {e}", "agb": None, "carbon": None}

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        agb_raw = model(image_tensor).item()

    # Clamp to a reasonable biological range
    agb = max(0.0, float(agb_raw))
    carbon = round(agb * CARBON_FACTOR, 4)
    agb = round(agb, 4)

    # Compute quality diagnostics
    diagnostics = compute_image_diagnostics(image_path)

    return {
        "agb": agb,
        "carbon": carbon,
        "model_version": MODEL_VERSION,
        "diagnostics": diagnostics,
        "quality_warning": diagnostics["quality_warning"],
        "preprocessing_ok": True,
    }


def predict_image_from_bytes(image_bytes: bytes):
    """
    Run AGB prediction from raw image bytes (used by FastAPI).

    Args:
        image_bytes: Raw bytes of the image file.

    Returns: same dict as predict_image().
    """
    import io
    try:
        model, device, metadata = load_model_and_metadata()
    except FileNotFoundError as e:
        return {"error": str(e), "agb": None, "carbon": None}

    transform = get_preprocessing_transform()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Cannot open image from bytes: {e}", "agb": None, "carbon": None}

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        agb_raw = model(image_tensor).item()

    agb = max(0.0, float(agb_raw))
    carbon = round(agb * CARBON_FACTOR, 4)
    agb = round(agb, 4)

    diagnostics = compute_image_diagnostics_from_pil(image)

    return {
        "agb": agb,
        "carbon": carbon,
        "model_version": MODEL_VERSION,
        "diagnostics": diagnostics,
        "quality_warning": diagnostics["quality_warning"],
        "preprocessing_ok": True,
    }


def compute_image_diagnostics_from_pil(img: Image.Image):
    """
    Compute image diagnostics from an already-opened PIL Image.
    Uses a small thumbnail copy for luminance/contrast to avoid allocating large arrays.
    """
    width, height = img.size

    # Memory-safe thumbnail for statistical checks (max 256x256)
    thumb = img.copy()
    thumb.thumbnail((256, 256), Image.Resampling.BILINEAR)
    arr = np.array(thumb.convert("RGB"), dtype=np.float32) / 255.0
    del thumb

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    mean_rgb = {
        "r": round(float(r.mean()), 4),
        "g": round(float(g.mean()), 4),
        "b": round(float(b.mean()), 4),
    }

    green_mask = (g > r) & (g > b) & (g > 0.3)
    green_ratio = round(float(green_mask.mean()), 4)

    lum = 0.299 * r + 0.587 * g + 0.114 * b
    brightness = round(float(lum.mean()), 4)
    contrast = round(float(lum.std()), 4)
    is_blank = bool(lum.std() < 1e-6)
    del arr

    is_extremely_dark = bool(brightness < 0.05)
    is_very_small = bool(width < 64 or height < 64)
    quality_warning = bool(is_blank or is_extremely_dark or is_very_small)

    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 4) if height else 0.0,
        "mean_rgb": mean_rgb,
        "green_pixel_ratio": green_ratio,
        "brightness": brightness,
        "contrast": contrast,
        "is_blank": is_blank,
        "is_extremely_dark": is_extremely_dark,
        "is_very_small": is_very_small,
        "quality_warning": quality_warning,
    }

