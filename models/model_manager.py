"""
Centralized Model Manager and Singleton Cache for Green Campus ML Service.

Features:
- Enforces CPU-only execution (map_location='cpu', device='cpu', 1 thread).
- Thread-safe singleton pattern for all 3 models:
    1. CatBoost AGB Regressor
    2. EfficientNet-B0 Image AGB Regressor
    3. Satellite CNN AGBD Regressor
- Decouples metadata loading from model weight loading:
    /model-info endpoints read JSON without loading heavy PyTorch weights into RAM.
- Lazy-loads models only when actual prediction is requested.
- Prevents duplicate model instantiations and excessive memory consumption.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import transforms

logger = logging.getLogger("uvicorn")

# Base repository directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model and metadata filepaths
CATBOOST_AGB_PATH = BASE_DIR / "models" / "catboost_agb_model.cbm"
CATBOOST_AGB_META_PATH = BASE_DIR / "models" / "catboost_agb_metadata.json"

EFFICIENTNET_MODEL_PATH = BASE_DIR / "models" / "efficientnet_agb.pth"
EFFICIENTNET_METADATA_PATH = BASE_DIR / "models" / "efficientnet_agb_metadata.json"

SATELLITE_MODEL_PATH = BASE_DIR / "models" / "satellite_cnn_agb.pth"
SATELLITE_METADATA_PATH = BASE_DIR / "models" / "satellite_cnn_agb_metadata.json"

# Force PyTorch CPU execution limits to prevent thread-pool memory explosion on Render
try:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
except Exception as e:
    logger.warning("Could not set PyTorch thread limits: %s", e)

CPU_DEVICE = torch.device("cpu")


# =====================================================================
# Model Architecture Definitions
# =====================================================================

class EfficientNetAGB(nn.Module):
    """EfficientNet-B0 backbone with custom regression head for AGB prediction."""

    def __init__(self, num_outputs: int = 1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# =====================================================================
# Model Manager Singleton
# =====================================================================

class ModelManager:
    """Thread-safe singleton registry for ML models and metadata."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_manager()
        return cls._instance

    def _init_manager(self):
        # Locks for individual model loading
        self._agb_lock = threading.Lock()
        self._image_lock = threading.Lock()
        self._satellite_lock = threading.Lock()

        # Cached models
        self._agb_model = None
        self._image_model = None
        self._image_transform = None
        self._satellite_predictor = None

        # Cached metadata (lightweight JSON)
        self._agb_metadata: Optional[Dict[str, Any]] = None
        self._image_metadata: Optional[Dict[str, Any]] = None
        self._satellite_metadata: Optional[Dict[str, Any]] = None

        # Errors if model files are missing
        self._image_model_error: Optional[str] = None
        self._satellite_model_error: Optional[str] = None

    # -----------------------------------------------------------------
    # CatBoost AGB Model
    # -----------------------------------------------------------------
    def get_agb_model(self):
        """Get or lazily load the CatBoost AGB regression model."""
        if self._agb_model is not None:
            return self._agb_model

        with self._agb_lock:
            if self._agb_model is not None:
                return self._agb_model

            if not CATBOOST_AGB_PATH.exists() or CATBOOST_AGB_PATH.stat().st_size == 0:
                raise FileNotFoundError(f"CatBoost model missing at {CATBOOST_AGB_PATH}")

            from catboost import CatBoostRegressor
            logger.info("Loading CatBoost AGB model from %s", CATBOOST_AGB_PATH)
            model = CatBoostRegressor()
            model.load_model(str(CATBOOST_AGB_PATH))
            self._agb_model = model
            return self._agb_model

    def get_agb_metadata(self) -> Dict[str, Any]:
        """Read and cache CatBoost metadata without heavy model loading."""
        if self._agb_metadata is not None:
            return self._agb_metadata

        if CATBOOST_AGB_META_PATH.exists():
            try:
                with open(CATBOOST_AGB_META_PATH, "r", encoding="utf-8") as f:
                    self._agb_metadata = json.load(f)
            except Exception as e:
                logger.warning("Failed to read CatBoost metadata: %s", e)
                self._agb_metadata = {}
        else:
            self._agb_metadata = {}
        return self._agb_metadata

    # -----------------------------------------------------------------
    # EfficientNet-B0 Image Model
    # -----------------------------------------------------------------
    def get_image_metadata(self) -> Optional[Dict[str, Any]]:
        """Read and cache EfficientNet metadata without loading PyTorch weights."""
        if self._image_metadata is not None:
            return self._image_metadata

        if EFFICIENTNET_METADATA_PATH.exists():
            try:
                with open(EFFICIENTNET_METADATA_PATH, "r", encoding="utf-8") as f:
                    self._image_metadata = json.load(f)
            except Exception as e:
                logger.warning("Failed to read EfficientNet metadata: %s", e)
                self._image_metadata = None
        return self._image_metadata

    def get_image_model_and_transform(self) -> Tuple[nn.Module, Any]:
        """
        Get or lazily load the EfficientNet-B0 image model and transform.
        Guarantees CPU loading and evaluation mode.
        """
        if self._image_model is not None and self._image_transform is not None:
            return self._image_model, self._image_transform

        with self._image_lock:
            if self._image_model is not None and self._image_transform is not None:
                return self._image_model, self._image_transform

            if not EFFICIENTNET_MODEL_PATH.exists() or EFFICIENTNET_MODEL_PATH.stat().st_size == 0:
                self._image_model_error = "IMAGE_MODEL_NOT_TRAINED"
                raise FileNotFoundError("EfficientNet image model checkpoint is missing or empty")

            try:
                logger.info("Loading EfficientNet-B0 model on CPU from %s", EFFICIENTNET_MODEL_PATH)
                model = EfficientNetAGB(num_outputs=1)
                state_dict = torch.load(EFFICIENTNET_MODEL_PATH, map_location=CPU_DEVICE)
                model.load_state_dict(state_dict)
                model.to(CPU_DEVICE)
                model.eval()

                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                self._image_model = model
                self._image_transform = transform
                self._image_model_error = None
                return self._image_model, self._image_transform
            except Exception as e:
                self._image_model_error = "IMAGE_MODEL_LOAD_FAILED"
                logger.error("Failed to load EfficientNet image model: %s", e)
                raise

    def get_image_model_error(self) -> Optional[str]:
        if not EFFICIENTNET_MODEL_PATH.exists() or EFFICIENTNET_MODEL_PATH.stat().st_size == 0:
            return "IMAGE_MODEL_NOT_TRAINED"
        return self._image_model_error

    # -----------------------------------------------------------------
    # Satellite CNN Model
    # -----------------------------------------------------------------
    def get_satellite_metadata(self) -> Optional[Dict[str, Any]]:
        """Read and cache Satellite CNN metadata without loading PyTorch weights."""
        if self._satellite_metadata is not None:
            return self._satellite_metadata

        if SATELLITE_METADATA_PATH.exists():
            try:
                with open(SATELLITE_METADATA_PATH, "r", encoding="utf-8") as f:
                    self._satellite_metadata = json.load(f)
            except Exception as e:
                logger.warning("Failed to read satellite metadata: %s", e)
                self._satellite_metadata = None
        return self._satellite_metadata

    def get_satellite_predictor(self):
        """
        Get or lazily load the SatellitePredictor singleton.
        Guarantees CPU loading, evaluation mode, and Baseline architecture support.
        """
        if self._satellite_predictor is not None:
            return self._satellite_predictor

        with self._satellite_lock:
            if self._satellite_predictor is not None:
                return self._satellite_predictor

            if not SATELLITE_MODEL_PATH.exists() or SATELLITE_MODEL_PATH.stat().st_size == 0:
                self._satellite_model_error = "SATELLITE_MODEL_NOT_TRAINED"
                raise FileNotFoundError("Satellite model checkpoint is missing or empty")

            try:
                from inference.satellite_predictor import SatellitePredictor
                logger.info("Loading SatellitePredictor on CPU from %s", SATELLITE_MODEL_PATH)
                predictor = SatellitePredictor(
                    model_path=SATELLITE_MODEL_PATH,
                    metadata_path=SATELLITE_METADATA_PATH,
                )
                predictor.load()
                self._satellite_predictor = predictor
                self._satellite_model_error = None
                return self._satellite_predictor
            except Exception as e:
                self._satellite_model_error = "SATELLITE_MODEL_LOAD_FAILED"
                logger.error("Failed to load satellite predictor: %s", e)
                raise

    def get_satellite_model_error(self) -> Optional[str]:
        if not SATELLITE_MODEL_PATH.exists() or SATELLITE_MODEL_PATH.stat().st_size == 0:
            return "SATELLITE_MODEL_NOT_TRAINED"
        return self._satellite_model_error

    # -----------------------------------------------------------------
    # Status Helper
    # -----------------------------------------------------------------
    def get_status_overview(self) -> Dict[str, Any]:
        """
        Return the status of all three models without triggering heavy loads.
        """
        agb_status = "loaded" if self._agb_model is not None else ("ready" if CATBOOST_AGB_PATH.exists() else "missing")
        image_status = "loaded" if self._image_model is not None else ("ready" if EFFICIENTNET_MODEL_PATH.exists() else "not_trained")
        sat_status = "loaded" if self._satellite_predictor is not None else ("ready" if SATELLITE_MODEL_PATH.exists() else "not_trained")

        sat_meta = self.get_satellite_metadata() or {}
        in_channels = sat_meta.get("in_channels", 15)

        return {
            "agb": agb_status,
            "image": image_status,
            "satellite": sat_status,
            "satellite_in_channels": in_channels,
        }


# Global helper functions
def get_model_manager() -> ModelManager:
    return ModelManager()

def get_agb_model():
    return ModelManager().get_agb_model()

def get_image_model_and_transform():
    return ModelManager().get_image_model_and_transform()

def get_image_metadata():
    return ModelManager().get_image_metadata()

def get_satellite_predictor():
    return ModelManager().get_satellite_predictor()

def get_satellite_metadata():
    return ModelManager().get_satellite_metadata()
