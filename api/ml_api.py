import os
import logging
import io
from pathlib import Path

from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostRegressor, Pool
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import json


BASE_DIR = Path(__file__).resolve().parent.parent

logger = logging.getLogger("uvicorn")

app = FastAPI(
    title="Green Campus AI API",
    description="Machine Learning API for Green Campus Biomass Monitoring",
    version="4.0.0",
)


# =========================================================
# Load AGB CatBoost model (production, must be preserved)
# =========================================================

AGB_MODEL_PATH = BASE_DIR / "models" / "catboost_agb_model.cbm"

agb_model = CatBoostRegressor()
agb_model.load_model(str(AGB_MODEL_PATH))

AGB_FEATURES = ["diameter", "height", "year", "group", "site"]
AGB_CATEGORICAL = ["group", "site"]

REQUIRED_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]


# =========================================================
# EfficientNet-B0 Image Model
# =========================================================

EFFICIENTNET_MODEL_PATH = BASE_DIR / "models" / "efficientnet_agb.pth"
EFFICIENTNET_METADATA_PATH = BASE_DIR / "models" / "efficientnet_agb_metadata.json"

image_model = None
image_transform = None
image_metadata = None
image_model_error = None


class EfficientNetAGB(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        from torchvision.models import efficientnet_b0
        self.backbone = efficientnet_b0()
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def _load_image_model():
    global image_model, image_transform, image_metadata, image_model_error

    if image_model is not None or image_model_error is not None:
        return

    if not EFFICIENTNET_MODEL_PATH.exists() or EFFICIENTNET_MODEL_PATH.stat().st_size == 0:
        image_model_error = "IMAGE_MODEL_NOT_TRAINED"
        logger.warning("EfficientNet image model checkpoint is missing or empty")
        return

    try:
        model = EfficientNetAGB()
        state_dict = torch.load(EFFICIENTNET_MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        image_model = model

        from torchvision import transforms
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if EFFICIENTNET_METADATA_PATH.exists():
            with open(EFFICIENTNET_METADATA_PATH) as f:
                image_metadata = json.load(f)

        logger.info("EfficientNet-B0 image model loaded successfully")
    except Exception as e:
        image_model_error = "IMAGE_MODEL_LOAD_FAILED"
        logger.error("Failed to load EfficientNet image model: %s", str(e))


# =========================================================
# Satellite CNN Model
# =========================================================

SATELLITE_MODEL_PATH = BASE_DIR / "models" / "satellite_cnn_agb.pth"
SATELLITE_METADATA_PATH = BASE_DIR / "models" / "satellite_cnn_agb_metadata.json"

satellite_model = None
satellite_predictor = None
satellite_model_error = None
satellite_in_channels = 6
satellite_metadata = None


def _load_satellite_model():
    global satellite_model, satellite_predictor, satellite_model_error, satellite_in_channels, satellite_metadata

    if satellite_model is not None or satellite_model_error is not None:
        return

    if not SATELLITE_MODEL_PATH.exists() or SATELLITE_MODEL_PATH.stat().st_size == 0:
        satellite_model_error = "SATELLITE_MODEL_NOT_TRAINED"
        logger.warning("Satellite model checkpoint is missing or empty")
        return

    try:
        from inference.satellite_predictor import SatellitePredictor
        satellite_predictor = SatellitePredictor(
            model_path=SATELLITE_MODEL_PATH,
            metadata_path=SATELLITE_METADATA_PATH,
        )
        satellite_predictor.load()
        satellite_model = satellite_predictor.model
        satellite_in_channels = satellite_predictor.in_channels

        if SATELLITE_METADATA_PATH.exists():
            with open(SATELLITE_METADATA_PATH) as f:
                satellite_metadata = json.load(f)

        logger.info("Satellite CNN model loaded successfully")
    except Exception as e:
        satellite_model_error = "SATELLITE_MODEL_LOAD_FAILED"
        logger.error("Failed to load satellite model: %s", str(e))


# =========================================================
# Request Models
# =========================================================

class AGBRequest(BaseModel):
    diameter: float
    height: float
    year: int
    group: str
    site: str


# =========================================================
# Image quality checks (requirement 21)
# =========================================================

def check_image_quality(img: Image.Image):
    """
    Basic image quality assessment.
    Returns a dict with warnings.
    """
    warnings = []
    w, h = img.size
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0

    if w < 64 or h < 64:
        warnings.append("Image is very small and may affect prediction accuracy.")
    if w == 0 or h == 0:
        warnings.append("Image has zero dimensions.")

    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    brightness = float(lum.mean())
    contrast = float(lum.std())

    if brightness < 0.05:
        warnings.append("Image is extremely dark.")
    if lum.std() < 1e-6:
        warnings.append("Image appears to be blank or solid colour.")
    if contrast < 0.01:
        warnings.append("Image has very low contrast.")

    return {
        "warnings": warnings,
        "quality_ok": len(warnings) == 0,
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "width": w,
        "height": h,
    }


# =========================================================
# Health Check
# =========================================================

@app.get("/")
def root():
    _load_image_model()
    _load_satellite_model()

    return {
        "message": "Green Campus AI API is running",
        "status": "healthy",
        "version": "4.0.0",
        "models": [
            {
                "type": "CatBoostRegressor",
                "target": "AGB (Above-Ground Biomass)",
                "version": "catboost-agb-v1",
                "status": "loaded",
            },
            {
                "type": "EfficientNet-B0 + Regression Head",
                "target": "AGB from Images",
                "version": "efficientnet-b0-v1",
                "status": "loaded" if image_model is not None else "not_trained",
                "model_version": "efficientnet-b0-v1",
            },
            {
                "type": "Satellite CNN",
                "target": "AGBD from Satellite Images",
                "version": "satellite-cnn-prototype-v1",
                "status": "ready" if satellite_model is not None else "not_trained",
                "in_channels": satellite_in_channels,
                "satellite_model_type": "prototype",
            }
        ],
    }


# =========================================================
# Model Info endpoint (for frontend to load real metrics)
# =========================================================

@app.get("/model-info/image-agb")
def get_image_model_info():
    """
    Return image model metadata for the frontend.
    Includes real training metrics from the metadata JSON.
    """
    _load_image_model()

    if image_metadata:
        metrics = image_metadata.get("metrics", {})
        dataset_stats = image_metadata.get("dataset_stats", {})
        dataset_size = image_metadata.get("dataset_size", "unknown")
        train_size = image_metadata.get("train_size", "unknown")
        val_size = image_metadata.get("validation_size", "unknown")
        test_size = image_metadata.get("test_size", "unknown")

        return {
            "success": True,
            "model": image_metadata.get("model", "EfficientNet-B0"),
            "architecture": image_metadata.get("architecture", "EfficientNet-B0 CNN"),
            "modelVersion": image_metadata.get("modelVersion", "efficientnet-b0-v1"),
            "status": image_metadata.get("status", "unknown"),
            "dataset": {
                "description": image_metadata.get("dataset_description", ""),
                "source": image_metadata.get("dataset_source", ""),
                "total": dataset_size,
                "train": train_size,
                "validation": val_size,
                "test": test_size,
                "agb_min": dataset_stats.get("agb_min"),
                "agb_max": dataset_stats.get("agb_max"),
                "agb_mean": dataset_stats.get("agb_mean"),
            },
            "metrics": {
                "r2": metrics.get("r2", "pending"),
                "mae": metrics.get("mae", "pending"),
                "rmse": metrics.get("rmse", "pending"),
                "val_mae": metrics.get("val_mae", "pending"),
                "val_rmse": metrics.get("val_rmse", "pending"),
                "val_r2": metrics.get("val_r2", "pending"),
            },
            "split_strategy": image_metadata.get("split_strategy", ""),
            "loss_function": image_metadata.get("loss_function", ""),
            "pretrained": image_metadata.get("pretrained", True),
        }

    # Model not trained — return stub
    return {
        "success": False,
        "model": "EfficientNet-B0",
        "architecture": "EfficientNet-B0 CNN",
        "modelVersion": "efficientnet-b0-v1",
        "status": "not_trained",
        "metrics": {
            "r2": "Pending",
            "mae": "Pending",
            "rmse": "Pending",
        },
        "message": "Model not trained yet. Run ai/training/train_image_cnn.py first.",
    }


@app.get("/model-info/satellite-agb")
def get_satellite_model_info():
    """
    Return satellite model metadata for the frontend.
    """
    _load_satellite_model()

    if satellite_metadata and satellite_metadata.get("status") == "ready":
        metrics = satellite_metadata.get("metrics", {})
        return {
            "success": True,
            "model": satellite_metadata.get("model", "Satellite CNN"),
            "architecture": "Satellite CNN (multispectral regression)",
            "modelVersion": satellite_metadata.get("model_version", "satellite-cnn-prototype-v1"),
            "status": "ready",
            "type": "prototype",
            "target": satellite_metadata.get("target", "AGBD"),
            "unit": satellite_metadata.get("unit", "Mg/ha"),
            "carbonFactor": satellite_metadata.get("carbon_factor", 0.47),
            "dataset": {
                "description": satellite_metadata.get("dataset", "Prototype synthetic sample data"),
                "total": satellite_metadata.get("dataset_size", "prototype"),
                "train": satellite_metadata.get("train_size", "prototype"),
                "validation": satellite_metadata.get("validation_size", "prototype"),
                "test": satellite_metadata.get("test_size", "prototype"),
            },
            "metrics": {
                "r2": metrics.get("r2", "pending"),
                "mae": metrics.get("mae", "pending"),
                "rmse": metrics.get("rmse", "pending"),
            },
            "metricsLabel": "Prototype test metrics",
            "in_channels": satellite_in_channels,
            "bands": satellite_metadata.get("bands", REQUIRED_BANDS),
            "indices": satellite_metadata.get("indices", ["ndvi", "evi", "savi", "ndwi", "msi"]),
            "prototype": True,
            "note": satellite_metadata.get("note", "Prototype model trained on synthetic sample data."),
        }

    return {
        "success": False,
        "model": "Satellite CNN",
        "architecture": "Satellite CNN (multispectral regression)",
        "modelVersion": "satellite-cnn-prototype-v1",
        "status": "not_trained",
        "type": "prototype",
        "target": "AGBD",
        "unit": "Mg/ha",
        "carbonFactor": 0.47,
        "metrics": {
            "r2": "Pending",
            "mae": "Pending",
            "rmse": "Pending",
        },
        "metricsLabel": "Prototype test metrics",
        "in_channels": satellite_in_channels,
        "bands": REQUIRED_BANDS,
        "indices": ["ndvi", "evi", "savi", "ndwi", "msi"],
        "message": "Satellite biomass model is not trained yet. A labelled satellite dataset with ground-truth AGBD is required.",
    }


# =========================================================
# AGB Prediction from Field Data (CatBoost — PRESERVED)
# =========================================================

@app.post("/predict/agb")
def predict_agb(request: AGBRequest):
    try:
        data = pd.DataFrame([{
            "diameter": request.diameter,
            "height": request.height,
            "year": request.year,
            "group": request.group,
            "site": request.site,
        }])

        data = data[AGB_FEATURES]
        categorical_indices = [AGB_FEATURES.index(f) for f in AGB_CATEGORICAL]
        pool = Pool(data, cat_features=categorical_indices)
        prediction = agb_model.predict(pool)[0]
        carbon = prediction * 0.47

        return {
            "success": True,
            "predictionType": "field",
            "model": "CatBoostRegressor",
            "model_version": "catboost-agb-v1",
            "agb": round(float(prediction), 4),
            "carbon": round(float(carbon), 4),
            "unit": "kg",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# AGB Prediction from Tree/Canopy Image
# =========================================================

@app.post("/predict/image-agb")
def predict_image_agb(file: UploadFile = File(...)):
    _load_image_model()

    if image_model_error:
        code = image_model_error
        if code == "IMAGE_MODEL_LOAD_FAILED":
            message = "Image biomass model failed to load. Please check the model checkpoint."
        else:
            message = "Image biomass model is not trained yet. A labelled image dataset with ground-truth AGB is required for valid biomass prediction."
        raise HTTPException(status_code=503, detail={"code": code, "message": message})

    if image_model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "IMAGE_MODEL_NOT_TRAINED",
                "message": "Image biomass model is not trained yet. Run ai/training/train_image_cnn.py first."
            }
        )

    # MIME type check
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported media type. Please upload an image file.")

    # Extension check
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Supported formats: JPG, JPEG, PNG, WEBP."
        )

    # File size check
    contents = file.file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image size exceeds the allowed limit of 10MB.")

    # Verify image is readable and not corrupted
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")

    # Re-open for prediction (verify() closes the file)
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Image quality check
    quality = check_image_quality(img)

    # Preprocess and predict
    image_tensor = image_transform(img).unsqueeze(0)

    with torch.no_grad():
        agb_raw = image_model(image_tensor).item()

    agb = max(0.0, float(agb_raw))
    carbon = agb * 0.47

    response = {
        "success": True,
        "predictionType": "image",
        "model": "EfficientNet-B0",
        "modelVersion": "efficientnet-b0-v1",
        "agb": round(agb, 4),
        "carbon": round(carbon, 4),
        "unit": "kg",
    }

    # Add quality warnings if any
    if quality["warnings"]:
        response["qualityWarnings"] = quality["warnings"]
        response["qualityOk"] = False
        response["quality"] = {
            "brightness": quality["brightness"],
            "contrast": quality["contrast"],
            "width": quality["width"],
            "height": quality["height"],
        }
    else:
        response["qualityOk"] = True

    return response


# =========================================================
# AGBD Prediction from Satellite Image
# =========================================================

@app.post("/predict/satellite-agb")
def predict_satellite_agb(file: UploadFile = File(...)):
    _load_satellite_model()

    if satellite_model_error:
        code = satellite_model_error
        if code == "SATELLITE_MODEL_LOAD_FAILED":
            message = "Satellite biomass model failed to load. Please check the model checkpoint."
        else:
            message = (
                "Satellite biomass prediction is not available yet. "
                "A trained satellite CNN with ground-truth AGBD data is required."
            )
        raise HTTPException(status_code=503, detail={"code": code, "message": message})

    if satellite_predictor is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "SATELLITE_MODEL_NOT_TRAINED",
                "message": (
                    "Satellite biomass prediction is not available yet. "
                    "A trained satellite CNN with ground-truth AGBD data is required."
                ),
            },
        )

    ct = (file.content_type or "").lower()
    if not (ct.startswith("image/") or ct in {"application/octet-stream", "application/x-tiff", "application/tiff"}):
        raise HTTPException(status_code=415, detail="Unsupported media type.")

    allowed_extensions = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Invalid format. Supported formats: GeoTIFF (.tif, .tiff), JPG, JPEG, PNG, WEBP.",
        )

    contents = file.file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File size exceeds the allowed limit of 50MB.")

    try:
        result = satellite_predictor.predict_bytes(contents, filename=file.filename or "upload.tif")
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail={"code": "SATELLITE_MODEL_NOT_TRAINED", "message": str(e)})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Satellite prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail="Satellite prediction failed.")
