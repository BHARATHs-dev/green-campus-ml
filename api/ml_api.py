import os
import logging
import io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import pandas as pd
from catboost import Pool
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Set PyTorch CPU thread limit to avoid memory bloat in containers
try:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
except Exception:
    pass

from models.model_manager import (
    get_model_manager,
    get_agb_model,
    get_image_model_and_transform,
    get_image_metadata,
    get_satellite_predictor,
    get_satellite_metadata,
    EfficientNetAGB,
)

BASE_DIR = Path(__file__).resolve().parent.parent

logger = logging.getLogger("uvicorn")

app = FastAPI(
    title="Green Campus AI API",
    description="Machine Learning API for Green Campus Biomass Monitoring",
    version="4.0.0",
)


# =========================================================
# Lightweight Health Check (Task 6)
# Must NOT load models, database, or run prediction
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "green-campus-ml",
    }


# =========================================================
# Feature and Band Definitions
# =========================================================

AGB_FEATURES = ["diameter", "height", "year", "group", "site"]
AGB_CATEGORICAL = ["group", "site"]
REQUIRED_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]


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
# Memory-Optimized Image Quality Assessment
# =========================================================

def check_image_quality(img: Image.Image):
    """
    Basic image quality assessment.
    Memory-optimized: uses thumbnail for luminance/contrast calculation
    to avoid allocating huge float32 arrays on high-resolution uploads.
    """
    warnings = []
    w, h = img.size

    if w < 64 or h < 64:
        warnings.append("Image is very small and may affect prediction accuracy.")
    if w == 0 or h == 0:
        warnings.append("Image has zero dimensions.")

    # Downsample thumbnail copy (max 256x256) for statistical calculations
    thumb = img.copy()
    thumb.thumbnail((256, 256), Image.Resampling.BILINEAR)
    arr = np.array(thumb.convert("RGB"), dtype=np.float32) / 255.0
    del thumb

    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    brightness = float(lum.mean())
    contrast = float(lum.std())
    is_blank = bool(lum.std() < 1e-6)
    del arr

    if brightness < 0.05:
        warnings.append("Image is extremely dark.")
    if is_blank:
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
# Root Endpoint (Lightweight status, no eager model loading)
# =========================================================

@app.get("/")
def root():
    manager = get_model_manager()
    overview = manager.get_status_overview()

    return {
        "message": "Green Campus AI API is running",
        "status": "healthy",
        "version": "4.0.0",
        "models": [
            {
                "type": "CatBoostRegressor",
                "target": "AGB (Above-Ground Biomass)",
                "version": "catboost-agb-v1",
                "status": overview["agb"],
            },
            {
                "type": "EfficientNet-B0 + Regression Head",
                "target": "AGB from Images",
                "version": "efficientnet-b0-v1",
                "status": overview["image"],
                "model_version": "efficientnet-b0-v1",
            },
            {
                "type": "Satellite CNN",
                "target": "AGBD from Satellite Images",
                "version": "satellite-cnn-prototype-v1",
                "status": overview["satellite"],
                "in_channels": overview["satellite_in_channels"],
                "satellite_model_type": "prototype",
            }
        ],
    }


# =========================================================
# Model Info Endpoints (Reads JSON metadata directly)
# =========================================================

@app.get("/model-info/image-agb")
def get_image_model_info():
    """
    Return image model metadata for the frontend.
    Includes real training metrics from the metadata JSON without loading PyTorch weights.
    """
    image_metadata = get_image_metadata()

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

    # Model not trained — return fallback
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
    Reads JSON metadata without loading PyTorch weights.
    """
    satellite_metadata = get_satellite_metadata()

    if satellite_metadata and satellite_metadata.get("status") in ("ready", "trained"):
        metrics = satellite_metadata.get("metrics", {})
        in_channels = satellite_metadata.get("in_channels", 15)
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
            "in_channels": in_channels,
            "bands": satellite_metadata.get("bands", REQUIRED_BANDS),
            "indices": satellite_metadata.get("indices", ["ndvi", "evi", "savi", "ndwi", "msi"]),
            "prototype": satellite_metadata.get("prototype", True),
            "note": satellite_metadata.get("note", "Prototype model trained on synthetic sample data."),
        }

    in_channels = 15
    if satellite_metadata:
        in_channels = satellite_metadata.get("in_channels", 15)

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
        "in_channels": in_channels,
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
        model = get_agb_model()

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
        prediction = model.predict(pool)[0]
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

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail={"code": "AGB_MODEL_NOT_FOUND", "message": str(e)})
    except Exception as e:
        logger.error("CatBoost prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# AGB Prediction from Tree/Canopy Image
# =========================================================

@app.post("/predict/image-agb")
def predict_image_agb(file: UploadFile = File(...)):
    manager = get_model_manager()
    err = manager.get_image_model_error()
    if err:
        if err == "IMAGE_MODEL_LOAD_FAILED":
            message = "Image biomass model failed to load. Please check the model checkpoint."
        else:
            message = "Image biomass model is not trained yet. A labelled image dataset with ground-truth AGB is required for valid biomass prediction."
        raise HTTPException(status_code=503, detail={"code": err, "message": message})

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

    # File size check (max 10MB)
    contents = file.file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image size exceeds the allowed limit of 10MB.")

    # Verify image integrity
    try:
        with Image.open(io.BytesIO(contents)) as raw_img:
            raw_img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")

    try:
        with Image.open(io.BytesIO(contents)) as img:
            img_rgb = img.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to process image data.")

    # Free raw bytes immediately
    del contents

    try:
        # Image quality check (uses memory-safe thumbnail)
        quality = check_image_quality(img_rgb)

        # Get cached model and transform
        model, transform = get_image_model_and_transform()

        # Preprocess to CPU tensor
        image_tensor = transform(img_rgb).unsqueeze(0)
        del img_rgb

        with torch.no_grad():
            agb_raw = model(image_tensor).item()
        del image_tensor

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

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail={"code": "IMAGE_MODEL_NOT_TRAINED", "message": str(e)},
        )
    except Exception as e:
        logger.error("Image prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Image prediction failed: {str(e)}")


# =========================================================
# AGBD Prediction from Satellite Image
# =========================================================

@app.post("/predict/satellite-agb")
def predict_satellite_agb(file: UploadFile = File(...)):
    manager = get_model_manager()
    err = manager.get_satellite_model_error()
    if err:
        if err == "SATELLITE_MODEL_LOAD_FAILED":
            message = "Satellite biomass model failed to load. Please check the model checkpoint."
        else:
            message = (
                "Satellite biomass prediction is not available yet. "
                "A trained satellite CNN with ground-truth AGBD data is required."
            )
        raise HTTPException(status_code=503, detail={"code": err, "message": message})

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
        predictor = get_satellite_predictor()
        result = predictor.predict_bytes(contents, filename=file.filename or "upload.tif")
        del contents
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail={"code": "SATELLITE_MODEL_NOT_TRAINED", "message": str(e)})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Satellite prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail="Satellite prediction failed.")


# =========================================================
# Render Entry Point
# =========================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.ml_api:app", host="0.0.0.0", port=port)
