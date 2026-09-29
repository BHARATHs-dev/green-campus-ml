"""
PyTorch Dataset / transforms for image-based AGB regression.

Supports the labelled image dataset at ai/datasets/image_agb/ with a
labels.csv containing at minimum:  image, agb_kg  (and optionally tree_id).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms


EFFICIENTNET_MEAN = [0.485, 0.456, 0.406]
EFFICIENTNET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


class ImageDataset(Dataset):
    """
    Dataset mapping tree/canopy images to ground-truth AGB values.

    labels_csv must contain an ``image`` column (filename) and an ``agb_kg``
    column.  An optional ``tree_id`` column is used for group-based splitting.
    """

    def __init__(self, root_dir, labels_df, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Accept either a DataFrame or a CSV path
        if isinstance(labels_df, str) or isinstance(labels_df, Path):
            labels_df = pd.read_csv(labels_df)

        # Normalise column name: image_id -> image
        if "image_id" in labels_df.columns and "image" not in labels_df.columns:
            labels_df = labels_df.rename(columns={"image_id": "image"})

        if "image" not in labels_df.columns:
            raise ValueError("labels CSV must contain an 'image' column")
        if "agb_kg" not in labels_df.columns:
            raise ValueError("labels CSV must contain an 'agb_kg' column")

        self.labels_df = labels_df.reset_index(drop=True)
        self.tree_id_col = "tree_id" if "tree_id" in labels_df.columns else None

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_name = str(row["image"])
        img_path = self.root_dir / img_name

        image = Image.open(img_path).convert("RGB")
        label = float(row["agb_kg"])

        if self.transform:
            image = self.transform(image)

        tree_id = str(row[self.tree_id_col]) if self.tree_id_col else None
        return image, torch.tensor(label, dtype=torch.float32), tree_id


def get_transforms(input_size=INPUT_SIZE):
    """
    Return train/val/test transforms.

    Train: resize + random crop + flip + rotation + colour jitter + normalisation
    Val/Test: deterministic resize + normalisation

    Normalisation uses ImageNet statistics expected by EfficientNet-B0.
    """
    normalize = transforms.Normalize(mean=EFFICIENTNET_MEAN, std=EFFICIENTNET_STD)

    train_transform = transforms.Compose([
        transforms.Resize((input_size + 32, input_size + 32)),
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return {
        "train": train_transform,
        "val": val_transform,
        "test": test_transform,
    }


def compute_image_diagnostics(image_path):
    """
    Compute interpretable image statistics for diagnostics (requirement 22).

    These are NOT fed into the model — they are for display / quality checks only.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mean_rgb = {
        "r": float(r.mean()),
        "g": float(g.mean()),
        "b": float(b.mean()),
    }

    # Green pixel ratio: pixels where green channel is dominant
    green_mask = (g > r) & (g > b) & (g > 0.3)
    green_ratio = float(green_mask.mean())

    # Brightness (luminance)
    brightness = float((0.299 * r + 0.587 * g + 0.114 * b).mean())

    # Contrast (std of luminance)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    contrast = float(lum.std())

    # Check if completely blank (all pixels same value)
    is_blank = (lum.std() < 1e-6)

    # Check if extremely dark
    is_extremely_dark = (brightness < 0.05)

    width, height = img.size
    aspect_ratio = width / height

    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(aspect_ratio, 4),
        "mean_rgb": {k: round(v, 4) for k, v in mean_rgb.items()},
        "green_pixel_ratio": round(green_ratio, 4),
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "is_blank": bool(is_blank),
        "is_extremely_dark": bool(is_extremely_dark),
        "quality_warning": bool(is_blank or is_extremely_dark or width < 64 or height < 64),
    }
