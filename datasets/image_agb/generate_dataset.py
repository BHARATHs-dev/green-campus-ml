"""
Synthetic Tree / Canopy Image Dataset Generator.

This script creates a labelled image dataset for AGB (above-ground biomass)
regression.  Each image is procedurally rendered so that its visual features
- canopy area, trunk girth, leaf density, vertical extent - are a *deterministic*
function of the tree's ground-truth AGB value.

This is NOT random pairing: the AGB is the generative ground-truth for each
image.  Multiple view variations (camera angle, lighting, background) are
produced per tree so that the same tree can never leak between train/val/test
splits (grouping by tree_id).

The AGB values come from the project's existing ai/dsets/image/image_labels.csv
which contains real field-measured biomass values for 30 trees.

Output structure:
    ai/datasets/image_agb/
        images/
            image_0000.jpg
            ...
        labels.csv          (image,agb_kg,tree_id)
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

BASE_DIR = Path(__file__).resolve().parent
SOURCE_LABELS = (
    Path(__file__).resolve().parent.parent
    / "image"
    / "image_labels.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_CSV = OUTPUT_DIR / "labels.csv"

IMG_SIZE = 224
VIEWS_PER_TREE = 20
SEED = 42

# AGB range in the source data
MIN_AGB = 2.0
MAX_AGB = 16.0


def _agb_to_seed(tree_id: str, view_idx: int) -> int:
    """Derive a deterministic numpy seed from tree_id + view index."""
    raw = hashlib.sha256(f"{tree_id}_{view_idx}".encode()).digest()
    return int.from_bytes(raw[:4], "little")


def _sky_gradient(draw: ImageDraw, size: int):
    """Draw a vertical sky gradient in the top 65 % of the canvas."""
    top_color = (135, 206, 235)   # lightskyblue
    bot_color = (176, 224, 230)   # powderblue
    horizon = int(size * 0.65)
    for y in range(horizon):
        t = y / max(horizon - 1, 1)
        r = int(top_color[0] + (bot_color[0] - top_color[0]) * t)
        g = int(top_color[1] + (bot_color[1] - top_color[1]) * t)
        b = int(top_color[2] + (bot_color[2] - top_color[2]) * t)
        draw.line([(0, y), (size, y)], fill=(r, g, b))


def _ground_gradient(draw: ImageDraw, size: int):
    """Draw a horizontal ground band in the bottom 35 % of the canvas."""
    horizon = int(size * 0.65)
    top_color = (60, 120, 60)    # dark green
    bot_color = (80, 150, 80)    # medium green
    for y in range(horizon, size):
        t = (y - horizon) / max(size - horizon - 1, 1)
        r = int(top_color[0] + (bot_color[0] - top_color[0]) * t)
        g = int(top_color[1] + (bot_color[1] - top_color[1]) * t)
        b = int(top_color[2] + (bot_color[2] - top_color[3]) * t) if False else top_color[2]
        draw.line([(0, y), (size, y)], fill=(r, g, b))
    # Grass tufts
    for _ in range(40):
        gx = np.random.randint(0, size)
        gy = np.random.randint(horizon, size)
        gw = np.random.randint(2, 6)
        gh = np.random.randint(4, 10)
        draw.ellipse(
            [gx, gy, gx + gw, gy + gh],
            fill=(np.random.randint(40, 90), np.random.randint(120, 180), 40),
        )


def _trunk(draw: ImageDraw, cx: int, ground_y: int, agb: float, rng: np.random.RandomState):
    """Draw a brown tree trunk whose girth scales with AGB."""
    # Trunk width proportional to AGB (proxy for DBH)
    trunk_w = 5 + (agb / MAX_AGB) * 12          # 5-17 px
    trunk_h = 25 + (agb / MAX_AGB) * 50          # 25-75 px
    # Tapered trunk: narrow at top, wide at bottom
    x0 = cx - trunk_w
    x1 = cx + trunk_w
    y_top = ground_y - trunk_h
    brown_base = (101, 67, 33)
    # Draw trunk as stacked rectangles tapering toward top
    steps = int(trunk_h / 4)
    for i in range(steps):
        t = i / max(steps - 1, 1)
        taper = 1 - t * 0.3
        iw = max(1, int(trunk_w * taper))
        y_top = ground_y - int(trunk_h * (i + 1) / steps)
        y_bottom = ground_y - int(trunk_h * i / steps)
        brown_var = tuple(
            max(0, min(255, c + rng.randint(-15, 15))) for c in brown_base
        )
        draw.rectangle([cx - iw, y_top, cx + iw, y_bottom], fill=brown_var)
    return trunk_w


def _canopy(draw: ImageDraw, cx: int, base_y: int, agb: float, rng: np.random.RandomState):
    """
    Draw green canopy blobs above the trunk.
    Total canopy area and density scale with AGB.
    """
    # Normalise AGB to 0-1
    a = (agb - MIN_AGB) / max(MAX_AGB - MIN_AGB, 1e-6)

    # Number of blob layers increases with AGB
    n_layers = 3 + int(a * 5)  # 3-8 layers
    # Total canopy spread (radius) scales sub-linearly with AGB
    canopy_r = 22 + a * 40     # 22-62 px radius

    green_base = (34, 139, 34)   # forestgreen

    for layer in range(n_layers):
        # Each layer is a cluster of overlapping ellipses
        layer_r = canopy_r * (0.6 + layer * 0.15)
        n_blobs = 4 + int(a * 8)   # 4-12 blobs per layer
        for _ in range(n_blobs):
            # Position blobs in a ring around the trunk centre
            angle = rng.uniform(0, 2 * np.pi)
            ring_r = layer_r * rng.uniform(0.3, 0.9)
            bx = cx + int(np.cos(angle) * ring_r)
            by = base_y + int(np.sin(angle) * rng.uniform(-layer_r * 0.3, layer_r * 0.3))
            blob_r = layer_r * rng.uniform(0.4, 0.8)
            # Green intensity: denser foliage for higher AGB
            sat = int(a * 60)
            green = (
                max(0, min(255, green_base[0] - rng.randint(0, 20))),
                max(0, min(255, green_base[1] - rng.randint(0, 20) + sat)),
                max(0, min(255, green_base[2] - rng.randint(0, 20))),
            )
            draw.ellipse(
                [bx - blob_r, by - blob_r, bx + blob_r, by + blob_r],
                fill=green,
            )

    # Leaf texture: small dots for foliage density
    n_dots = 200 + int(a * 600)
    for _ in range(n_dots):
        angle = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0, canopy_r)
        dx = cx + int(np.cos(angle) * r)
        dy = base_y + int(np.sin(angle) * r * 0.6)
        dot_size = rng.randint(1, 4)
        draw.ellipse(
            [dx - dot_size, dy - dot_size, dx + dot_size, dy + dot_size],
            fill=(
                max(0, min(255, green_base[0] + rng.randint(-10, 10))),
                max(0, min(255, green_base[1] + rng.randint(-10, 10) + sat)),
                max(0, min(255, green_base[2] + rng.randint(-10, 10))),
            ),
        )


def _add_noise(arr: np.ndarray, rng: np.random.RandomState, intensity: float = 8.0):
    """Add subtle Gaussian noise for texture realism."""
    noise = rng.normal(0, intensity, arr.shape)
    return np.clip(arr + noise, 0, 255).astype(np.uint8)


def generate_tree_image(agb: float, tree_id: str, view_idx: int) -> Image.Image:
    """Render a single tree image from an AGB label."""
    seed = _agb_to_seed(tree_id, view_idx)
    rng = np.random.RandomState(seed)

    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)

    # Background
    if rng.random() < 0.7:
        _sky_gradient(draw, IMG_SIZE)
    else:
        # Overcast sky variant
        for y in range(int(IMG_SIZE * 0.7)):
            draw.line([(0, y), (IMG_SIZE, y)], fill=(180, 190, 200))
    _ground_gradient(draw, IMG_SIZE)

    horizon = int(IMG_SIZE * 0.65)
    ground_y = horizon + rng.randint(5, 15)

    # Tree centre with slight horizontal jitter
    cx = IMG_SIZE // 2 + rng.randint(-12, 12)

    # Trunk
    trunk_w = _trunk(draw, cx, ground_y, agb, rng)

    # Canopy starts just above the trunk
    canopy_base_y = ground_y - trunk_w * 2 - rng.randint(0, 10)
    _canopy(draw, cx, canopy_base_y, agb, rng)

    # Convert to array for noise + effects
    arr = np.array(img)
    arr = _add_noise(arr, rng, intensity=rng.uniform(4, 10))

    img = Image.fromarray(arr)

    # Brightness / contrast variation (lighting)
    brightness = rng.uniform(0.85, 1.20)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    contrast = rng.uniform(0.85, 1.15)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # Slight rotation (camera angle)
    angle = rng.uniform(-25, 25)
    img = img.rotate(
        angle,
        resample=Image.BICUBIC,
        fillcolor=(135, 206, 235),
    )

    # Blur occasionally to simulate distance
    if rng.random() < 0.3:
        radius = rng.uniform(0.3, 0.8)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    return img


def main():
    np.random.seed(SEED)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE_LABELS.exists():
        raise FileNotFoundError(
            f"Source labels not found at {SOURCE_LABELS}. "
            "The existing image_labels.csv with tree_id + agb_kg is required."
        )

    df = pd.read_csv(SOURCE_LABELS)
    print(f"Loaded {len(df)} tree labels from {SOURCE_LABELS}")
    print(f"Columns: {list(df.columns)}")
    print(f"AGB range: {df['agb_kg'].min():.2f} - {df['agb_kg'].max():.2f} kg")

    records = []
    img_idx = 0
    for _, row in df.iterrows():
        tree_id = str(row["tree_id"])
        agb = float(row["agb_kg"])
        for view in range(VIEWS_PER_TREE):
            fname = f"image_{img_idx:04d}.jpg"
            img = generate_tree_image(agb, tree_id, view)
            img.save(IMAGES_DIR / fname, "JPEG", quality=85)
            records.append({"image": fname, "agb_kg": agb, "tree_id": tree_id})
            img_idx += 1

    labels_df = pd.DataFrame(records)
    labels_df.to_csv(LABELS_CSV, index=False)

    print(f"\nDataset generated:")
    print(f"  Images: {len(records)}")
    print(f"  Trees:  {df['tree_id'].nunique()}")
    print(f"  Views per tree: {VIEWS_PER_TREE}")
    print(f"  Images dir: {IMAGES_DIR}")
    print(f"  Labels CSV: {LABELS_CSV}")
    print(f"  AGB range: {labels_df['agb_kg'].min():.2f} - {labels_df['agb_kg'].max():.2f} kg")
    print(f"  AGB mean:  {labels_df['agb_kg'].mean():.2f} kg")


if __name__ == "__main__":
    main()
