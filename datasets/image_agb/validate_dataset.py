"""
Dataset validation for image → AGB regression.

Performs all checks required by the specification (section 6) and prints a
summary report.  Can be run standalone or imported as a function by the
training pipeline.

Usage (standalone):

    python validate_dataset.py --dataset-dir ai/datasets/image_agb

Exit codes:
    0  - all images valid, dataset ready for training
    1  - one or more critical issues found
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def validate_dataset(dataset_dir: str | Path):
    """
    Validate a labelled image dataset.

    Returns a dict with the full report including:
      - total images, valid images, invalid images
      - min / max / mean / median AGB
      - list of invalid samples with reasons
      - duplicate detection
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    labels_csv = dataset_dir / "labels.csv"

    report = {
        "dataset_dir": str(dataset_dir),
        "images_dir": str(images_dir),
        "labels_csv": str(labels_csv),
        "total_images": 0,
        "valid_images": 0,
        "invalid_images": 0,
        "agb_min": None,
        "agb_max": None,
        "agb_mean": None,
        "agb_median": None,
        "duplicate_image_entries": 0,
        "duplicate_tree_ids": 0,
        "missing_labels": 0,
        "invalid_samples": [],
        "image_size_stats": {},
        "all_valid": False,
    }

    # ------------------------------------------------------------------
    # 0. Check labels CSV exists
    # ------------------------------------------------------------------
    if not labels_csv.exists():
        report["error"] = f"labels.csv not found at {labels_csv}"
        report["invalid_samples"].append({"row": "N/A", "reason": report["error"]})
        return report

    df = pd.read_csv(labels_csv)
    required_cols = {"image", "agb_kg"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        report["error"] = f"labels.csv missing required columns: {missing_cols}"
        report["invalid_samples"].append({"row": "N/A", "reason": report["error"]})
        return report

    # tree_id is needed for split grouping
    has_tree_id = "tree_id" in df.columns
    report["has_tree_id"] = has_tree_id

    # ------------------------------------------------------------------
    # 1. Detect duplicate image entries
    # ------------------------------------------------------------------
    dup_mask = df["image"].duplicated(keep=False)
    if dup_mask.any():
        dup_images = df.loc[dup_mask, "image"].unique().tolist()
        report["duplicate_image_entries"] = len(dup_images)
        for img in dup_images:
            report["invalid_samples"].append(
                {"image": img, "reason": "duplicate image entry in labels.csv"}
            )

    # ------------------------------------------------------------------
    # 2. Detect duplicate filenames on disk (case-insensitive)
    # ------------------------------------------------------------------
    if labels_csv.exists():
        # Check for case-insensitive filename collisions
        seen_lower = {}
        for fname in df["image"].astype(str):
            fl = fname.lower()
            if fl in seen_lower:
                report["invalid_samples"].append(
                    {"image": fname, "reason": f"case-insensitive collision with {seen_lower[fl]}"}
                )
            seen_lower[fl] = fname

    # ------------------------------------------------------------------
    # 3. Validate each row
    # ------------------------------------------------------------------
    total = len(df)
    valid = 0
    invalid = 0
    agb_values = []
    img_sizes = []

    for idx, row in df.iterrows():
        image_name = str(row["image"])
        agb_val = row["agb_kg"]

        # Check AGB is numeric
        try:
            agb_float = float(agb_val)
        except (ValueError, TypeError):
            report["invalid_samples"].append(
                {"image": image_name, "row": idx, "reason": f"AGB is not numeric: {agb_val!r}"}
            )
            invalid += 1
            continue

        # Check AGB is positive
        if agb_float <= 0:
            report["invalid_samples"].append(
                {"image": image_name, "row": idx, "reason": f"AGB is not positive: {agb_float}"}
            )
            invalid += 1
            continue

        agb_values.append(agb_float)

        # Check image file exists
        img_path = images_dir / image_name
        if not img_path.exists():
            report["missing_labels"] += 1
            report["invalid_samples"].append(
                {"image": image_name, "row": idx, "reason": "image file does not exist"}
            )
            invalid += 1
            continue

        # Check valid image extension
        ext = img_path.suffix.lower()
        if ext not in VALID_EXTENSIONS:
            report["invalid_samples"].append(
                {"image": image_name, "row": idx, "reason": f"unsupported extension: {ext}"}
            )
            invalid += 1
            continue

        # Check image readability
        try:
            with Image.open(img_path) as img:
                img.verify()
            # Re-open for mode/size check (verify() closes the file)
            with Image.open(img_path) as img:
                if img.mode != "RGB":
                    pass  # will be converted downstream
                img_sizes.append(img.size)
            valid += 1
        except Exception as e:
            report["invalid_samples"].append(
                {"image": image_name, "row": idx, "reason": f"image not readable: {e}"}
            )
            invalid += 1
            continue

    # ------------------------------------------------------------------
    # 4. Compute summary statistics
    # ------------------------------------------------------------------
    report["total_images"] = total
    report["valid_images"] = valid
    report["invalid_images"] = invalid

    if agb_values:
        arr = np.array(agb_values)
        report["agb_min"] = float(arr.min())
        report["agb_max"] = float(arr.max())
        report["agb_mean"] = float(arr.mean())
        report["agb_median"] = float(np.median(arr))

    if img_sizes:
        report["image_size_stats"] = {
            "min_width": min(s[0] for s in img_sizes),
            "max_width": max(s[0] for s in img_sizes),
            "min_height": min(s[1] for s in img_sizes),
            "max_height": max(s[1] for s in img_sizes),
        }

    # Duplicate tree_ids with conflicting AGB
    if has_tree_id:
        tree_agb = df.groupby("tree_id")["agb_kg"].nunique()
        conflicting = tree_agb[tree_agb > 1]
        report["duplicate_tree_ids"] = len(conflicting)
        if len(conflicting) > 0:
            for tid in conflicting.index:
                report["invalid_samples"].append(
                    {"image": f"tree_id={tid}", "reason": "tree_id has conflicting AGB values"}
                )

    report["all_valid"] = (valid == total) and not report["duplicate_image_entries"]

    return report


def print_report(report: dict):
    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Dataset dir:     {report['dataset_dir']}")
    print(f"Labels CSV:      {report['labels_csv']}")
    print(f"Images dir:      {report['images_dir']}")
    print(f"Total images:    {report['total_images']}")
    print(f"Valid images:    {report['valid_images']}")
    print(f"Invalid images:  {report['invalid_images']}")
    print(f"Duplicate entries: {report['duplicate_image_entries']}")
    print(f"Missing labels:  {report['missing_labels']}")
    print(f"Duplicate tree_ids (conflicting AGB): {report['duplicate_tree_ids']}")
    if report.get("has_tree_id"):
        print(f"Has tree_id:     Yes (split-grouping enabled)")
    else:
        print(f"Has tree_id:     No (cannot guarantee split grouping)")

    if report.get("agb_min") is not None:
        print("-" * 60)
        print("AGB Statistics:")
        print(f"  Min:    {report['agb_min']:.4f} kg")
        print(f"  Max:    {report['agb_max']:.4f} kg")
        print(f"  Mean:   {report['agb_mean']:.4f} kg")
        print(f"  Median: {report['agb_median']:.4f} kg")

    if report.get("image_size_stats"):
        print("-" * 60)
        print("Image Size Stats:")
        s = report["image_size_stats"]
        print(f"  Width:  {s['min_width']} - {s['max_width']}")
        print(f"  Height: {s['min_height']} - {s['max_height']}")

    if report["invalid_samples"]:
        print("-" * 60)
        print(f"Invalid samples ({len(report['invalid_samples'])}):")
        for sample in report["invalid_samples"][:20]:
            img = sample.get("image", sample.get("row", "?"))
            print(f"  {img}: {sample['reason']}")
        if len(report["invalid_samples"]) > 20:
            print(f"  ... and {len(report['invalid_samples']) - 20} more")

    print("=" * 60)
    if report["all_valid"]:
        print("RESULT: ALL VALID - Dataset is ready for training.")
    else:
        print("RESULT: ISSUES FOUND - Review invalid samples above.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate image AGB dataset")
    parser.add_argument(
        "--dataset-dir",
        default=str(Path(__file__).resolve().parent),
        help="Path to the dataset directory (containing images/ and labels.csv)",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    report = validate_dataset(args.dataset_dir)

    if args.json:
        import json
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)

    sys.exit(0 if report["all_valid"] else 1)
