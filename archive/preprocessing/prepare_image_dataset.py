import shutil
import random
from pathlib import Path

from PIL import Image

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE_LIMITS = (32, 32)
TEST_RATIO = 0.15
VAL_RATIO = 0.15
SEED = 42


def validate_images(class_dir):
    valid = []
    invalid = []

    for image_path in sorted(class_dir.glob("*")):
        if image_path.is_dir():
            continue

        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            invalid.append((image_path.name, "unsupported extension"))
            continue

        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as exc:
            invalid.append((image_path.name, str(exc)))
            continue

        try:
            with Image.open(image_path) as img:
                img.load()
                width, height = img.size
        except Exception as exc:
            invalid.append((image_path.name, str(exc)))
            continue

        if width < IMAGE_SIZE_LIMITS[0] or height < IMAGE_SIZE_LIMITS[1]:
            invalid.append((image_path.name, f"too small: {width}x{height}"))
            continue

        valid.append(image_path)

    return valid, invalid


def split_images(valid_images, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, seed=SEED):
    random.seed(seed)
    images = list(valid_images)
    random.shuffle(images)

    total = len(images)
    test_count = int(total * test_ratio)
    val_count = int(total * val_ratio)
    train_count = total - test_count - val_count

    train = images[:train_count]
    validation = images[train_count:train_count + val_count]
    test = images[train_count + val_count:]

    return train, validation, test


def copy_images(images, destination_dir):
    destination_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        shutil.copy2(image_path, destination_dir / image_path.name)


def prepare_dataset(raw_root, output_root):
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    class_names = sorted([d.name for d in raw_root.iterdir() if d.is_dir()])

    report_lines = ["IMAGE DATASET CHECK", "─" * 40]
    total_valid = 0
    total_invalid = 0

    split_plan = {}

    for class_name in class_names:
        class_dir = raw_root / class_name
        valid, invalid = validate_images(class_dir)

        report_lines.append(f"{class_name:25s}: {len(valid)}")
        total_valid += len(valid)
        total_invalid += len(invalid)

        if invalid:
            for name, reason in invalid:
                report_lines.append(f"  INVALID: {name} ({reason})")

        if len(valid) == 0:
            raise ValueError(f"No valid images found in {class_dir}")

        train, validation, test = split_images(valid)
        split_plan[class_name] = (train, validation, test)

        report_lines.append(
            f"  train={len(train)}, validation={len(validation)}, test={len(test)}"
        )

        for split_name, split_images in zip(
            ["train", "validation", "test"], [train, validation, test]
        ):
            destination = output_root / split_name / class_name
            copy_images(split_images, destination)

    report_lines.append("─" * 40)
    report_lines.append(f"Valid images            : {total_valid}")
    report_lines.append(f"Invalid images          : {total_invalid}")
    report_lines.append(f"Image classes           : {len(class_names)}")
    report_lines.append("")

    print("\n".join(report_lines))

    return split_plan


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_root = project_root / "ai" / "image_data" / "raw"
    output_root = project_root / "ai" / "image_data"

    if not raw_root.exists():
        raise FileNotFoundError(
            f"Raw image folder not found: {raw_root}\n"
            "Please place your labeled images in ai/image_data/raw/<class_name>/"
        )

    prepare_dataset(raw_root, output_root)
    print("Dataset prepared successfully.")
