import shutil
import random
from pathlib import Path
import pandas as pd


def prepare_image_dataset(
    source_dir,
    labels_csv,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    random.seed(seed)

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    labels_csv = Path(labels_csv)

    df = pd.read_csv(labels_csv)

    required_cols = {'image_id', 'agb_kg'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"labels_csv must contain columns: {required_cols}")

    image_ids = df['image_id'].tolist()
    random.shuffle(image_ids)

    n_total = len(image_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:n_train + n_val]
    test_ids = image_ids[n_train + n_val:]

    splits = {
        'train': train_ids,
        'validation': val_ids,
        'test': test_ids
    }

    for split_name, ids in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        split_df = df[df['image_id'].isin(ids)].copy()
        split_df.to_csv(split_dir / 'labels.csv', index=False)

        for img_id in ids:
            candidates = [
                source_dir / f"{img_id}.jpg",
                source_dir / f"{img_id}.jpeg",
                source_dir / f"{img_id}.png",
            ]
            for src in candidates:
                if src.exists():
                    dst = split_dir / src.name
                    shutil.copy2(src, dst)
                    break

    split_summary = {
        'train': len(train_ids),
        'validation': len(val_ids),
        'test': len(test_ids),
        'total': n_total
    }

    print(f"Dataset split complete: {split_summary}")
    return split_summary


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent.parent
    prepare_image_dataset(
        source_dir=base_dir / 'datasets' / 'image' / 'raw',
        labels_csv=base_dir / 'datasets' / 'image' / 'image_labels.csv',
        output_dir=base_dir / 'datasets' / 'image'
    )
