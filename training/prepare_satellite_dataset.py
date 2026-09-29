import shutil
import random
from pathlib import Path
import pandas as pd


def prepare_satellite_dataset(
    source_dir,
    labels_csv,
    output_dir,
    bands=None,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    random.seed(seed)

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    labels_csv = Path(labels_csv)

    if bands is None:
        bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']

    df = pd.read_csv(labels_csv)

    required_cols = {'patch_id', 'agb_kg'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"labels_csv must contain columns: {required_cols}")

    patch_ids = df['patch_id'].tolist()
    random.shuffle(patch_ids)

    n_total = len(patch_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = patch_ids[:n_train]
    val_ids = patch_ids[n_train:n_train + n_val]
    test_ids = patch_ids[n_train + n_val:]

    splits = {
        'train': train_ids,
        'validation': val_ids,
        'test': test_ids
    }

    for split_name, ids in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        split_df = df[df['patch_id'].isin(ids)].copy()
        split_df.to_csv(split_dir / 'labels.csv', index=False)

        for patch_id in ids:
            for band in bands:
                candidates = [
                    source_dir / f"{patch_id}_{band}.tif",
                    source_dir / f"{patch_id}_{band}.jpg",
                    source_dir / f"{patch_id}_{band}.png",
                    source_dir / f"{patch_id}_{band.lower()}.tif",
                    source_dir / f"{patch_id}_{band.lower()}.jpg",
                    source_dir / f"{patch_id}_{band.lower()}.png",
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
        'total': n_total,
        'bands': bands
    }

    print(f"Satellite dataset split complete: {split_summary}")
    return split_summary


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent.parent
    prepare_satellite_dataset(
        source_dir=base_dir / 'datasets' / 'satellite' / 'raw',
        labels_csv=base_dir / 'datasets' / 'satellite' / 'satellite_labels.csv',
        output_dir=base_dir / 'datasets' / 'satellite'
    )
