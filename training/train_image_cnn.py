"""
Train EfficientNet-B0 for AGB regression from tree / canopy images.

Architecture
------------
    Image (224x224 RGB)
      -> EfficientNet-B0 (pretrained on ImageNet1K, transfer learning)
      -> regression head (Linear, 1 neuron, no activation)
      -> AGB (kg)

Carbon = AGB * 0.47

Training strategy (transfer learning)
    1. Freeze the EfficientNet-B0 backbone; train only the regression head
       for a few epochs to stabilise the new layers.
    2. Unfreeze all backbone layers and fine-tune end-to-end with a lower
       learning rate.

Loss: MSELoss
    The AGB range is narrow (≈3–16 kg) so MSE gives well-scaled gradients.
    MSELoss is also the most interpretable choice for regression.

Split: group-based by tree_id (80 % train / 10 % val / 10 % test)
    All images of the same tree stay in one split — no leakage.

Usage:
    python ai/training/train_image_cnn.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from torchvision import transforms

# ---------------------------------------------------------------------------
# Path setup — make the script runnable from anywhere
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent            # ai/training/
AI_BASE = SCRIPT_DIR.parent                             # ai/
PROJECT_ROOT = AI_BASE.parent                           # repo root
sys.path.insert(0, str(AI_BASE))

from utils.preprocessing import (                       # noqa: E402
    ImageDataset,
    get_transforms,
    compute_image_diagnostics,
    INPUT_SIZE,
    EFFICIENTNET_MEAN,
    EFFICIENTNET_STD,
)
from utils.carbon import calculate_carbon_stock           # noqa: E402

# Import dataset validator
DATASET_DIR = PROJECT_ROOT / "ai" / "datasets" / "image_agb"
sys.path.insert(0, str(DATASET_DIR))
from validate_dataset import validate_dataset, print_report  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_DIR = PROJECT_ROOT / "ai" / "datasets" / "image_agb"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_CSV = DATASET_DIR / "labels.csv"

MODEL_SAVE_PATH = PROJECT_ROOT / "ai" / "models" / "efficientnet_agb.pth"
METADATA_SAVE_PATH = PROJECT_ROOT / "ai" / "models" / "efficientnet_agb_metadata.json"
EVAL_REPORT_PATH = PROJECT_ROOT / "ai" / "models" / "efficientnet_agb_evaluation.json"

CARBON_FACTOR = 0.47
MODEL_VERSION = "efficientnet-b0-v1"


# ---------------------------------------------------------------------------
# Model definition (must match ai/inference/image_predictor.py)
# ---------------------------------------------------------------------------
class EfficientNetAGB(nn.Module):
    """EfficientNet-B0 with a regression head for AGB prediction."""

    def __init__(self, num_outputs=1):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
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


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def evaluate_model(model, dataloader, criterion, device):
    """Return loss / MAE / RMSE / R2 for a dataloader."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            n_samples += images.size(0)
            all_preds.extend(outputs.squeeze(1).cpu().numpy().tolist())
            all_targets.extend(targets.squeeze(1).cpu().numpy().tolist())

    avg_loss = total_loss / max(n_samples, 1)
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))

    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

    return {
        "loss": float(avg_loss),
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
        "predictions": preds.tolist(),
        "targets": targets.tolist(),
    }


def group_split(labels_df, seed=42):
    """
    Split labels into train / val / test by *tree_id* (group-based).

    Ensures images of the same tree never appear in different splits
    (no data leakage).  Uses GroupShuffleSplit.
    """
    groups = labels_df["tree_id"].values if "tree_id" in labels_df.columns else np.arange(len(labels_df))

    # First split off 20 % (val + test combined)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, val_test_idx = next(gss1.split(labels_df, labels_df["agb_kg"], groups))

    # Split the 20 % into val (10 %) and test (10 %)
    train_groups = groups[train_idx]
    val_test_df = labels_df.iloc[val_test_idx].reset_index(drop=True)
    val_test_groups = groups[val_test_idx]

    # Determine which trees are in val_test and split them
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    vt_train_idx, vt_test_idx = next(
        gss2.split(val_test_df, val_test_df["agb_kg"], val_test_groups)
    )
    val_idx = val_test_idx[vt_train_idx]
    test_idx = val_test_idx[vt_test_idx]

    train_df = labels_df.iloc[train_idx].reset_index(drop=True)
    val_df = labels_df.iloc[val_idx].reset_index(drop=True)
    test_df = labels_df.iloc[test_idx].reset_index(drop=True)

    # Verify no tree_id overlap between splits
    train_trees = set(train_df["tree_id"].unique())
    val_trees = set(val_df["tree_id"].unique())
    test_trees = set(test_df["tree_id"].unique())
    assert not (train_trees & val_trees), "Train/val tree_id overlap!"
    assert not (train_trees & test_trees), "Train/test tree_id overlap!"
    assert not (val_trees & test_trees), "Val/test tree_id overlap!"

    split_summary = {
        "train": len(train_df),
        "validation": len(val_df),
        "test": len(test_df),
        "train_trees": len(train_trees),
        "val_trees": len(val_trees),
        "test_trees": len(test_trees),
    }
    return train_df, val_df, test_df, split_summary


def train_image_model(
    dataset_dir=DATASET_DIR,
    epochs_head=6,
    epochs_finetune=14,
    batch_size=32,
    lr_head=1e-3,
    lr_finetune=3e-5,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # ------------------------------------------------------------------
    # 1. Load and validate dataset
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Dataset Validation")
    print("=" * 60)

    report = validate_dataset(dataset_dir)
    print_report(report)

    if not report["all_valid"]:
        raise RuntimeError("Dataset validation failed. Fix issues before training.")

    labels_df = pd.read_csv(LABELS_CSV)
    print(f"\nLoaded {len(labels_df)} labelled images from {LABELS_CSV}")

    # ------------------------------------------------------------------
    # 2. Split data (group-based by tree_id)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Train/Val/Test Split (group-based by tree_id)")
    print("=" * 60)

    train_df, val_df, test_df, split_summary = group_split(labels_df, seed=seed)
    print(f"  Train:      {split_summary['train']} images ({split_summary['train_trees']} trees)")
    print(f"  Validation: {split_summary['validation']} images ({split_summary['val_trees']} trees)")
    print(f"  Test:       {split_summary['test']} images ({split_summary['test_trees']} trees)")
    print(f"  No tree_id overlap between splits (leakage prevented)")

    # Save split manifests
    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        df.to_csv(DATASET_DIR / f"split_{name}.csv", index=False)
        print(f"  Saved split_{name}.csv ({len(df)} rows)")

    # ------------------------------------------------------------------
    # 3. Create datasets and dataloaders
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Datasets & Dataloaders")
    print("=" * 60)

    transforms_dict = get_transforms(INPUT_SIZE)
    train_dataset = ImageDataset(IMAGES_DIR, train_df, transform=transforms_dict["train"])
    val_dataset = ImageDataset(IMAGES_DIR, val_df, transform=transforms_dict["val"])
    test_dataset = ImageDataset(IMAGES_DIR, test_df, transform=transforms_dict["test"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train batches:      {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches:       {len(test_loader)}")

    # ------------------------------------------------------------------
    # 4. Load model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Model — EfficientNet-B0 (pretrained ImageNet)")
    print("=" * 60)

    model = EfficientNetAGB().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    criterion = nn.MSELoss()
    print(f"  Loss: MSELoss (narrow AGB range 3–16 kg; MSE gives well-scaled gradients)")

    history = []

    # ------------------------------------------------------------------
    # 5. Phase 1 — freeze backbone, train regression head
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 5: Phase 1 — Train regression head (frozen backbone)")
    print(f"         {epochs_head} epochs, lr={lr_head}")
    print("=" * 60)

    # Freeze only the convolutional backbone, keep the regression head trainable
    for p in model.backbone.features.parameters():
        p.requires_grad = False

    head_params = [p for p in model.backbone.classifier.parameters() if p.requires_grad]
    optimizer_head = torch.optim.Adam(head_params, lr=lr_head)

    best_val_mae = float("inf")
    best_val_r2 = -float("inf")
    phase1_start = time.time()

    for epoch in range(epochs_head):
        model.train()
        train_loss_sum = 0.0
        n_train = 0
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device).unsqueeze(1)
            optimizer_head.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_head.step()
            train_loss_sum += loss.item() * images.size(0)
            n_train += images.size(0)

        train_loss = train_loss_sum / max(n_train, 1)
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        history.append({"epoch": epoch + 1, "phase": "head", "train_loss": train_loss, **{k: v for k, v in val_metrics.items() if k != "predictions" and k != "targets"}})

        print(f"  Epoch {epoch+1}/{epochs_head} — "
              f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
              f"val_MAE={val_metrics['mae']:.4f}  val_R2={val_metrics['r2']:.4f}")

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_val_r2 = val_metrics["r2"]
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"    -> New best (val MAE={best_val_mae:.4f}), model saved")

    phase1_time = time.time() - phase1_start

    # ------------------------------------------------------------------
    # 6. Phase 2 — fine-tune all layers
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 6: Phase 2 — Fine-tune (all layers unfrozen)")
    print(f"         {epochs_finetune} epochs, lr={lr_finetune}")
    print("=" * 60)

    # Unfreeze everything for fine-tuning
    for p in model.parameters():
        p.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_params:,} (all unfrozen)")

    optimizer_ft = torch.optim.Adam(model.parameters(), lr=lr_finetune, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ft, mode="min", factor=0.5, patience=3
    )

    phase2_start = time.time()
    best_val_loss = float("inf")

    for epoch in range(epochs_finetune):
        model.train()
        train_loss_sum = 0.0
        n_train = 0
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device).unsqueeze(1)
            optimizer_ft.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_ft.step()
            train_loss_sum += loss.item() * images.size(0)
            n_train += images.size(0)

        train_loss = train_loss_sum / max(n_train, 1)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        epoch_record = {"epoch": epoch + 1, "phase": "finetune", "train_loss": train_loss, **{k: v for k, v in val_metrics.items() if k != "predictions" and k != "targets"}}
        history.append(epoch_record)

        print(f"  Epoch {epoch+1}/{epochs_finetune} — "
              f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
              f"val_MAE={val_metrics['mae']:.4f}  val_R2={val_metrics['r2']:.4f}  "
              f"lr={optimizer_ft.param_groups[0]['lr']:.6f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"    -> New best (val_loss={best_val_loss:.4f}), model saved")

    phase2_time = time.time() - phase2_start

    # ------------------------------------------------------------------
    # 7. Load best model and evaluate on test set
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: Test-set Evaluation (best checkpoint)")
    print("=" * 60)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    test_metrics = evaluate_model(model, test_loader, criterion, device)

    print(f"  Test Loss:  {test_metrics['loss']:.4f}")
    print(f"  Test MAE:   {test_metrics['mae']:.4f} kg")
    print(f"  Test RMSE:  {test_metrics['rmse']:.4f} kg")
    print(f"  Test R2:    {test_metrics['r2']:.4f}")

    # Prediction vs actual sample
    preds = np.array(test_metrics["predictions"])
    targets = np.array(test_metrics["targets"])
    print("\n  Sample predictions vs actual (first 10):")
    print(f"  {'Predicted':>10} {'Actual':>10} {'Error':>10}")
    for i in range(min(10, len(preds))):
        err = preds[i] - targets[i]
        print(f"  {preds[i]:10.4f} {targets[i]:10.4f} {err:10.4f}")

    # ------------------------------------------------------------------
    # 8. Save metadata
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8: Saving model & metadata")
    print("=" * 60)

    metadata = {
        "model": "EfficientNet-B0",
        "architecture": "EfficientNet-B0 CNN",
        "task": "AGB Regression",
        "target": "AGB",
        "unit": "kg",
        "carbon_factor": CARBON_FACTOR,
        "modelVersion": MODEL_VERSION,
        "input_size": INPUT_SIZE,
        "pretrained": True,
        "pretrained_weights": "EfficientNet_B0_Weights.IMAGENET1K_V1",
        "trained_on_campus_data": False,
        "dataset_source": str(DATASET_DIR),
        "dataset_description": "Synthetic tree/canopy images with AGB encoded as visual features (canopy density, trunk girth, leaf coverage). Generated deterministically from field-measured AGB labels. General-purpose for tree/canopy images.",
        "loss_function": "MSELoss",
        "loss_rationale": "Narrow AGB range (3-16 kg) gives well-scaled gradients; MSE is interpretable for regression.",
        "status": "trained",
        "split_strategy": "group_shuffle_split by tree_id (80/10/10), no leakage",
        "training": {
            "seed": seed,
            "batch_size": batch_size,
            "phase1": {
                "description": "Frozen backbone, train regression head",
                "epochs": epochs_head,
                "learning_rate": lr_head,
                "elapsed_seconds": round(phase1_time, 1),
            },
            "phase2": {
                "description": "Fine-tune all layers (unfrozen)",
                "epochs": epochs_finetune,
                "learning_rate": lr_finetune,
                "weight_decay": 1e-5,
                "elapsed_seconds": round(phase2_time, 1),
            },
            "total_epochs": epochs_head + epochs_finetune,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=3)",
            "augmentation": [
                "RandomResizedCrop(224, scale=0.8-1.0)",
                "RandomHorizontalFlip(p=0.5)",
                "RandomRotation(15 degrees)",
                "ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)",
            ],
        },
        "dataset_size": len(labels_df),
        "train_size": len(train_df),
        "validation_size": len(val_df),
        "test_size": len(test_df),
        "train_trees": split_summary["train_trees"],
        "validation_trees": split_summary["val_trees"],
        "test_trees": split_summary["test_trees"],
        "metrics": {
            "r2": test_metrics["r2"],
            "mae": test_metrics["mae"],
            "rmse": test_metrics["rmse"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "test_loss": test_metrics["loss"],
        },
        "dataset_stats": {
            "agb_min": float(labels_df["agb_kg"].min()),
            "agb_max": float(labels_df["agb_kg"].max()),
            "agb_mean": float(labels_df["agb_kg"].mean()),
            "agb_median": float(labels_df["agb_kg"].median()),
        },
    }

    with open(METADATA_SAVE_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {METADATA_SAVE_PATH}")

    # Save evaluation report
    eval_report = {
        "model": "EfficientNet-B0",
        "modelVersion": MODEL_VERSION,
        "input_size": INPUT_SIZE,
        "carbon_factor": CARBON_FACTOR,
        "dataset_size": len(labels_df),
        "split": split_summary,
        "test_metrics": {
            "r2": test_metrics["r2"],
            "mae": test_metrics["mae"],
            "rmse": test_metrics["rmse"],
            "loss": test_metrics["loss"],
        },
        "sample_predictions": [
            {"predicted": float(p), "actual": float(t), "error": float(p - t)}
            for p, t in zip(preds[:20], targets[:20])
        ],
        "validation_metrics": {
            "r2": val_metrics["r2"],
            "mae": val_metrics["mae"],
            "rmse": val_metrics["rmse"],
            "loss": val_metrics["loss"],
        },
        "training_history": history,
    }
    with open(EVAL_REPORT_PATH, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"  Evaluation report saved to {EVAL_REPORT_PATH}")

    # Verify model file is non-empty
    if MODEL_SAVE_PATH.exists():
        size = MODEL_SAVE_PATH.stat().st_size
        print(f"  Model checkpoint size: {size:,} bytes")
        assert size > 0, "Model file is empty!"
    else:
        raise RuntimeError("Model checkpoint was not saved!")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model:    {MODEL_SAVE_PATH}")
    print(f"  Metadata: {METADATA_SAVE_PATH}")
    print(f"  Eval:     {EVAL_REPORT_PATH}")
    print(f"  Test R2:   {test_metrics['r2']:.4f}")
    print(f"  Test MAE:  {test_metrics['mae']:.4f} kg")
    print(f"  Test RMSE: {test_metrics['rmse']:.4f} kg")
    return metadata


if __name__ == "__main__":
    train_image_model()
