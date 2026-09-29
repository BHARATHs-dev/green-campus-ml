import os
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IMAGE_SIZE = 64
PATCH_SIZE = 8
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
MLP_RATIO = 2
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

MODEL_DIR = "ai/models"
MODEL_PATH = os.path.join(MODEL_DIR, "vegetation_vit.pth")
HISTORY_PATH = os.path.join(MODEL_DIR, "vit_training_history.json")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "vit_confusion_matrix.png")
CLASS_NAMES = ["healthy", "moderately_degraded", "severely_degraded"]


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=PATCH_SIZE, in_channels=3, embed_dim=EMBED_DIM):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=PATCH_SIZE, in_channels=3,
                 embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                 mlp_ratio=MLP_RATIO, num_classes=NUM_CLASSES):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)
        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm(x)
        cls_output = x[:, 0]
        x = self.head(cls_output)
        return x


class SyntheticVegetationDataset(Dataset):
    def __init__(self, num_samples_per_class=100, image_size=IMAGE_SIZE, transform=None, class_names=CLASS_NAMES, seed=42):
        self.transform = transform
        self.class_names = class_names
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.images = []
        self.labels = []
        rng = np.random.RandomState(seed)
        for class_idx, class_name in enumerate(class_names):
            for _ in range(num_samples_per_class):
                if class_name == "healthy":
                    base_color = np.array([80, 160, 60], dtype=np.uint8)
                elif class_name == "moderately_degraded":
                    base_color = np.array([180, 170, 80], dtype=np.uint8)
                else:
                    base_color = np.array([160, 100, 50], dtype=np.uint8)
                noise = rng.randint(-30, 30, (*image_size, 3), dtype=np.int16)
                img_array = np.clip(base_color + noise, 0, 255).astype(np.uint8)
                self.images.append(Image.fromarray(img_array))
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def build_transforms(image_size=IMAGE_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def plot_training_history(history, save_path=os.path.join(MODEL_DIR, "vit_training_history.png")):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["train_acc"], label="Train Accuracy", marker="o")
    axes[0].plot(history["val_acc"], label="Validation Accuracy", marker="s")
    axes[0].set_title("ViT Training & Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["train_loss"], label="Train Loss", marker="o")
    axes[1].plot(history["val_loss"], label="Validation Loss", marker="s")
    axes[1].set_title("ViT Training & Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training history plot saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=CONFUSION_MATRIX_PATH):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.title("ViT Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def main():
    print("=" * 60)
    print("VEGETATION VISION TRANSFORMER (ViT) PROTOTYPE")
    print("=" * 60)

    train_transform, eval_transform = build_transforms()

    print("\nGenerating synthetic dataset...")
    train_dataset = SyntheticVegetationDataset(num_samples_per_class=150, transform=train_transform, seed=42)
    val_dataset = SyntheticVegetationDataset(num_samples_per_class=40, transform=eval_transform, seed=123)
    test_dataset = SyntheticVegetationDataset(num_samples_per_class=40, transform=eval_transform, seed=456)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    model = VisionTransformer().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}, Embed dim: {EMBED_DIM}, Heads: {NUM_HEADS}, Layers: {NUM_LAYERS}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")

    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, DEVICE)

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n" + "=" * 60)
    print("ViT EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Accuracy  : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall   : {recall:.4f}")
    print(f"Macro F1       : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    plot_training_history(history)
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)

    history_data = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "best_val_accuracy": float(best_val_acc),
        "epochs_trained": EPOCHS,
        "model_parameters": total_params,
        "architecture": {
            "patch_size": PATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "mlp_ratio": MLP_RATIO,
        },
        "train_history": {k: [float(v) for v in vals] for k, vals in history.items()},
    }
    with open(HISTORY_PATH, "w") as f:
        json.dump(history_data, f, indent=2)
    print(f"Training history saved: {HISTORY_PATH}")

    print(f"\nModel saved: {MODEL_PATH}")
    print("=" * 60)
    print("ViT PROTOTYPE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
