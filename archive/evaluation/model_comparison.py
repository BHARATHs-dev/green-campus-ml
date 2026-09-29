import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "ai/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMPARISON_JSON = os.path.join(OUTPUT_DIR, "model_comparison.json")
COMPARISON_PNG = os.path.join(OUTPUT_DIR, "model_comparison.png")

model_comparison = {
    "project": "AI-Based Green Campus Biomass Monitoring System",
    "date": "2026-08-30",
    "agb_regression": {
        "task": "Above Ground Biomass (AGB) Prediction",
        "unit": "kg",
        "training_samples": 229,
        "total_dataset": 4663,
        "note": "Only 229 records had valid positive diameter, height, and AGB values for reliable training.",
        "models": {
            "KNN": {
                "R2": 0.9859,
                "MAE": None,
                "RMSE": None,
                "CV_R2": None,
                "selected": False
            },
            "Neural_Network": {
                "R2": 0.9914,
                "MAE": None,
                "RMSE": None,
                "CV_R2": None,
                "selected": False
            },
            "Random_Forest": {
                "R2": 0.9989,
                "MAE": 0.0368,
                "RMSE": 0.0712,
                "CV_R2": 0.999,
                "selected": True
            }
        },
        "selected_model": "Random Forest",
        "selection_reason": "Highest R² (0.9989) and lowest error metrics"
    },
    "degradation_classification": {
        "task": "Vegetation Degradation Level Classification",
        "classes": ["Healthy", "Moderately Degraded", "Severely Degraded"],
        "training_samples": 6000,
        "models": {
            "KNN": {
                "Accuracy": 0.9142,
                "Macro_F1": None,
                "selected": False
            },
            "Neural_Network": {
                "Accuracy": 0.9317,
                "Macro_F1": None,
                "selected": False
            },
            "Random_Forest": {
                "Accuracy": 0.9375,
                "Macro_F1": 0.9276,
                "selected": True
            }
        },
        "selected_model": "Random Forest",
        "selection_reason": "Highest accuracy (93.75%) and Macro F1 (0.9276)"
    },
    "image_prototype": {
        "task": "Vegetation Image Classification (Prototype)",
        "dataset": "Synthetic data for pipeline verification",
        "note": "These are prototype results using synthetic image data. Not real-world performance.",
        "models": {
            "CNN": {
                "Accuracy": 1.0,
                "Macro_F1": 1.0,
                "Parameters": 24003,
                "selected": False
            },
            "ViT": {
                "Accuracy": 1.0,
                "Macro_F1": 1.0,
                "Parameters": 563715,
                "selected": False
            }
        },
        "selected_model": "TBD",
        "selection_reason": "Real image collection pending. Final selection will be based on real-world performance."
    },
    "architecture_decision": {
        "production_model": "Random Forest",
        "production_tasks": ["AGB Regression", "Degradation Classification"],
        "image_models": "Prototype stage - CNN and ViT require real campus images for production deployment",
        "future_work": "Collect real vegetation images and retrain CNN/ViT for production image classification"
    }
}

with open(COMPARISON_JSON, "w") as f:
    json.dump(model_comparison, f, indent=2)
print(f"Model comparison JSON saved: {COMPARISON_JSON}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

agb_models = list(model_comparison["agb_regression"]["models"].keys())
agb_r2 = [model_comparison["agb_regression"]["models"][m]["R2"] for m in agb_models]
agb_colors = ["#2ecc71" if model_comparison["agb_regression"]["models"][m]["selected"] else "#95a5a6" for m in agb_models]
bars1 = axes[0].bar(agb_models, agb_r2, color=agb_colors, edgecolor="black", linewidth=0.5)
axes[0].set_title("AGB Regression — R² Score", fontsize=12, fontweight="bold")
axes[0].set_ylabel("R² Score")
axes[0].set_ylim(0.98, 1.0)
for bar, val in zip(bars1, agb_r2):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[0].grid(axis="y", alpha=0.3)

deg_models = list(model_comparison["degradation_classification"]["models"].keys())
deg_acc = [model_comparison["degradation_classification"]["models"][m]["Accuracy"] for m in deg_models]
deg_colors = ["#2ecc71" if model_comparison["degradation_classification"]["models"][m]["selected"] else "#95a5a6" for m in deg_models]
bars2 = axes[1].bar(deg_models, deg_acc, color=deg_colors, edgecolor="black", linewidth=0.5)
axes[1].set_title("Degradation Classification — Accuracy", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0.9, 0.95)
for bar, val in zip(bars2, deg_acc):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{val:.2%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)

img_models = list(model_comparison["image_prototype"]["models"].keys())
img_acc = [model_comparison["image_prototype"]["models"][m]["Accuracy"] for m in img_models]
img_colors = ["#3498db", "#9b59b6"]
bars3 = axes[2].bar(img_models, img_acc, color=img_colors, edgecolor="black", linewidth=0.5)
axes[2].set_title("Image Prototype — Accuracy (Synthetic)", fontsize=12, fontweight="bold")
axes[2].set_ylabel("Accuracy")
axes[2].set_ylim(0.9, 1.05)
for bar, val in zip(bars3, img_acc):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[2].grid(axis="y", alpha=0.3)

fig.suptitle("AI Model Comparison — Green Campus Biomass Monitoring",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(COMPARISON_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"Model comparison chart saved: {COMPARISON_PNG}")

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON SUMMARY")
print("=" * 60)
print("\nAGB Regression:")
for m, v in model_comparison["agb_regression"]["models"].items():
    sel = " ★" if v["selected"] else ""
    print(f"  {m:20s} R² = {v['R2']:.4f}{sel}")
print("\nDegradation Classification:")
for m, v in model_comparison["degradation_classification"]["models"].items():
    sel = " ★" if v["selected"] else ""
    print(f"  {m:20s} Accuracy = {v['Accuracy']:.2%}{sel}")
print("\nImage Prototype (Synthetic):")
for m, v in model_comparison["image_prototype"]["models"].items():
    print(f"  {m:20s} Accuracy = {v['Accuracy']:.0%}  F1 = {v['Macro_F1']:.4f}")
print("\n" + "=" * 60)
print("★ = Selected for production")
print("=" * 60)
