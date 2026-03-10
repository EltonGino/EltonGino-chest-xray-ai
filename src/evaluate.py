"""
Evaluation & Inference for NIH Chest X-ray14
==============================================
Per-class threshold optimization, Grad-CAM visualization,
comprehensive metrics reporting, and model comparison.

Author: Elton Gino Santos

Usage:
    python -m src.evaluate --checkpoint checkpoints/best_model_efficientnet_v2_s.pt
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report,
)

from src.dataset import CLASS_NAMES, NUM_CLASSES, create_dataloaders
from src.models import build_model


# ── Per-Class Threshold Optimization ──────────────────────────────────────

def optimize_thresholds(
    targets: np.ndarray,
    probs: np.ndarray,
    metric: str = "f1",
    search_range: tuple = (0.1, 0.9),
    num_steps: int = 100,
) -> np.ndarray:
    """
    Find optimal classification threshold per class.
    
    Instead of using 0.5 for all classes, we search for the threshold
    that maximizes F1 (or another metric) independently per class.
    This is critical for imbalanced datasets like CXR14.
    """
    thresholds = np.zeros(NUM_CLASSES)
    candidates = np.linspace(search_range[0], search_range[1], num_steps)

    for i in range(NUM_CLASSES):
        best_score = 0.0
        best_thresh = 0.5

        for thresh in candidates:
            preds = (probs[:, i] >= thresh).astype(float)

            if metric == "f1":
                score = f1_score(targets[:, i], preds, zero_division=0)
            elif metric == "youden":
                # Youden's J = sensitivity + specificity - 1
                tp = ((preds == 1) & (targets[:, i] == 1)).sum()
                tn = ((preds == 0) & (targets[:, i] == 0)).sum()
                fp = ((preds == 1) & (targets[:, i] == 0)).sum()
                fn = ((preds == 0) & (targets[:, i] == 1)).sum()
                sens = tp / (tp + fn + 1e-8)
                spec = tn / (tn + fp + 1e-8)
                score = sens + spec - 1
            else:
                score = f1_score(targets[:, i], preds, zero_division=0)

            if score > best_score:
                best_score = score
                best_thresh = thresh

        thresholds[i] = best_thresh

    return thresholds


# ── Comprehensive Evaluation ──────────────────────────────────────────────

def full_evaluation(
    model: nn.Module,
    dataloader,
    device: torch.device,
    thresholds: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Full evaluation pipeline with all clinically relevant metrics.
    """
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_probs = 1 / (1 + np.exp(-all_logits))  # Sigmoid

    # Optimize thresholds if not provided
    if thresholds is None:
        print("Optimizing per-class thresholds...")
        thresholds = optimize_thresholds(all_targets, all_probs, metric="f1")

    all_preds = (all_probs >= thresholds[np.newaxis, :]).astype(float)

    # Per-class metrics
    results = {"per_class": {}, "thresholds": thresholds}

    print(f"\n{'='*80}")
    print(f"{'Class':>22s} | {'AUC-ROC':>8s} | {'AUC-PR':>8s} | {'F1':>8s} | "
          f"{'Prec':>8s} | {'Recall':>8s} | {'Thresh':>8s} | {'Prev':>8s}")
    print(f"{'-'*80}")

    aucs, aps, f1s = [], [], []

    for i, name in enumerate(CLASS_NAMES):
        has_both = all_targets[:, i].sum() > 0 and all_targets[:, i].sum() < len(all_targets)

        if has_both:
            auc = roc_auc_score(all_targets[:, i], all_probs[:, i])
            ap = average_precision_score(all_targets[:, i], all_probs[:, i])
        else:
            auc = ap = 0.0

        f1 = f1_score(all_targets[:, i], all_preds[:, i], zero_division=0)
        prec = precision_score(all_targets[:, i], all_preds[:, i], zero_division=0)
        rec = recall_score(all_targets[:, i], all_preds[:, i], zero_division=0)
        prev = all_targets[:, i].mean()

        results["per_class"][name] = {
            "auc_roc": auc, "auc_pr": ap, "f1": f1,
            "precision": prec, "recall": rec,
            "threshold": thresholds[i], "prevalence": prev,
        }

        aucs.append(auc)
        aps.append(ap)
        f1s.append(f1)

        print(f"{name:>22s} | {auc:>8.4f} | {ap:>8.4f} | {f1:>8.4f} | "
              f"{prec:>8.4f} | {rec:>8.4f} | {thresholds[i]:>8.3f} | {prev:>8.4f}")

    print(f"{'-'*80}")
    print(f"{'MEAN':>22s} | {np.mean(aucs):>8.4f} | {np.mean(aps):>8.4f} | "
          f"{np.mean(f1s):>8.4f} |")
    print(f"{'='*80}\n")

    results["mean_auc"] = np.mean(aucs)
    results["mean_ap"] = np.mean(aps)
    results["mean_f1"] = np.mean(f1s)
    results["probs"] = all_probs
    results["targets"] = all_targets
    results["preds"] = all_preds

    # Generate plots
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_roc_curves(all_targets, all_probs, save_dir / "roc_curves.png")
        plot_pr_curves(all_targets, all_probs, save_dir / "pr_curves.png")
        plot_confusion_matrices(all_targets, all_preds, save_dir / "confusion_matrices.png")
        plot_threshold_analysis(all_targets, all_probs, thresholds, save_dir / "threshold_analysis.png")

    return results


# ── Visualization Functions ───────────────────────────────────────────────

def plot_roc_curves(targets: np.ndarray, probs: np.ndarray, save_path: str):
    """Plot ROC curves for all 14 disease classes (excluding No Finding)."""
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    axes = axes.ravel()

    for i in range(NUM_CLASSES):
        ax = axes[i]
        if targets[:, i].sum() > 0 and targets[:, i].sum() < len(targets):
            fpr, tpr, _ = roc_curve(targets[:, i], probs[:, i])
            auc = roc_auc_score(targets[:, i], probs[:, i])
            ax.plot(fpr, tpr, 'b-', linewidth=2, label=f"AUC = {auc:.3f}")
            ax.fill_between(fpr, tpr, alpha=0.15, color='blue')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_title(CLASS_NAMES[i], fontsize=11, fontweight="bold")
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.suptitle("ROC Curves — Per-Class Performance", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pr_curves(targets: np.ndarray, probs: np.ndarray, save_path: str):
    """Plot Precision-Recall curves (more informative for imbalanced data)."""
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    axes = axes.ravel()

    for i in range(NUM_CLASSES):
        ax = axes[i]
        if targets[:, i].sum() > 0:
            prec, rec, _ = precision_recall_curve(targets[:, i], probs[:, i])
            ap = average_precision_score(targets[:, i], probs[:, i])
            ax.plot(rec, prec, 'r-', linewidth=2, label=f"AP = {ap:.3f}")
            ax.fill_between(rec, prec, alpha=0.15, color='red')
            # Baseline (prevalence)
            prevalence = targets[:, i].mean()
            ax.axhline(y=prevalence, color='gray', linestyle='--', alpha=0.5, label=f"Prev = {prevalence:.3f}")
        ax.set_title(CLASS_NAMES[i], fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

    plt.suptitle("Precision-Recall Curves — Per-Class Performance", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrices(targets: np.ndarray, preds: np.ndarray, save_path: str):
    """Plot normalized confusion matrices for each class."""
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    axes = axes.ravel()

    for i in range(NUM_CLASSES):
        ax = axes[i]
        cm = confusion_matrix(targets[:, i], preds[:, i], normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", ax=ax,
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
        ax.set_title(CLASS_NAMES[i], fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.suptitle("Normalized Confusion Matrices", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_threshold_analysis(
    targets: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray,
    save_path: str,
):
    """Show how F1 varies with threshold per class."""
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    axes = axes.ravel()

    candidates = np.linspace(0.05, 0.95, 50)

    for i in range(NUM_CLASSES):
        ax = axes[i]
        f1s = []
        for t in candidates:
            preds = (probs[:, i] >= t).astype(float)
            f1s.append(f1_score(targets[:, i], preds, zero_division=0))

        ax.plot(candidates, f1s, 'g-', linewidth=2)
        ax.axvline(x=thresholds[i], color='red', linestyle='--',
                    label=f"Optimal = {thresholds[i]:.2f}")
        ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label="Default 0.5")
        ax.set_title(CLASS_NAMES[i], fontsize=11, fontweight="bold")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1 Score")
        ax.legend(fontsize=8)

    plt.suptitle("F1 Score vs. Classification Threshold", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Grad-CAM ─────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for model interpretability.
    Shows which regions of the CXR the model focuses on for each prediction.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.gradients = None
        self.activations = None

        # Auto-detect last convolutional layer
        if target_layer is None:
            target_layer = self._find_last_conv(model)

        self.target_layer = target_layer

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _find_last_conv(self, model: nn.Module) -> nn.Module:
        """Find the last convolutional layer in the backbone."""
        last_conv = None
        for module in model.backbone.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                last_conv = module
        if last_conv is None:
            raise ValueError("No convolutional layer found in backbone")
        return last_conv

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image: torch.Tensor,
        class_idx: int,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a specific class.
        
        Args:
            image: Single image tensor [1, 3, H, W]
            class_idx: Target class index
        Returns:
            cam: Normalized heatmap [H, W] in range [0, 1]
        """
        self.model.eval()
        output = self.model(image)

        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)

        # Global average pool gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Resize to input size
        cam = nn.functional.interpolate(
            cam, size=image.shape[2:], mode='bilinear', align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def visualize_gradcam(
    model: nn.Module,
    dataset,
    device: torch.device,
    num_samples: int = 10,
    save_dir: str = "outputs/gradcam",
):
    """Generate Grad-CAM visualizations for sample images."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    grad_cam = GradCAM(model)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for idx in range(min(num_samples, len(dataset))):
        image, label = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)

        # Get predictions
        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Top 3 predicted classes
        top_classes = np.argsort(probs)[::-1][:3]
        active_labels = [i for i, v in enumerate(label) if v == 1.0]

        # Denormalize image for display
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * std + mean).clip(0, 1)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Original image
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title(f"Original\nGT: {', '.join(CLASS_NAMES[i] for i in active_labels)}")
        axes[0].axis("off")

        # Grad-CAM for top 3 predictions
        for j, cls_idx in enumerate(top_classes):
            cam = grad_cam.generate(image_tensor, cls_idx)
            axes[j + 1].imshow(img_np, cmap='gray')
            axes[j + 1].imshow(cam, cmap='jet', alpha=0.4)
            axes[j + 1].set_title(f"{CLASS_NAMES[cls_idx]}\nProb: {probs[cls_idx]:.3f}")
            axes[j + 1].axis("off")

        plt.tight_layout()
        plt.savefig(save_dir / f"gradcam_sample_{idx:03d}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Grad-CAM visualizations saved to {save_dir}")


# ── Model Comparison Report ───────────────────────────────────────────────

def compare_models(results_dict: Dict[str, Dict], save_path: str = "outputs/model_comparison.png"):
    """
    Compare multiple model results side by side.
    
    Args:
        results_dict: {model_name: evaluation_results}
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    model_names = list(results_dict.keys())
    x = np.arange(NUM_CLASSES)
    width = 0.8 / len(model_names)

    # AUC comparison
    for j, (name, res) in enumerate(results_dict.items()):
        aucs = [res["per_class"][c]["auc_roc"] for c in CLASS_NAMES]
        axes[0].bar(x + j * width, aucs, width, label=name, alpha=0.8)
    axes[0].set_title("AUC-ROC per Class", fontweight="bold")
    axes[0].set_xticks(x + width * (len(model_names) - 1) / 2)
    axes[0].set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    axes[0].legend()

    # F1 comparison
    for j, (name, res) in enumerate(results_dict.items()):
        f1s = [res["per_class"][c]["f1"] for c in CLASS_NAMES]
        axes[1].bar(x + j * width, f1s, width, label=name, alpha=0.8)
    axes[1].set_title("F1 Score per Class", fontweight="bold")
    axes[1].set_xticks(x + width * (len(model_names) - 1) / 2)
    axes[1].set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    axes[1].legend()

    # Summary
    summary_metrics = ["mean_auc", "mean_ap", "mean_f1"]
    summary_labels = ["Mean AUC", "Mean AP", "Mean F1"]
    x_summary = np.arange(len(summary_metrics))
    for j, (name, res) in enumerate(results_dict.items()):
        vals = [res[m] for m in summary_metrics]
        axes[2].bar(x_summary + j * width, vals, width, label=name, alpha=0.8)
    axes[2].set_title("Summary Metrics", fontweight="bold")
    axes[2].set_xticks(x_summary + width * (len(model_names) - 1) / 2)
    axes[2].set_xticklabels(summary_labels)
    axes[2].legend()
    axes[2].set_ylim([0, 1])

    plt.suptitle("Model Architecture Comparison — NIH Chest X-ray14", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Model comparison saved to {save_path}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="outputs/evaluation")
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--num-gradcam", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = build_model(
        arch=config["model"]["architecture"],
        pretrained=False,
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded {config['model']['architecture']} from epoch {ckpt['epoch']}")

    # Create test dataloader
    loaders = create_dataloaders(
    batch_size=16,  # evaluation can be smaller to be safe
    image_size=config["preprocessing"]["image_size"],
    max_samples=None,
    num_workers=2,
    pin_memory=False,

    # Local mode
    use_streaming=False,
    csv_path="/Volumes/ROG SSD/PyCharm/nih-chest-xray/data/NIH_chest_x-ray/Data_Entry_2017.csv",
    image_dir="/Volumes/ROG SSD/PyCharm/nih-chest-xray/data/NIH_chest_x-ray/images",

    split_mode="official",
    train_val_list="/Volumes/ROG SSD/PyCharm/nih-chest-xray/data/NIH_chest_x-ray/train_val_list.txt",
    test_list="/Volumes/ROG SSD/PyCharm/nih-chest-xray/data/NIH_chest_x-ray/test_list.txt",
    seed=42,
)

    # Full evaluation
    results = full_evaluation(
        model, loaders["test"], device, save_dir=args.save_dir,
    )

    # Grad-CAM
    if args.gradcam:
        visualize_gradcam(
            model, loaders["test"].dataset, device,
            num_samples=args.num_gradcam,
            save_dir=f"{args.save_dir}/gradcam",
        )


if __name__ == "__main__":
    main()
