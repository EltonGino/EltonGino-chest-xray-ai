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
from scipy.optimize import minimize_scalar
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
    precomputed_probs: Optional[np.ndarray] = None,
    precomputed_targets: Optional[np.ndarray] = None,
) -> Dict:
    """
    Full evaluation pipeline with all clinically relevant metrics.

    Pass precomputed_probs / precomputed_targets to skip inference
    (used by the TTA path).
    """
    if precomputed_probs is not None and precomputed_targets is not None:
        all_probs   = precomputed_probs
        all_targets = precomputed_targets
    else:
        from tqdm import tqdm
        model.eval()
        all_logits  = []
        all_targets = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
                images = images.to(device)
                logits = model(images)
                all_logits.append(logits.cpu())
                all_targets.append(labels.cpu())

        all_logits  = torch.cat(all_logits).numpy()
        all_targets = torch.cat(all_targets).numpy()
        all_probs   = 1 / (1 + np.exp(-all_logits))  # Sigmoid

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


# ── Test-Time Augmentation ────────────────────────────────────────────────

def tta_predict(
    model: nn.Module,
    dataloader,
    device: torch.device,
    n_augments: int = 6,
    image_size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Test-Time Augmentation: run multiple augmented passes and average logits.

    Augmentations applied per pass (randomly sampled):
      - Horizontal flip
      - Small rotation (±10°)
      - Slight scale crop (85–100%)
    The original unaugmented pass is always included as pass 0.

    Args:
        model:       eval-mode model
        dataloader:  test DataLoader (no augmentation applied by dataset)
        device:      torch device
        n_augments:  number of additional augmented passes (total = n_augments + 1)
        image_size:  expected input resolution

    Returns:
        probs   (N, C) averaged probabilities
        targets (N, C) ground-truth labels
    """
    from torchvision import transforms as T

    # No horizontal flip — CXR has clinically significant laterality
    # (cardiac apex, gastric bubble, effusion side, pneumothorax side).
    aug_transform = T.Compose([
        T.RandomRotation(degrees=5),
        T.RandomResizedCrop(image_size, scale=(0.90, 1.0), ratio=(0.95, 1.05)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Inverse-normalize to get back to [0,1] tensors before re-augmenting
    _inv_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _inv_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    from tqdm import tqdm

    model.eval()
    all_logits_sum = []
    all_targets    = []

    total_passes = n_augments + 1

    # Pass 0 — original (no augmentation)
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"TTA pass 1/{total_passes} (original)", leave=False):
            images = images.to(device)
            logits = model(images).cpu()
            all_logits_sum.append(logits)
            all_targets.append(labels)

    all_logits_sum = [t.clone() for t in all_logits_sum]

    # Passes 1..n_augments — augmented
    for aug_idx in range(n_augments):
        batch_logits = []
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc=f"TTA pass {aug_idx + 2}/{total_passes} (augmented)", leave=False):
                # Undo ImageNet normalisation → [0,1], apply aug, re-normalise
                imgs_01 = (images * _inv_std + _inv_mean).clamp(0, 1)
                imgs_aug = torch.stack([aug_transform(img) for img in imgs_01])
                imgs_aug = imgs_aug.to(device)
                batch_logits.append(model(imgs_aug).cpu())

        for j, bl in enumerate(batch_logits):
            all_logits_sum[j] = all_logits_sum[j] + bl

    total_passes = n_augments + 1
    all_logits_avg = [l / total_passes for l in all_logits_sum]
    all_logits_np  = torch.cat(all_logits_avg).numpy()
    all_targets_np = torch.cat(all_targets).numpy()
    all_probs      = 1 / (1 + np.exp(-all_logits_np))

    return all_probs, all_targets_np


# ── Calibration ───────────────────────────────────────────────────────────

def compute_ece(
    targets: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Expected Calibration Error for each class and overall mean.

    Args:
        targets: (N, C) binary ground-truth array
        probs:   (N, C) predicted probabilities
        n_bins:  number of equal-width bins in [0, 1]

    Returns:
        dict with per-class ECE values and "mean_ece"
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_per_class: Dict[str, float] = {}

    for i, cls in enumerate(CLASS_NAMES):
        p = probs[:, i]
        t = targets[:, i]
        ece = 0.0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (p >= lo) & (p < hi)
            if mask.sum() == 0:
                continue
            avg_conf = p[mask].mean()
            avg_acc  = t[mask].mean()
            ece += (mask.sum() / len(p)) * abs(avg_conf - avg_acc)
        ece_per_class[cls] = round(float(ece), 6)

    ece_per_class["mean_ece"] = round(float(np.mean(list(ece_per_class.values()))), 6)
    return ece_per_class


def find_temperature(
    val_logits: np.ndarray,
    val_targets: np.ndarray,
) -> float:
    """
    Find the optimal scalar temperature T that minimises binary cross-entropy
    on the validation set (temperature scaling / Platt scaling for multi-label).

    Args:
        val_logits: (N, C) raw model logits (before sigmoid)
        val_targets: (N, C) binary ground-truth

    Returns:
        Optimal temperature T  (> 1 means model is over-confident)
    """
    logits_t = torch.tensor(val_logits, dtype=torch.float32)
    targets_t = torch.tensor(val_targets, dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss()

    def nll(T: float) -> float:
        return bce(logits_t / T, targets_t).item()

    result = minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
    return float(result.x)


def plot_reliability_diagrams(
    targets: np.ndarray,
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    temperature: float,
    save_path: str,
    n_bins: int = 10,
) -> None:
    """
    Plot per-class reliability diagrams (before vs after temperature scaling).
    Each subplot: diagonal = perfect calibration, bars = actual accuracy per bin.
    """
    n_cols = 5
    n_rows = -(-NUM_CLASSES // n_cols)  # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for i, cls in enumerate(CLASS_NAMES):
        ax = axes[i]
        for probs, label, color in [
            (probs_before, "Before (T=1.0)", "#4878CF"),
            (probs_after,  f"After  (T={temperature:.2f})", "#E8625A"),
        ]:
            p = probs[:, i]
            t = targets[:, i]
            accs, confs = [], []
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = (p >= lo) & (p < hi)
                if mask.sum() == 0:
                    accs.append(np.nan)
                    confs.append((lo + hi) / 2)
                else:
                    accs.append(t[mask].mean())
                    confs.append(p[mask].mean())
            ax.plot(confs, accs, marker="o", markersize=4, label=label, color=color)

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(cls, fontsize=9, fontweight="bold")
        ax.set_xlabel("Confidence", fontsize=7)
        ax.set_ylabel("Accuracy", fontsize=7)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for j in range(len(CLASS_NAMES), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Reliability Diagrams — Temperature Scaling (T={temperature:.3f})\n"
        "NIH Chest X-ray14",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Reliability diagrams saved to {save_path}")


def run_calibration(
    model: nn.Module,
    val_loader,
    test_loader,
    device: torch.device,
    save_dir: str,
) -> None:
    """
    Full calibration pipeline:
      1. Collect val logits + ground truth
      2. Fit temperature T on val set
      3. Collect test probs before/after scaling
      4. Compute ECE before/after
      5. Plot reliability diagrams
      6. Print summary table
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.eval()

    def _collect(loader):
        all_logits, all_probs, all_targets = [], [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                probs  = torch.sigmoid(logits).cpu()
                all_logits.append(logits.cpu().numpy())
                all_probs.append(probs.numpy())
                all_targets.append(labels.numpy())
        return (
            np.concatenate(all_logits),
            np.concatenate(all_probs),
            np.concatenate(all_targets),
        )

    print("Collecting val set predictions for temperature fitting...")
    val_logits, _, val_targets = _collect(val_loader)

    print("Fitting temperature on validation set...")
    T = find_temperature(val_logits, val_targets)
    print(f"  Optimal temperature: T = {T:.4f}")

    print("Collecting test set predictions...")
    test_logits, test_probs_before, test_targets = _collect(test_loader)
    test_probs_after = torch.sigmoid(torch.tensor(test_logits) / T).numpy()

    ece_before = compute_ece(test_targets, test_probs_before)
    ece_after  = compute_ece(test_targets, test_probs_after)

    # Print summary
    print(f"\n{'Class':<22} {'ECE before':>12} {'ECE after':>12} {'Delta':>10}")
    print("-" * 58)
    for cls in CLASS_NAMES:
        eb = ece_before[cls]
        ea = ece_after[cls]
        print(f"  {cls:<20} {eb:>12.4f} {ea:>12.4f} {ea - eb:>+10.4f}")
    print("-" * 58)
    print(f"  {'Mean':<20} {ece_before['mean_ece']:>12.4f} {ece_after['mean_ece']:>12.4f} "
          f"{ece_after['mean_ece'] - ece_before['mean_ece']:>+10.4f}")

    # Reliability diagrams
    plot_reliability_diagrams(
        test_targets,
        test_probs_before,
        test_probs_after,
        temperature=T,
        save_path=f"{save_dir}/reliability_diagrams.png",
    )

    # Save temperature to file for later use in api.py / inference
    import json
    cal_path = f"{save_dir}/calibration.json"
    with open(cal_path, "w") as f:
        json.dump({"temperature": T, "ece_before": ece_before, "ece_after": ece_after}, f, indent=2)
    print(f"Calibration results saved to {cal_path}")


# ── Ensemble ──────────────────────────────────────────────────────────────

def ensemble_predict(
    checkpoint_paths: list,
    dataloader,
    device: torch.device,
    n_augments: int = 6,
    image_size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load multiple checkpoints, run TTA on each, average probabilities.

    Each model contributes equally (simple average). Simple averaging
    outperforms weighted averaging unless you have a held-out set to
    optimise weights — which would risk overfitting on 15 classes.

    Args:
        checkpoint_paths: list of .pt checkpoint paths
        dataloader:       test DataLoader
        device:           torch device
        n_augments:       TTA augmented passes per model
        image_size:       input resolution

    Returns:
        probs   (N, C) averaged probabilities
        targets (N, C) ground-truth labels (from first model pass)
    """
    all_model_probs = []
    targets = None

    for i, ckpt_path in enumerate(checkpoint_paths):
        print(f"\n[{i+1}/{len(checkpoint_paths)}] Loading {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        arch = ckpt["config"]["model"]["architecture"]

        model = build_model(arch=arch, pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"    arch={arch}, epoch={ckpt['epoch']}, val_auc={ckpt.get('best_auc', 0):.4f}")

        probs, tgts = tta_predict(model, dataloader, device,
                                  n_augments=n_augments, image_size=image_size)
        all_model_probs.append(probs)

        if targets is None:
            targets = tgts

        # Free VRAM between models
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ensemble_probs = np.mean(all_model_probs, axis=0)
    return ensemble_probs, targets


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint path (use --ensemble for multiple)")
    parser.add_argument("--ensemble", type=str, nargs="+", default=None,
                        help="Two or more checkpoint paths to ensemble")
    parser.add_argument("--save-dir", type=str, default="outputs/evaluation")
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--num-gradcam", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for evaluation (default: 64)")
    parser.add_argument("--tta", action="store_true",
                        help="Use Test-Time Augmentation (6 augmented passes + original)")
    parser.add_argument("--tta-passes", type=int, default=6,
                        help="Number of augmented TTA passes (default: 6)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run temperature scaling calibration analysis")
    # Data paths — override checkpoint config (required when running on a different machine)
    parser.add_argument("--csv-path",       type=str, default=None)
    parser.add_argument("--image-dir",      type=str, default=None)
    parser.add_argument("--train-val-list", type=str, default=None)
    parser.add_argument("--test-list",      type=str, default=None)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    if not args.checkpoint and not args.ensemble:
        parser.error("Either --checkpoint or --ensemble is required.")

    # Load primary checkpoint for config/data-path resolution
    primary_ckpt_path = args.checkpoint or args.ensemble[0]
    ckpt = torch.load(primary_ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    # Data paths: CLI args override, then checkpoint config, then relative defaults
    dp = config.get("data", {})
    csv_path       = args.csv_path       or dp.get("csv_path",       "data/NIH_chest_x-ray/Data_Entry_2017.csv")
    image_dir      = args.image_dir      or dp.get("image_dir",      "data/NIH_chest_x-ray/images")
    train_val_list = args.train_val_list or dp.get("train_val_list", "data/NIH_chest_x-ray/train_val_list.txt")
    test_list      = args.test_list      or dp.get("test_list",      "data/NIH_chest_x-ray/test_list.txt")

    loaders = create_dataloaders(
        batch_size=args.batch_size,
        image_size=config["preprocessing"]["image_size"],
        max_samples=None,
        num_workers=4,
        pin_memory=device.type == "cuda",
        use_streaming=False,
        csv_path=csv_path,
        image_dir=image_dir,
        split_mode="official",
        train_val_list=train_val_list,
        test_list=test_list,
        seed=42,
    )

    # ── Ensemble mode ──
    if args.ensemble:
        print(f"\nEnsemble mode: {len(args.ensemble)} models")
        ensemble_probs, ensemble_targets = ensemble_predict(
            checkpoint_paths=args.ensemble,
            dataloader=loaders["test"],
            device=device,
            n_augments=args.tta_passes,
            image_size=config["preprocessing"]["image_size"],
        )
        thresholds = optimize_thresholds(ensemble_targets, ensemble_probs, metric="f1")
        # Load any model for the full_evaluation signature (not used for inference)
        model = build_model(
            arch=config["model"]["architecture"], pretrained=False, freeze_backbone=False,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        results = full_evaluation(
            model, loaders["test"], device,
            thresholds=thresholds,
            save_dir=args.save_dir,
            precomputed_probs=ensemble_probs,
            precomputed_targets=ensemble_targets,
        )

    # ── Single model mode ──
    else:
        model = build_model(
            arch=config["model"]["architecture"],
            pretrained=False,
            freeze_backbone=False,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded {config['model']['architecture']} from epoch {ckpt['epoch']}")

        if args.tta:
            image_size = config["preprocessing"]["image_size"]
            print(f"Running TTA ({args.tta_passes} augmented passes + original)...")
            tta_probs, tta_targets = tta_predict(
                model, loaders["test"], device,
                n_augments=args.tta_passes,
                image_size=image_size,
            )
            thresholds = optimize_thresholds(tta_targets, tta_probs, metric="f1")
            results = full_evaluation(
                model, loaders["test"], device,
                thresholds=thresholds,
                save_dir=args.save_dir,
                precomputed_probs=tta_probs,
                precomputed_targets=tta_targets,
            )
        else:
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

    # Calibration analysis
    if args.calibrate:
        if "val" not in loaders:
            print("[WARN] Val loader not available — skipping calibration.")
        else:
            run_calibration(
                model,
                val_loader=loaders["val"],
                test_loader=loaders["test"],
                device=device,
                save_dir=f"{args.save_dir}/calibration",
            )


if __name__ == "__main__":
    main()
