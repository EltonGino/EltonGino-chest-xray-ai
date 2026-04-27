# -*- coding: utf-8 -*-
"""
Training Pipeline for NIH Chest X-ray14
=========================================
Features: AMP, gradient accumulation, cosine warmup scheduler,
differential LR, backbone unfreezing, early stopping, checkpointing.

Author: Elton Gino Santos

Usage:
    python -m src.train --config configs/config.yaml
    python -m src.train --arch convnext_tiny --epochs 30 --batch-size 64
"""

import os
import sys
import time

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1)
import argparse
import platform
from pathlib import Path
from typing import Optional, Dict

import yaml
import numpy as np
import torch
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from contextlib import nullcontext
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn as swa_update_bn
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from src.dataset import compute_pos_weight_from_csv

from src.dataset import (
    create_dataloaders,
    CLASS_NAMES, NUM_CLASSES,
)
from src.models import build_model, get_loss_function


# ── Device Detection ──────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# ── Cosine Schedule with Warmup ───────────────────────────────────────────

def get_scheduler(optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-7):
    """Cosine annealing with linear warmup."""
    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=min_lr,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


# ── MixUp ─────────────────────────────────────────────────────────────────

def mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.4):
    """Blend two random samples. Soft targets make ASL more robust to label noise."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0), device=images.device)
    return lam * images + (1 - lam) * images[idx], lam * labels + (1 - lam) * labels[idx]


# ── Training Step ─────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer,
    scaler,  # torch.amp.GradScaler or None
    device: torch.device,
    accum_steps: int = 1,
    use_amp: bool = True,
    mixup_alpha: float = 0.0,
) -> Dict[str, float]:
    """Single training epoch with optional AMP, gradient accumulation, and MixUp."""
    model.train()
    running_loss = 0.0
    all_logits = []
    all_targets = []

    from tqdm import tqdm

    optimizer.zero_grad()

    for step, (images, labels) in enumerate(tqdm(dataloader, desc="  train", leave=False)):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        orig_labels = labels  # keep binary labels for metric computation
        if mixup_alpha > 0:
            images, labels = mixup_batch(images, labels, mixup_alpha)

        # AMP only on CUDA; MPS/CPU use normal precision context
        amp_ctx = torch.amp.autocast("cuda", enabled=use_amp) if (use_amp and device.type == "cuda") else nullcontext()
        with amp_ctx:
            logits = model(images)
            loss = criterion(logits, labels) / accum_steps

        # Backward
        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
            if scaler is not None and use_amp and device.type == "cuda":
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        # Track metrics (use original binary labels, not mixed)
        running_loss += loss.item() * accum_steps
        all_logits.append(logits.detach().cpu())
        all_targets.append(orig_labels.detach().cpu())

        # Progress
        if (step + 1) % 100 == 0:
            print(f"    Step {step+1}/{len(dataloader)} — Loss: {loss.item() * accum_steps:.4f}")

    # Epoch metrics
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    probs = torch.sigmoid(all_logits).numpy()
    targets_np = all_targets.numpy()

    epoch_loss = running_loss / len(dataloader)

    # AUC-ROC per class (skip classes with no positive samples)
    aucs = []
    for i in range(NUM_CLASSES):
        if targets_np[:, i].sum() > 0 and targets_np[:, i].sum() < len(targets_np):
            aucs.append(roc_auc_score(targets_np[:, i], probs[:, i]))
        else:
            aucs.append(0.0)

    return {
        "loss": epoch_loss,
        "auc_mean": np.mean(aucs),
        "aucs": aucs,
    }


# ── Validation Step ───────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> Dict:
    """Validation with comprehensive metrics."""
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_targets = []

    from tqdm import tqdm

    for images, labels in tqdm(dataloader, desc="  val  ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        amp_ctx = torch.amp.autocast("cuda", enabled=use_amp) if (use_amp and device.type == "cuda") else nullcontext()
        with amp_ctx:
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item()
        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    probs = torch.sigmoid(all_logits).numpy()
    targets_np = all_targets.numpy()
    preds_np = (probs > 0.5).astype(float)

    epoch_loss = running_loss / len(dataloader)

    # Per-class metrics
    aucs, aps, f1s = [], [], []
    for i in range(NUM_CLASSES):
        if targets_np[:, i].sum() > 0 and targets_np[:, i].sum() < len(targets_np):
            aucs.append(roc_auc_score(targets_np[:, i], probs[:, i]))
            aps.append(average_precision_score(targets_np[:, i], probs[:, i]))
            f1s.append(f1_score(targets_np[:, i], preds_np[:, i], zero_division=0))
        else:
            aucs.append(0.0)
            aps.append(0.0)
            f1s.append(0.0)

    return {
        "loss": epoch_loss,
        "auc_mean": np.mean(aucs),
        "ap_mean": np.mean(aps),
        "f1_mean": np.mean(f1s),
        "aucs": aucs,
        "aps": aps,
        "f1s": f1s,
        "logits": all_logits,
        "targets": all_targets,
    }


# ── Early Stopping ────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 7, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def __call__(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            return False

        improved = (
            metric > self.best + self.min_delta
            if self.mode == "max"
            else metric < self.best - self.min_delta
        )

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


# ── Main Training Loop ────────────────────────────────────────────────────

def train(config: dict):
    """Main training orchestrator."""
    
    # Setup
    device = get_device()
    is_mac = platform.system() == "Darwin"

    # Resolve hardware-specific settings
    batch_size = config["training"].get(
        "batch_size_mac" if is_mac else "batch_size", 64
    )
    num_workers = config["training"].get(
        "num_workers_mac" if is_mac else "num_workers", 8
    )
    use_amp = config["training"]["amp"] and device.type == "cuda"

    print(f"\n{'='*60}")
    print(f"NIH Chest X-ray14 — Training Pipeline")
    print(f"{'='*60}")
    print(f"Architecture: {config['model']['architecture']}")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {num_workers}")
    print(f"Mixed precision: {use_amp}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # W&B init
    use_wandb = config["logging"].get("use_wandb", False) and WANDB_AVAILABLE
    if use_wandb:
        img_size = config["preprocessing"]["image_size"]
        wandb.init(
            project=config["logging"]["project_name"],
            config=config,
            name=f"{config['model']['architecture']}_{img_size}px",
        )
        print("W&B run initialized.")

    # ── Data ──
    dp = config["data"]
    loaders = create_dataloaders(
        batch_size=batch_size,
        image_size=config["preprocessing"]["image_size"],
        max_samples=config["dataset"].get("max_samples"),
        num_workers=num_workers,
        pin_memory=config["training"]["pin_memory"] and device.type == "cuda",

        use_streaming=False,
        csv_path=dp["csv_path"],
        image_dir=dp["image_dir"],
        split_mode="official",
        train_val_list=dp["train_val_list"],
        test_list=dp["test_list"],
        seed=42,
    )

    # ── Model ──
    model = build_model(
        arch=config["model"]["architecture"],
        num_classes=config["dataset"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        drop_rate=config["model"]["drop_rate"],
        drop_path_rate=config["model"]["drop_path_rate"],
        freeze_backbone=config["model"]["freeze_backbone"],
    ).to(device)

    # ── Fine-tune from existing checkpoint (backbone weights only) ──
    finetune_from = config.get("training", {}).get("finetune_from")
    if finetune_from and Path(finetune_from).exists():
        print(f"Loading backbone weights from: {finetune_from}")
        ft_ckpt = torch.load(finetune_from, map_location=device, weights_only=False)
        src_sd = ft_ckpt["model_state_dict"]
        backbone_sd = {k[len("backbone."):]: v for k, v in src_sd.items() if k.startswith("backbone.")}
        missing, unexpected = model.backbone.load_state_dict(backbone_sd, strict=False)
        loaded = len(backbone_sd) - len(missing)
        print(f"  Loaded {loaded}/{len(backbone_sd)} backbone params"
              + (f" ({len(missing)} missing — expected for arch changes)" if missing else ""))
    elif finetune_from:
        print(f"[WARN] finetune_from path not found: {finetune_from}")

    # ── Loss ──
    pos_weight = compute_pos_weight_from_csv(
        csv_path=dp["csv_path"],
        split_mode="official",
        split="train",
        train_val_list=dp["train_val_list"],
        test_list=dp["test_list"],
        seed=42,
    ).to(device)
    criterion = get_loss_function(
        config["training"]["loss"],
        pos_weight=pos_weight,
        focal_gamma=config["training"].get("focal_gamma", 2.0),
        focal_alpha=config["training"].get("focal_alpha", 0.25),
    )

    # ── Optimizer ──
    if config["model"]["freeze_backbone"]:
        # Only train head initially
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            betas=tuple(config["training"]["betas"]),
        )
    else:
        optimizer = AdamW(
            model.get_param_groups(
                head_lr=config["training"]["learning_rate"],
                backbone_lr=config["training"]["backbone_lr"],
            ),
            weight_decay=config["training"]["weight_decay"],
            betas=tuple(config["training"]["betas"]),
        )

    # ── Scheduler ──
    scheduler = get_scheduler(
        optimizer,
        warmup_epochs=config["training"]["warmup_epochs"],
        total_epochs=config["training"]["epochs"],
        min_lr=config["training"]["min_lr"],
    )

    # ── AMP Scaler ──
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda")) if device.type == "cuda" else None

    # ── SWA Setup ──
    swa_cfg = config["training"].get("swa", {})
    swa_enabled = swa_cfg.get("enabled", False)
    swa_start   = swa_cfg.get("start_epoch", max(1, int(config["training"]["epochs"] * 0.75)))
    swa_model   = AveragedModel(model) if swa_enabled else None
    swa_sched   = SWALR(optimizer, swa_lr=swa_cfg.get("lr", 5e-6), anneal_epochs=5) if swa_enabled else None
    if swa_enabled:
        print(f"SWA enabled — averaging from epoch {swa_start}")

    # ── MixUp ──
    mixup_alpha = config["training"].get("mixup_alpha", 0.0)
    if mixup_alpha > 0:
        print(f"MixUp enabled — alpha={mixup_alpha}")

    # ── Training ──
    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping"]["patience"],
        mode=config["training"]["early_stopping"]["mode"],
    )

    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_auc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": [], "lr": []}

    for epoch in range(1, config["training"]["epochs"] + 1):
        t0 = time.time()

        # ── Unfreeze backbone after N epochs ──
        if (
            config["model"]["freeze_backbone"]
            and epoch == config["model"]["freeze_epochs"] + 1
        ):
            print(f"\n{'='*40}")
            print(f"UNFREEZING BACKBONE at epoch {epoch}")
            print(f"{'='*40}")
            model.unfreeze_backbone()

            # Rebuild optimizer with differential LR
            optimizer = AdamW(
                model.get_param_groups(
                    head_lr=config["training"]["learning_rate"] * 0.1,
                    backbone_lr=config["training"]["backbone_lr"],
                ),
                weight_decay=config["training"]["weight_decay"],
            )
            scheduler = get_scheduler(
                optimizer,
                warmup_epochs=2,
                total_epochs=config["training"]["epochs"] - epoch + 1,
                min_lr=config["training"]["min_lr"],
            )

        # ── Train ──
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        print("  LRs:", ", ".join([f"{lr:.2e}" for lr in lrs]))

        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler,
            device, config["training"]["gradient_accumulation_steps"], use_amp,
            mixup_alpha=mixup_alpha,
        )

        # ── Validate ──
        val_metrics = validate(model, loaders["val"], criterion, device, use_amp)

        # ── SWA update or normal scheduler step ──
        if swa_enabled and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
        else:
            scheduler.step()

        elapsed = time.time() - t0

        # ── Logging ──
        print(f"  Train Loss: {train_metrics['loss']:.4f} | AUC: {train_metrics['auc_mean']:.4f}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f} | AUC: {val_metrics['auc_mean']:.4f} | "
              f"F1: {val_metrics['f1_mean']:.4f} | AP: {val_metrics['ap_mean']:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        # Per-class AUC
        print(f"  Per-class AUC:")
        for name, auc in zip(CLASS_NAMES, val_metrics["aucs"]):
            print(f"    {name:>20s}: {auc:.4f}")

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc_mean"])
        history["val_f1"].append(val_metrics["f1_mean"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if use_wandb:
            log = {
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/auc_mean": train_metrics["auc_mean"],
                "val/loss": val_metrics["loss"],
                "val/auc_mean": val_metrics["auc_mean"],
                "val/f1_mean": val_metrics["f1_mean"],
                "val/ap_mean": val_metrics["ap_mean"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            # Per-class val AUC
            for name, auc in zip(CLASS_NAMES, val_metrics["aucs"]):
                log[f"val/auc_{name.lower().replace(' ', '_')}"] = auc
            wandb.log(log)

        # ── Checkpointing ──
        if val_metrics["auc_mean"] > best_auc:
            best_auc = val_metrics["auc_mean"]
            ckpt_path = checkpoint_dir / f"best_model_{config['model']['architecture']}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "val_metrics": val_metrics,
                "config": config,
                "history": history,
            }, ckpt_path)
            print(f"  >> New best AUC: {best_auc:.4f} -- saved to {ckpt_path}")

        # ── Early Stopping ──
        if early_stopping(val_metrics["auc_mean"]):
            print(f"\n  Early stopping triggered after {epoch} epochs")
            break

    # ── SWA Finalization ──
    if swa_enabled and swa_model is not None:
        print("\nUpdating BatchNorm statistics for SWA model...")
        swa_update_bn(loaders["train"], swa_model, device=device)
        swa_val = validate(swa_model, loaders["val"], criterion, device, use_amp)
        print(f"SWA Val AUC: {swa_val['auc_mean']:.4f}  (regular best: {best_auc:.4f})")
        swa_path = checkpoint_dir / f"best_model_{config['model']['architecture']}_swa.pt"
        # Strip the AveragedModel wrapper so the checkpoint loads like a normal one
        inner_sd = {k[len("module."):]: v
                    for k, v in swa_model.state_dict().items()
                    if k.startswith("module.")}
        torch.save({
            "epoch": f"swa_{swa_start}-{epoch}",
            "model_state_dict": inner_sd,
            "best_auc": swa_val["auc_mean"],
            "val_metrics": swa_val,
            "config": config,
            "history": history,
        }, swa_path)
        print(f"SWA checkpoint saved → {swa_path}")
        if swa_val["auc_mean"] > best_auc:
            best_auc = swa_val["auc_mean"]
        if use_wandb:
            wandb.summary["swa_val_auc"] = swa_val["auc_mean"]

    print(f"\n{'='*60}")
    print(f"Training complete! Best AUC: {best_auc:.4f}")
    print(f"{'='*60}")

    if use_wandb:
        wandb.summary["best_val_auc"] = best_auc
        wandb.finish()

    return model, history


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train NIH CXR14 Classifier")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--arch", type=str, default=None, help="Override model architecture")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size for debugging")
    parser.add_argument("--no-freeze", action="store_true", help="Don't freeze backbone")
    parser.add_argument("--finetune-from", type=str, default=None,
                        help="Load backbone weights from checkpoint before training")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.arch:
        config["model"]["architecture"] = args.arch
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
        config["training"]["batch_size_mac"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.max_samples:
        config["dataset"]["max_samples"] = args.max_samples
    if args.no_freeze:
        config["model"]["freeze_backbone"] = False
    if args.finetune_from:
        config["training"]["finetune_from"] = args.finetune_from

    train(config)


if __name__ == "__main__":
    main()
