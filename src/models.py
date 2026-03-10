"""
Model Architectures for NIH Chest X-ray14
==========================================
Multi-label classification heads on modern backbones.
All models use timm for pretrained weights and flexible configuration.

Author: Elton Gino Santos
"""

from typing import Optional

import torch
import torch.nn as nn
import timm
from timm.models import create_model


# ── Base Multi-Label Classifier ───────────────────────────────────────────

class ChestXrayClassifier(nn.Module):
    """
    Unified wrapper for multi-label chest X-ray classification.
    
    Supports:
        - EfficientNetV2 (S, M, L)
        - ConvNeXt (Tiny, Small, Base)
        - Vision Transformer (ViT-S/16, ViT-B/16, ViT-B/32)
    
    Architecture:
        backbone → global_pool → dropout → projection → classifier
    
    Args:
        arch: timm model name (e.g., "efficientnet_v2_s", "convnext_tiny")
        num_classes: number of output classes (15 for NIH CXR14)
        pretrained: use ImageNet pretrained weights
        drop_rate: classifier dropout rate
        drop_path_rate: stochastic depth rate (for training regularization)
        freeze_backbone: freeze backbone parameters initially
    """

    SUPPORTED_ARCHS = {
        # EfficientNetV2
        "efficientnet_v2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
        "efficientnet_v2_m": "tf_efficientnetv2_m.in21k_ft_in1k",
        "efficientnet_v2_l": "tf_efficientnetv2_l.in21k_ft_in1k",
        # ConvNeXt
        "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
        "convnext_small": "convnext_small.fb_in22k_ft_in1k",
        "convnext_base": "convnext_base.fb_in22k_ft_in1k",
        # Vision Transformer
        "vit_small_patch16_224": "vit_small_patch16_224.augreg_in21k_ft_in1k",
        "vit_base_patch16_224": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "vit_base_patch16_384": "vit_base_patch16_384.augreg_in21k_ft_in1k",
    }

    def __init__(
        self,
        arch: str = "efficientnet_v2_s",
        num_classes: int = 15,
        pretrained: bool = True,
        drop_rate: float = 0.3,
        drop_path_rate: float = 0.2,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes

        # Resolve timm model name
        timm_name = self.SUPPORTED_ARCHS.get(arch, arch)

        # Create backbone (remove original classifier head)
        self.backbone = create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier, get features only
            drop_rate=0.0,  # We handle dropout ourselves
            drop_path_rate=drop_path_rate,
        )

        # Get feature dimension from backbone
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat_dim = self.backbone(dummy).shape[-1]

        # Custom classification head
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p=drop_rate),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(p=drop_rate * 0.5),
            nn.Linear(512, num_classes),
        )

        # Initialize head weights
        self._init_head()

        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_backbone()

        # Print model summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model: {arch}")
        print(f"  Feature dim: {feat_dim}")
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")

    def _init_head(self):
        """Xavier initialization for classifier head."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        """Freeze all backbone parameters (transfer learning phase)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  Backbone FROZEN — only head is trainable")

    def unfreeze_backbone(self, lr_scale: float = 0.01):
        """
        Unfreeze backbone for fine-tuning.
        Returns parameter groups with differential learning rates.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("  Backbone UNFROZEN — full fine-tuning enabled")

    def get_param_groups(self, head_lr: float, backbone_lr: float):
        """
        Differential learning rates: lower LR for backbone, higher for head.
        """
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.head.parameters())

        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
        Returns:
            logits: Raw logits [B, num_classes] — apply sigmoid for probabilities
        """
        features = self.backbone(x)  # [B, feat_dim]
        logits = self.head(features)  # [B, num_classes]
        return logits

    def predict(self, x: torch.Tensor, thresholds: Optional[torch.Tensor] = None):
        """
        Predict with optional per-class thresholds.
        
        Returns:
            probs: Sigmoid probabilities [B, num_classes]
            preds: Binary predictions [B, num_classes]
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)

            if thresholds is None:
                thresholds = torch.full((self.num_classes,), 0.5, device=x.device)

            preds = (probs >= thresholds.unsqueeze(0)).float()

        return probs, preds


# ── Loss Functions ────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    Addresses class imbalance by down-weighting easy examples.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=self.pos_weight,
        )

        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * ce_loss
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) — particularly effective for multi-label CXR.
    Decouples focusing for positive and negative samples.
    
    Reference: "Asymmetric Loss For Multi-Label Classification" (Ben-Baruch et al.)
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Asymmetric clipping (hard thresholding for negatives)
        probs_neg = (probs - self.clip).clamp(min=0)

        # Loss calculation
        loss_pos = targets * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8))

        # Asymmetric focusing
        if self.gamma_pos > 0:
            loss_pos *= (1 - probs) ** self.gamma_pos
        if self.gamma_neg > 0:
            loss_neg *= probs_neg ** self.gamma_neg

        loss = -(loss_pos + loss_neg)
        return loss.mean()


# ── Loss Factory ──────────────────────────────────────────────────────────

def get_loss_function(
    loss_name: str = "focal",
    pos_weight: Optional[torch.Tensor] = None,
    **kwargs,
) -> nn.Module:
    """Create loss function by name."""
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_name == "focal":
        return FocalLoss(
            gamma=kwargs.get("focal_gamma", 2.0),
            alpha=kwargs.get("focal_alpha", 0.25),
            pos_weight=pos_weight,
        )
    elif loss_name == "asymmetric":
        return AsymmetricLoss(
            gamma_neg=kwargs.get("gamma_neg", 4.0),
            gamma_pos=kwargs.get("gamma_pos", 1.0),
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}. Use 'bce', 'focal', or 'asymmetric'.")


# ── Model Factory ─────────────────────────────────────────────────────────

def build_model(
    arch: str = "efficientnet_v2_s",
    num_classes: int = 15,
    pretrained: bool = True,
    drop_rate: float = 0.3,
    drop_path_rate: float = 0.2,
    freeze_backbone: bool = True,
) -> ChestXrayClassifier:
    """Build model from config."""
    return ChestXrayClassifier(
        arch=arch,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        freeze_backbone=freeze_backbone,
    )


if __name__ == "__main__":
    # Quick architecture comparison
    for arch in ["efficientnet_v2_s", "convnext_tiny", "vit_small_patch16_224"]:
        print(f"\n{'='*60}")
        model = build_model(arch=arch, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        print(f"  Output shape: {out.shape}")
        print(f"  Memory: {torch.cuda.memory_allocated() / 1e6:.1f} MB" if torch.cuda.is_available() else "")
