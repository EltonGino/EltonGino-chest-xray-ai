#!/bin/bash
# ============================================================================
# NIH Chest X-ray14 — Quick Start Script
# ============================================================================
# Usage: ./scripts/run.sh [command]
#
# Commands:
#   eda         — Generate EDA visualizations
#   train       — Train with default config (EfficientNetV2-S)
#   train-fast  — Quick training run (1000 samples, 5 epochs)
#   train-all   — Train all 3 architectures sequentially
#   evaluate    — Evaluate best checkpoint
#   gradcam     — Generate Grad-CAM visualizations
# ============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Auto-detect platform
if [[ "$(uname)" == "Darwin" ]]; then
    echo "🍎 Detected macOS — using MPS backend"
    export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
else
    echo "🖥️  Detected Linux — using CUDA"
fi

case "${1:-help}" in
    eda)
        echo "📊 Generating EDA visualizations..."
        python -m src.eda
        echo "Done! Check outputs/figures/"
        ;;

    train)
        echo "🚀 Training EfficientNetV2-S (default config)..."
        python -m src.train --config configs/config.yaml
        ;;

    train-fast)
        echo "⚡ Quick training run (debug mode)..."
        python -m src.train \
            --config configs/config.yaml \
            --max-samples 1000 \
            --epochs 5 \
            --batch-size 32 \
            --no-freeze
        ;;

    train-all)
        echo "🔬 Training all architectures..."
        for arch in efficientnet_v2_s convnext_tiny vit_small_patch16_224; do
            echo ""
            echo "════════════════════════════════════════"
            echo "  Training: $arch"
            echo "════════════════════════════════════════"
            python -m src.train --config configs/config.yaml --arch "$arch"
        done
        echo "All models trained!"
        ;;

    evaluate)
        CKPT="${2:-checkpoints/best_model_efficientnet_v2_s.pt}"
        echo "📈 Evaluating checkpoint: $CKPT"
        python -m src.evaluate --checkpoint "$CKPT" --save-dir outputs/evaluation
        ;;

    gradcam)
        CKPT="${2:-checkpoints/best_model_efficientnet_v2_s.pt}"
        echo "🔍 Generating Grad-CAM visualizations..."
        python -m src.evaluate --checkpoint "$CKPT" --gradcam --num-gradcam 30
        ;;

    *)
        echo "NIH Chest X-ray14 Classification Pipeline"
        echo ""
        echo "Usage: ./scripts/run.sh [command]"
        echo ""
        echo "Commands:"
        echo "  eda         Generate EDA visualizations"
        echo "  train       Train with default config"
        echo "  train-fast  Quick debug run (1000 samples)"
        echo "  train-all   Train all 3 architectures"
        echo "  evaluate    Evaluate best checkpoint"
        echo "  gradcam     Generate Grad-CAM heatmaps"
        ;;
esac
