# CLAUDE.md — NIH Chest X-Ray Project Context

## Project Overview
Multi-label chest X-ray pathology classifier trained on NIH ChestX-ray14 (112k images, 15 classes).
Full-stack demo: FastAPI backend + React frontend + Grad-CAM + Ollama radiology reports.

**Current best result: 0.793 mean AUC (Ensemble + TTA)**
| Checkpoint | Val AUC | Test AUC | Epoch |
|---|---|---|---|
| `checkpoints/best_model_efficientnet_v2_s.pt` | 0.7859 | 0.7585 | 28 |
| `checkpoints/best_model_convnext_base.pt`     | 0.8199 | 0.770  | 54 |
| **Ensemble + TTA (7 passes)**                 | —      | **0.793** | — |

---

## Completed Roadmap

### Priority 1 — Fix open bugs ✅
- Grad-CAM hook memory leak — GradCAM instantiated once at startup, reused per request
- `useState` used as `useEffect` in `App.jsx` — fixed with correct dependency arrays

### Priority 2 — Experiment tracking (W&B) ✅
- Per-epoch W&B logging: train/val loss, mean AUC, F1, AP, LR, per-class val AUC
- `wandb.summary["best_val_auc"]` set at end of training; graceful fallback if W&B absent

### Priority 3 — Push AUC above 0.80 ✅ (val: 0.820, test: 0.793)
- TTA: `tta_predict()` in `evaluate.py`, 7 passes, no horizontal flip (CXR laterality)
- ConvNeXt-Base trained 60 epochs, patience 10 — val AUC 0.8199 at epoch 54
- Ensemble: `ensemble_predict()` in `evaluate.py`, averages probabilities across both models
- `api.py` updated to load both checkpoints at startup, average predictions at inference

### Priority 4 — Unit tests ✅
- 29 tests in `tests/test_dataset.py` — encode_findings, patient splits, pos_weight
- GitHub Actions CI: `.github/workflows/tests.yml` runs pytest on every push

### Priority 5 — Docker ✅
- `Dockerfile` (Python 3.12-slim, CPU torch, checkpoint mounted as volume)
- `docker-compose.yml` (api + ollama services, OLLAMA_URL wired)
- `.dockerignore` excludes data/, .venv/, checkpoints/, frontend/

### Priority 6 — Model calibration ✅
- ECE per class + mean, temperature scaling, reliability diagrams (15-panel)
- `--calibrate` flag in `evaluate.py`; `calibration.json` loaded by `api.py` at startup

---

## Next: Push AUC Above 0.80 (test)

Ranked by impact-to-effort:

1. **Medical-domain pretraining** — swap ImageNet weights for CheXpert/MIMIC pretrained (HF Hub). Biggest single gain (+0.02–0.04 AUC expected).
2. **320px resolution** — `image_size_large: 320` already in config. Use batch 48, accumulation ×5. Benefits nodule/fibrosis detail.
3. **Add Swin-Transformer** — third ensemble member with different inductive bias (+0.01–0.02 AUC from ConvNet+ViT diversity).
4. **Stochastic Weight Averaging (SWA)** — average last 10–15 epoch checkpoints from ConvNeXt. Free improvement.
5. **Label noise correction** — cleanlab on NIH labels. Removes the ~10% noise ceiling, especially for Pneumonia (0.717).

---

## Known Issues
- `src/nih-chest-xray.code-workspace` is inside `src/` — should be at project root
- `config.yaml` patient split: val+test+train sum > 1 (test value unused but confusing)

## Architecture Notes
- Data paths in `configs/config.yaml` → relative on Mac, absolute on Windows PC
- Training: Mac → SSH → Windows PC (RTX 5070 Ti at `elton-pc`)
- Checkpoint config bakes in data paths at save time — use `--csv-path` etc. to override
- `_filter_df_by_split()` is single source of truth for splits

## Tech Stack
| Layer | Tech |
|---|---|
| Backbones | EfficientNetV2-S + ConvNeXt-Base via timm |
| Loss | AsymmetricLoss (γ⁻=4, γ⁺=1) |
| API | FastAPI + Uvicorn (ensemble inference, temperature calibration) |
| Reports | Ollama (qwen2.5:latest) |
| Frontend | React 18 + Framer Motion + Vite |
| Training hardware | RTX 5070 Ti (17GB VRAM), SSH from Mac Mini |
