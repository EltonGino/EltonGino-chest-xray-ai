# CLAUDE.md — NIH Chest X-Ray Project Context

## Project Overview
Multi-label chest X-ray pathology classifier trained on NIH ChestX-ray14 (112k images, 15 classes).
Full-stack demo: FastAPI backend + React frontend + Grad-CAM + Ollama radiology reports.

**Current best checkpoint:** `checkpoints/best_model_efficientnet_v2_s.pt`
- Architecture: EfficientNetV2-S, 224px, batch 64
- Val AUC: 0.7859 (epoch 28)
- Test AUC: 0.7585 (mean across 15 classes)
- Trained on: RTX 5070 Ti via SSH from Mac Mini

---

## Roadmap to Senior Level

### Priority 1 — Fix open bugs ✅ DONE
- [x] **Grad-CAM hook memory leak** (`api.py`) — GradCAM now instantiated once at startup in `_state`, reused across all requests.
- [x] **`useState` used as `useEffect`** (`frontend/src/App.jsx`) — both `ReportPanel` typewriter and health check converted to `useEffect` with correct dependency arrays.

### Priority 2 — Experiment tracking (W&B)
- [ ] Uncomment and wire up W&B in `configs/config.yaml` (`use_wandb: false` → `true`)
- [ ] Add W&B logging to `src/train.py` — log loss, per-class AUC, LR per epoch
- [ ] Log config as W&B run config for reproducibility

### Priority 3 — Push AUC above 0.80
- [ ] Run ConvNeXt-Small (`convnext_small` already supported in `src/models.py`)
- [ ] Try Test-Time Augmentation (TTA) on the current checkpoint — quick win, no retraining
- [ ] Ensemble EfficientNetV2-S + ConvNeXt-Small predictions

### Priority 4 — Unit tests
- [ ] Test `encode_findings()` — especially the No Finding exclusivity fix
- [ ] Test `patient_hash_split` and `patient_hash_split_binary` — determinism, no overlap between splits
- [ ] Test `_filter_df_by_split` — correct row counts for official vs patient_hash mode
- [ ] Test `compute_pos_weight_from_csv` — output shape, clamping, no division by zero

### Priority 5 — Docker
- [ ] `Dockerfile` for the API (Python + model weights)
- [ ] `docker-compose.yml` — api + ollama services
- [ ] Document in README

### Priority 6 — Model calibration analysis
- [ ] Reliability diagrams per class
- [ ] Expected Calibration Error (ECE) before and after temperature scaling
- [ ] Add temperature scaling post-processing to `evaluate.py`

---

## Known Issues (lower priority)
- `requirements.txt` missing: `fastapi`, `uvicorn`, `opencv-python`, `requests`, `python-multipart`
- `src/nih-chest-xray.code-workspace` is inside `src/` — should be at project root
- `config.yaml` patient split values sum > 1 (val: 0.15 + test: 0.15 + train: 0.82) — test value is unused but confusing
- Hardcoded arch name in frontend empty state (`App.jsx:877`) — should use `modelInfo?.architecture`

---

## Architecture Notes
- Data paths live in `configs/config.yaml` under `data:` — relative paths work from project root on Mac, absolute needed on Windows PC
- Training is cross-machine: code on Mac, trained via SSH on Windows PC (RTX 5070 Ti at `elton-pc`)
- Checkpoint config bakes in the data paths at save time — use `--csv-path` etc. flags in `evaluate.py` to override on a different machine
- `_filter_df_by_split()` is the single source of truth for splits — both dataset and pos_weight use it

## Tech Stack
| Layer | Tech |
|---|---|
| Backbone | EfficientNetV2-S via timm |
| Loss | AsymmetricLoss (γ⁻=4, γ⁺=1) |
| API | FastAPI + Uvicorn |
| Reports | Ollama (qwen2.5:latest) via SSH tunnel |
| Frontend | React 18 + Framer Motion + Vite |
| Training hardware | RTX 5070 Ti (17GB VRAM), SSH from Mac Mini |
