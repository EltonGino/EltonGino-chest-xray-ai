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

### Priority 2 — Experiment tracking (W&B) ✅ DONE
- [x] Enabled W&B in `configs/config.yaml`
- [x] Added W&B init with full config logged as run config
- [x] Per-epoch logging: train/val loss, mean AUC, F1, AP, LR, per-class val AUC
- [x] `wandb.summary["best_val_auc"]` set at end of training
- [x] Graceful fallback if wandb not installed (`WANDB_AVAILABLE` flag)

### Priority 3 — Push AUC above 0.80
- [x] TTA implemented — `tta_predict()` in `evaluate.py`, 6 augmented passes + original, `--tta` flag
- [ ] Run ConvNeXt-Small (`convnext_small` already supported in `src/models.py`) — requires retraining
- [ ] Ensemble EfficientNetV2-S + ConvNeXt-Small predictions — requires ConvNeXt checkpoint first

### Priority 4 — Unit tests ✅ DONE
- [x] 29 tests in `tests/test_dataset.py`, all passing
- [x] `encode_findings` — encoding, No Finding exclusivity, unknown labels, all 15 classes
- [x] `patient_hash_split` — determinism, valid outputs, seed sensitivity, distribution
- [x] `patient_hash_split_binary` — determinism, no test split, full coverage, no overlap, distribution
- [x] `_filter_df_by_split` — no patient overlap, full coverage, determinism, invalid input errors
- [x] `compute_pos_weight_from_csv` — shape, dtype, all positive, clamped at 50, no NaN/Inf
- Run with: `pytest tests/ -v`

### Priority 5 — Docker ✅ DONE
- [x] `Dockerfile` — Python 3.12-slim, CPU torch, checkpoint mounted as volume
- [x] `docker-compose.yml` — api + ollama services, OLLAMA_URL wired between containers
- [x] `.dockerignore` — excludes data/, .venv/, checkpoints/, frontend/
- [x] `OLLAMA_URL` in `api.py` now reads from env var (defaults to localhost for local dev)
- Run with: `docker compose up --build`

### Priority 6 — Model calibration analysis ✅ DONE
- [x] `compute_ece()` — ECE per class and mean, equal-width bins
- [x] `find_temperature()` — scalar temperature via minimize_scalar on val BCE
- [x] `plot_reliability_diagrams()` — 15-panel grid, before vs after temp scaling
- [x] `run_calibration()` — full pipeline, saves `calibration.json` (T + ECE scores)
- [x] `--calibrate` flag wired into `main()` in `evaluate.py`
- Run with: `python -m src.evaluate --checkpoint ... --save-dir ... --calibrate`

---

## Known Issues (lower priority)
- `src/nih-chest-xray.code-workspace` is inside `src/` — should be at project root
- `config.yaml` patient split values sum > 1 (val: 0.15 + test: 0.15 + train: 0.82) — test value is unused but confusing
- Hardcoded arch name in frontend empty state (`App.jsx:877`) — should use `modelInfo?.architecture`

## Recent Fixes
- `api.py`: added missing `import os`, `import json`, `import pathlib.Path`; temperature now loaded from `calibration.json` at startup and applied as `logits / T` at inference
- `requirements.txt`: added `fastapi`, `uvicorn[standard]`, `opencv-python`, `requests`, `python-multipart`, `scipy`
- `.github/workflows/tests.yml`: GitHub Actions CI — runs `pytest tests/ -v` on every push/PR to main
- `evaluate.py`: `tta_predict()` added; `full_evaluation()` accepts `precomputed_probs/targets` to skip re-inference; `--tta` / `--tta-passes` flags wired into `main()`

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
