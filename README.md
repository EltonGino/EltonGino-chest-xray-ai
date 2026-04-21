# Chest X-Ray AI — Multi-Label Pathology Classification

> Ensemble of EfficientNetV2-S + ConvNeXt-Base trained on NIH ChestX-ray14 with a FastAPI + React inference demo, Grad-CAM visualizations, and AI-generated radiology reports via local LLM.

![Demo](outputs/figures/demo.png)
![Demo 1](outputs/figures/demo1.png)

---

## Results

Trained on the full NIH ChestX-ray14 dataset (112,120 images, 30,805 patients) using the official train/test split. Evaluated on the held-out test set of 25,596 images with Test-Time Augmentation (TTA, 7 passes).

| Class | AUC-ROC | AUC-PR | F1 | Prevalence |
|---|---|---|---|---|
| No Finding | 0.733 | 0.662 | 0.610 | 38.5% |
| Atelectasis | 0.767 | 0.343 | 0.387 | 12.8% |
| Cardiomegaly | 0.876 | 0.332 | 0.393 | 4.2% |
| Effusion | 0.830 | 0.518 | 0.535 | 18.2% |
| Infiltration | 0.707 | 0.400 | 0.491 | 23.9% |
| Mass | 0.799 | 0.288 | 0.346 | 6.8% |
| Nodule | 0.744 | 0.222 | 0.281 | 6.3% |
| Pneumonia | 0.717 | 0.050 | 0.093 | 2.2% |
| Pneumothorax | 0.857 | 0.402 | 0.456 | 10.4% |
| Consolidation | 0.739 | 0.146 | 0.238 | 7.1% |
| Edema | 0.841 | 0.153 | 0.230 | 3.6% |
| Emphysema | 0.873 | 0.275 | 0.349 | 4.3% |
| Fibrosis | 0.801 | 0.079 | 0.150 | 1.7% |
| Pleural Thickening | 0.751 | 0.120 | 0.196 | 4.5% |
| Hernia | 0.857 | 0.121 | 0.190 | 0.3% |
| **Mean** | **0.793** | **0.274** | **0.330** | — |

**CheXNet (DenseNet-121) baseline: ~0.745 mean AUC** — Rajpurkar et al., arXiv 2017.

### Improvement Progression

| Model | Test AUC |
|---|---|
| EfficientNetV2-S | 0.759 |
| EfficientNetV2-S + TTA | 0.765 |
| ConvNeXt-Base (60 epochs) + TTA | 0.770 |
| **Ensemble + TTA** | **0.793** |

---

## Architecture

### Model 1 — EfficientNetV2-S

```
Input (224×224 RGB)
       │
       ▼
EfficientNetV2-S backbone         ← ImageNet-21k pretrained (timm)
  tf_efficientnetv2_s.in21k_ft_in1k
  20.2M parameters
       │
       ▼
Global Average Pool → [1280]
       │
       ▼
Classification Head
  LayerNorm(1280) → Dropout(0.3) → Linear(1280→512) → GELU → Dropout(0.15) → Linear(512→15)
       │
       ▼
15-class multi-hot output (sigmoid)
```

### Model 2 — ConvNeXt-Base

```
Input (224×224 RGB)
       │
       ▼
ConvNeXt-Base backbone            ← ImageNet-22k pretrained (timm)
  convnext_base.fb_in22k_ft_in1k
  88.1M parameters
       │
       ▼
Global Average Pool → [1024]
       │
       ▼
Classification Head
  LayerNorm(1024) → Dropout(0.3) → Linear(1024→512) → GELU → Dropout(0.15) → Linear(512→15)
       │
       ▼
15-class multi-hot output (sigmoid)
```

### Ensemble

Predictions from both models are run with TTA (7 passes each) and averaged in probability space before threshold optimization.

### Training Recipe

| Phase | Epochs | Backbone | Head LR | Backbone LR |
|---|---|---|---|---|
| 1 — Transfer learning | 1–8 | Frozen | 1e-3 | — |
| 2 — Fine-tuning | 9–end | Unfrozen | 1e-4 | 1e-5 |

| Setting | EfficientNetV2-S | ConvNeXt-Base |
|---|---|---|
| Total epochs | 30 | 60 |
| Early stopping patience | 7 | 10 |
| Best epoch | 28 | 54 |
| Val AUC | 0.786 | 0.820 |
| drop_path_rate | 0.2 | 0.4 |

- **Loss**: Asymmetric Loss (γ⁻=4.0, γ⁺=1.0) — better than Focal Loss for extreme multi-label imbalance
- **Optimizer**: AdamW (weight decay 1e-2, effective batch 256 via gradient accumulation ×4)
- **Scheduler**: Cosine annealing with 3-epoch linear warmup
- **AMP**: Mixed precision (float16 on CUDA)
- **Hardware**: RTX 5070 Ti (17GB VRAM) — ~2h for EfficientNetV2-S, ~6h for ConvNeXt-Base

### Key Engineering Decisions

**Official patient-level split** — Uses NIH's `train_val_list.txt` and `test_list.txt` with a binary patient-hash inner split (82/18) to prevent data leakage. Avoids the common mistake of random image splits where the same patient appears in train and test.

**No Finding exclusivity** — The NIH labels are NLP-mined and contain ~2% noisy co-occurrences of "No Finding" with pathologies. Enforcing mutual exclusivity at label encoding fixed No Finding AUC from 0.492 → 0.733.

**Anti-shortcut augmentations** — `AutoCropNonBlack` removes scanner borders, `CornerMask` randomly blacks out corners to prevent the model from learning burned-in text/markers as shortcuts rather than actual pathology features.

**Asymmetric Loss over Focal + pos_weight** — Stacking both creates conflicting imbalance corrections. ASL handles positives and negatives with separate gamma values, no pos_weight needed.

**224px over 384px** — Experiments at 384px (batch 32) underperformed 224px (batch 64). The smaller batch size at higher resolution hurt gradient quality more than the resolution gain helped.

**No horizontal flip in TTA** — Chest X-rays have clinically significant laterality (cardiac apex, gastric bubble, effusion side, pneumothorax laterality). Horizontal flip TTA corrupts these features and hurts performance.

**Temperature scaling** — Post-hoc calibration via scalar temperature T fitted on the validation set. Reduces Expected Calibration Error without affecting AUC.

---

## Demo

A full-stack inference demo with:
- Real-time pathology probability bars with severity color coding
- Grad-CAM heatmap for the top predicted finding (INFERNO colormap)
- AI-generated structured radiology report via local LLM (Ollama)

### Stack

| Layer | Technology |
|---|---|
| ML inference + Grad-CAM | PyTorch, timm |
| API | FastAPI + Uvicorn |
| Report generation | Ollama (qwen2.5:latest) |
| Frontend | React 18 + Framer Motion |
| Build tool | Vite |

---

## Setup

### Requirements

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.ai) with `qwen2.5:latest` pulled

### 1. Clone and install Python dependencies

```bash
git clone https://github.com/EltonGino/chest-xray-ai.git
cd chest-xray-ai

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **CUDA (Windows/Linux):** Install PyTorch with CUDA first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
> pip install -r requirements.txt
> ```

### 2. Download the dataset

Download the NIH ChestX-ray14 dataset from [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) and place it as:

```
data/NIH_chest_x-ray/
├── Data_Entry_2017.csv
├── train_val_list.txt
├── test_list.txt
└── images/
    ├── 00000001_000.png
    └── ...
```

Update the `data:` paths in `configs/config.yaml` to match your local setup.

### 3. Train

```bash
# EfficientNetV2-S
python -m src.train --config configs/config.yaml

# ConvNeXt-Base
python -m src.train --config configs/config_convnext_base.yaml
```

### 4. Evaluate

```bash
# Single model with TTA
python -m src.evaluate \
  --checkpoint checkpoints/best_model_efficientnet_v2_s.pt \
  --save-dir outputs/eval \
  --tta

# Ensemble
python -m src.evaluate \
  --ensemble checkpoints/best_model_efficientnet_v2_s.pt checkpoints/best_model_convnext_base.pt \
  --save-dir outputs/eval_ensemble \
  --tta-passes 6
```

If running on a different machine than where training happened, override data paths:

```bash
python -m src.evaluate \
  --ensemble checkpoints/best_model_efficientnet_v2_s.pt checkpoints/best_model_convnext_base.pt \
  --save-dir outputs/eval_ensemble \
  --tta-passes 6 \
  --csv-path "data/NIH_chest_x-ray/Data_Entry_2017.csv" \
  --image-dir "data/NIH_chest_x-ray/images" \
  --train-val-list "data/NIH_chest_x-ray/train_val_list.txt" \
  --test-list "data/NIH_chest_x-ray/test_list.txt"
```

### 5. Run with Docker (recommended)

```bash
# Pull the qwen2.5 model into the ollama container (first time only)
docker compose run --rm ollama ollama pull qwen2.5

# Start API + Ollama
docker compose up --build
```

API available at `http://localhost:8000`, health check at `http://localhost:8000/health`.

The checkpoint is mounted from `./checkpoints/` — it is never baked into the image.

### 6. Run the inference demo (without Docker)

**Terminal 1 — Backend:**
```bash
ollama pull qwen2.5        # first time only
source .venv/bin/activate
python api.py
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

---

## Project Structure

```
nih-chest-xray/
├── src/
│   ├── dataset.py       # NIHChestXrayLocal, transforms, split logic
│   ├── models.py        # ChestXrayClassifier, AsymmetricLoss, FocalLoss
│   ├── train.py         # Training loop, AMP, early stopping, checkpointing
│   └── evaluate.py      # Per-class metrics, Grad-CAM, TTA, ensemble, calibration
├── configs/
│   ├── config.yaml               # EfficientNetV2-S config
│   └── config_convnext_base.yaml # ConvNeXt-Base config
├── frontend/
│   ├── src/App.jsx      # React inference demo
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── tests/
│   └── test_dataset.py  # 29 unit tests for dataset logic
├── .github/
│   └── workflows/
│       └── tests.yml    # CI — runs pytest on every push
├── api.py               # FastAPI inference server
├── Dockerfile           # CPU image for deployment
├── docker-compose.yml   # API + Ollama services
├── checkpoints/         # Saved model weights
└── outputs/             # Evaluation plots and figures
```

---

## Limitations

- **Label noise**: NIH labels are NLP-mined from radiology reports with ~10% estimated noise. Per-class performance reflects label quality as much as model quality.
- **Weak classes**: Pneumonia (0.717) and Infiltration (0.707) are the hardest classes — both are diagnostically ambiguous and have the noisiest labels in CXR14.
- **Not for clinical use**: This is a research/portfolio project. Predictions should not be used for medical diagnosis.
- **Dataset shift**: Performance may degrade on images from scanners or acquisition protocols significantly different from the NIH dataset.

---

## References

- Wang et al., *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks*, CVPR 2017
- Rajpurkar et al., *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*, arXiv 2017
- Ben-Baruch et al., *Asymmetric Loss For Multi-Label Classification*, ICCV 2021
- Tan & Le, *EfficientNetV2: Smaller Models and Faster Training*, ICML 2021
- Liu et al., *A ConvNet for the 2020s*, CVPR 2022

---

## Author

**Elton Gino Santos** — AI Engineer & Computer Vision Specialist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/elton-gino)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/EltonGino)

---

*Dataset: NIH Clinical Center, public domain with attribution. Code: MIT License.*
