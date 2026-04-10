# Chest X-Ray AI — Multi-Label Pathology Classification

> EfficientNetV2-S trained on NIH ChestX-ray14 with a FastAPI + React inference demo, Grad-CAM visualizations, and AI-generated radiology reports via local LLM.

![Demo](outputs/figures/demo.png)
![Demo 1](outputs/figures/demo1.png)

---

## Results

Trained on the full NIH ChestX-ray14 dataset (112,120 images, 30,805 patients) using the official train/test split. Evaluated on the held-out test set of 25,596 images.

| Class | AUC-ROC | AUC-PR | F1 | Prevalence |
|---|---|---|---|---|
| No Finding | 0.714 | 0.627 | 0.599 | 38.5% |
| Atelectasis | 0.739 | 0.297 | 0.359 | 12.8% |
| Cardiomegaly | 0.816 | 0.248 | 0.306 | 4.2% |
| Effusion | 0.805 | 0.480 | 0.508 | 18.2% |
| Infiltration | 0.687 | 0.388 | 0.467 | 23.9% |
| Mass | 0.729 | 0.192 | 0.254 | 6.8% |
| Nodule | 0.691 | 0.170 | 0.231 | 6.3% |
| Pneumonia | 0.679 | 0.042 | 0.084 | 2.2% |
| Pneumothorax | 0.823 | 0.349 | 0.411 | 10.4% |
| Consolidation | 0.716 | 0.133 | 0.219 | 7.1% |
| Edema | 0.818 | 0.130 | 0.204 | 3.6% |
| Emphysema | 0.813 | 0.175 | 0.255 | 4.3% |
| Fibrosis | 0.777 | 0.060 | 0.118 | 1.7% |
| Pleural Thickening | 0.732 | 0.107 | 0.182 | 4.5% |
| Hernia | 0.838 | 0.058 | 0.122 | 0.3% |
| **Mean** | **0.759** | **0.230** | **0.288** | — |

**CheXNet (DenseNet-121) baseline: ~0.745 mean AUC** — Wang et al., CVPR 2017.

---

## Architecture

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
  LayerNorm(1280)
  Dropout(0.3)
  Linear(1280 → 512)
  GELU
  Dropout(0.15)
  Linear(512 → 15)
       │
       ▼
15-class multi-hot output (sigmoid)
```

### Training Recipe

| Phase | Epochs | Backbone | Head LR | Backbone LR |
|---|---|---|---|---|
| 1 — Transfer learning | 1–8 | Frozen | 1e-3 | — |
| 2 — Fine-tuning | 9–30 | Unfrozen | 1e-4 | 1e-5 |

- **Loss**: Asymmetric Loss (γ⁻=4.0, γ⁺=1.0) — better than Focal Loss for extreme multi-label imbalance
- **Optimizer**: AdamW (weight decay 1e-2, effective batch 256 via gradient accumulation ×4)
- **Scheduler**: Cosine annealing with 3-epoch linear warmup
- **AMP**: Mixed precision (float16 on CUDA)
- **Early stopping**: patience 7 on val AUC-ROC
- **Hardware**: RTX 5070 Ti (17GB VRAM) — ~2h10min for 30 epochs at 224px with batch 64

### Key Engineering Decisions

**Official patient-level split** — Uses NIH's `train_val_list.txt` and `test_list.txt` with a binary patient-hash inner split (82/18) to prevent data leakage. Avoids the common mistake of random image splits where the same patient appears in train and test.

**No Finding exclusivity** — The NIH labels are NLP-mined and contain ~2% noisy co-occurrences of "No Finding" with pathologies. Enforcing mutual exclusivity at label encoding fixed No Finding AUC from 0.492 → 0.714.

**Anti-shortcut augmentations** — `AutoCropNonBlack` removes scanner borders, `CornerMask` randomly blacks out corners to prevent the model from learning burned-in text/markers as shortcuts rather than actual pathology features.

**Asymmetric Loss over Focal + pos_weight** — Stacking both creates conflicting imbalance corrections. ASL handles positives and negatives with separate gamma values, no pos_weight needed.

**224px over 384px** — Experiments at 384px (batch 32) underperformed 224px (batch 64). The smaller batch size at higher resolution hurt gradient quality more than the resolution gain helped. 224px with a larger batch converged faster and generalised better.

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

pip install torch torchvision timm
pip install fastapi uvicorn python-multipart opencv-python
pip install pandas scikit-learn matplotlib seaborn pyyaml
```

> **CUDA (Windows/Linux):** Replace the torch install with:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
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
python -m src.train --config configs/config.yaml
```

Approximate training time: ~2h10min on RTX 5070 Ti, ~8–10h on Mac Mini M-series (MPS).

### 4. Evaluate

```bash
python -m src.evaluate \
  --checkpoint checkpoints/best_model_efficientnet_v2_s.pt \
  --save-dir outputs/eval_full_run
```

If running evaluation on a different machine than where training happened, override the data paths:

```bash
python -m src.evaluate \
  --checkpoint checkpoints/best_model_efficientnet_v2_s.pt \
  --save-dir outputs/eval_full_run \
  --csv-path "data/NIH_chest_x-ray/Data_Entry_2017.csv" \
  --image-dir "data/NIH_chest_x-ray/images" \
  --train-val-list "data/NIH_chest_x-ray/train_val_list.txt" \
  --test-list "data/NIH_chest_x-ray/test_list.txt"
```

### 5. Run the inference demo

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
│   └── evaluate.py      # Per-class metrics, Grad-CAM, threshold optimization
├── configs/
│   └── config.yaml      # Full training configuration (update data: paths per machine)
├── frontend/
│   ├── src/App.jsx      # React inference demo
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── api.py               # FastAPI inference server
├── checkpoints/         # Saved model weights
└── outputs/             # Evaluation plots and figures
```

---

## Limitations

- **Label noise**: NIH labels are NLP-mined from radiology reports with ~10% estimated noise. Per-class performance reflects label quality as much as model quality.
- **Weak classes**: Pneumonia (0.679) and Infiltration (0.687) are the hardest classes — both are diagnostically ambiguous and have the noisiest labels in CXR14.
- **Not for clinical use**: This is a research/portfolio project. Predictions should not be used for medical diagnosis.
- **Dataset shift**: Performance may degrade on images from scanners or acquisition protocols significantly different from the NIH dataset.

---

## References

- Wang et al., *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks*, CVPR 2017
- Rajpurkar et al., *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*, arXiv 2017
- Ben-Baruch et al., *Asymmetric Loss For Multi-Label Classification*, ICCV 2021
- Tan & Le, *EfficientNetV2: Smaller Models and Faster Training*, ICML 2021

---

## Author

**Elton Gino Santos** — AI Engineer & Computer Vision Specialist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/elton-gino)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/EltonGino)

---

*Dataset: NIH Clinical Center, public domain with attribution. Code: MIT License.*
