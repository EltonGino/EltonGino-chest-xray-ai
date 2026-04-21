"""
FastAPI Inference API — NIH Chest X-ray14 Classifier
=====================================================
Endpoints:
  GET  /health        — model status
  POST /predict       — image → predictions + Grad-CAM + radiology report

Author: Elton Gino Santos
"""

import io
import os
import json
import base64
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import requests
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Adjust sys.path so src.* imports work when running from project root ──
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import build_model
from src.dataset import CLASS_NAMES, NUM_CLASSES, TransformConfig, build_transforms

# ── Config ─────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = "checkpoints/best_model_efficientnet_v2_s.pt"
IMAGE_SIZE = 224

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = _get_device()

# ── Global state ────────────────────────────────────────────────────────────

_state: dict = {}


# ── Grad-CAM ────────────────────────────────────────────────────────────────

class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, input, output):
        self._activations = output.detach()

    def _bwd_hook(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        logits = self.model(input_tensor)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        grads = self._gradients[0]        # [C, H, W]
        acts  = self._activations[0]      # [C, H, W]
        weights = grads.mean(dim=(1, 2))  # [C]

        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = torch.relu(cam).cpu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def _find_last_conv(model: torch.nn.Module) -> torch.nn.Module:
    """Return the last Conv2d layer in model.backbone."""
    last = None
    for m in model.backbone.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found in backbone.")
    return last


def _overlay_gradcam(
    gradcam: GradCAM,
    image_tensor: torch.Tensor,
    original_pil: Image.Image,
    class_idx: int,
) -> str:
    """Generate heatmap overlay and return as base64-encoded PNG."""
    cam = gradcam.generate(image_tensor.unsqueeze(0), class_idx)

    # Resize CAM → image size
    cam_resized = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig = np.array(original_pil.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB"))
    overlay = (orig * 0.55 + heatmap * 0.45).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Radiology Report (Ollama) ────────────────────────────────────────────────

OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:latest")


def _generate_report(predictions: dict[str, float]) -> str:
    """Call local Ollama (qwen2.5) to produce a structured radiology report."""
    pathology_lines = [
        f"  • {cls}: {prob:.1%}"
        for cls, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        if cls != "No Finding" and prob > 0.15
    ]
    findings_block = "\n".join(pathology_lines) if pathology_lines else "  • No significant pathological findings."
    no_finding_prob = predictions.get("No Finding", 0.0)

    prompt = f"""You are a board-certified radiologist reviewing AI-assisted chest X-ray analysis results.

AI Model Output:
  Normal probability : {no_finding_prob:.1%}
  Detected findings  :
{findings_block}

Write a concise, professional radiology report with three sections:

TECHNIQUE:
(One sentence describing the study type and patient positioning — invent plausible generic details.)

FINDINGS:
(Describe each detected abnormality using precise anatomical and radiological language. If no findings, state the chest is unremarkable.)

IMPRESSION:
(Two to three sentence clinical summary with a recommendation.)

Use formal medical language. Do not mention AI, machine learning, or probability scores. Write as if you reviewed the film directly."""

    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


# ── App lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading checkpoint: {CHECKPOINT_PATH}  →  device: {DEVICE}")
    ckpt   = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    config = ckpt.get("config", {})
    arch   = config.get("model", {}).get("architecture", "efficientnet_v2_s")

    model = build_model(arch=arch, num_classes=NUM_CLASSES, pretrained=False, freeze_backbone=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()

    # Build GradCAM once at startup — avoids registering new hooks on every request
    last_conv = _find_last_conv(model)
    gradcam   = GradCAM(model, last_conv)

    # Load temperature from calibration file if available
    temperature = 1.0
    for candidate in [
        Path("outputs/eval_full_run/calibration/calibration.json"),
        Path("outputs/evaluation/calibration/calibration.json"),
    ]:
        if candidate.exists():
            cal_data = json.loads(candidate.read_text())
            temperature = float(cal_data.get("temperature", 1.0))
            print(f"Loaded temperature T={temperature:.4f} from {candidate}")
            break
    else:
        print("No calibration.json found — using T=1.0 (uncalibrated)")

    _state["model"]       = model
    _state["gradcam"]     = gradcam
    _state["arch"]        = arch
    _state["epoch"]       = ckpt.get("epoch", "?")
    _state["best_auc"]    = ckpt.get("best_auc", 0.0)
    _state["temperature"] = temperature
    print(f"Model ready — arch={arch}, epoch={_state['epoch']}, best_auc={_state['best_auc']:.4f}")
    yield
    _state.clear()


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Chest X-Ray AI API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "architecture": _state.get("arch"),
        "epoch": _state.get("epoch"),
        "best_auc": round(_state.get("best_auc", 0), 4),
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    generate_report: bool = True,
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # ── Load & preprocess ──
    contents = await file.read()
    try:
        original_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    tcfg   = TransformConfig(image_size=IMAGE_SIZE)
    tfm    = build_transforms(tcfg, train=False)
    tensor = tfm(original_pil).to(DEVICE)

    model:       torch.nn.Module = _state["model"]
    gradcam:     GradCAM         = _state["gradcam"]
    temperature: float           = _state.get("temperature", 1.0)

    # ── Inference ──
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0))
        probs  = torch.sigmoid(logits / temperature)[0].cpu()

    predictions = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}

    # ── Grad-CAM for top pathology ──
    gradcam_b64   = None
    gradcam_class = None

    pathology_idx_probs = [(i, float(probs[i])) for i in range(1, NUM_CLASSES)]
    top_idx, top_prob   = max(pathology_idx_probs, key=lambda x: x[1])

    if top_prob > 0.10:
        try:
            gradcam_b64   = _overlay_gradcam(gradcam, tensor, original_pil, top_idx)
            gradcam_class = CLASS_NAMES[top_idx]
        except Exception as exc:
            print(f"[WARN] Grad-CAM failed: {exc}")

    # ── Radiology report ──
    report = None
    if generate_report:
        try:
            report = _generate_report(predictions)
        except Exception as exc:
            print(f"[WARN] Report generation failed: {exc}")
            report = "Report generation unavailable — please ensure ANTHROPIC_API_KEY is set."

    return JSONResponse({
        "predictions": dict(
            sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        ),
        "gradcam":       gradcam_b64,
        "gradcam_class": gradcam_class,
        "report":        report,
        "meta": {
            "architecture": _state["arch"],
            "epoch":        _state["epoch"],
            "best_auc":     round(_state.get("best_auc", 0), 4),
        },
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)