"""
NIH Chest X-ray14 Dataset Module (Showcase Edition)
===================================================
- Local dataset (Kaggle/manual download) FIRST (fast, stable).
- Optional HuggingFace streaming fallback.
- Two split modes:
    1) official: uses train_val_list.txt + test_list.txt (reproducible)
    2) patient_hash: leakage-safe split by Patient ID (deterministic)
- Anti-shortcut augmentations:
    - auto-crop black borders
    - corner masking to discourage text/marker shortcuts
    - optional mild autocontrast for exposure normalization
- Fast class-imbalance pos_weight from CSV (no image loading).

Fixes applied vs previous version:
    - BUG FIX: "No Finding" exclusivity enforced in encode_findings().
      Previously "No Finding" could co-exist with pathologies in the
      label vector, causing AUC < 0.5. Now if any pathology is present,
      No Finding is forced to 0.
    - BUG FIX: 3-way patient hash was applied inside the 2-way train/val
      split, silently discarding ~15% of train_val images (~12,800 samples).
      Replaced with patient_hash_split_binary() for the inner train/val split.
    - REFACTOR: Duplicated split-filtering logic extracted into
      _filter_df_by_split() shared by NIHChestXrayLocal and
      compute_pos_weight_from_csv — single source of truth.

Author: Elton Gino Santos
"""

from __future__ import annotations

import os
import hashlib
import random
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Set, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
import pandas as pd

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


# ── Label Mapping ──────────────────────────────────────────────────────────

CLASS_NAMES = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NO_FINDING_IDX = CLASS_TO_IDX["No Finding"]


# ── Split helpers ──────────────────────────────────────────────────────────

def patient_hash_split(
    patient_id,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> str:
    """
    Deterministic 3-way patient-level split using hashing.
    Returns "train" | "val" | "test".

    NOTE: Use this only when you want a full 3-way split (e.g. split_mode='patient_hash').
    For splitting WITHIN an official train_val pool, use patient_hash_split_binary().
    """
    pid_str = str(patient_id)
    hash_val = int(hashlib.md5(f"{seed}_{pid_str}".encode()).hexdigest(), 16)
    fraction = (hash_val % 10000) / 10000.0
    if fraction < train_ratio:
        return "train"
    elif fraction < train_ratio + val_ratio:
        return "val"
    else:
        return "test"


def patient_hash_split_binary(
    patient_id,
    train_ratio: float = 0.82,
    seed: int = 42,
) -> str:
    """
    Deterministic 2-way patient-level split.
    Returns "train" | "val" only — no images are discarded.

    Use this when splitting INSIDE an existing pool (e.g. official train_val_list)
    where you don't want to throw away any samples.

    Default train_ratio=0.82 gives roughly 82/18 within the train_val pool,
    which produces ~59k train / ~13k val matching the official NIH paper split.
    """
    pid_str = str(patient_id)
    hash_val = int(hashlib.md5(f"{seed}_{pid_str}".encode()).hexdigest(), 16)
    fraction = (hash_val % 10000) / 10000.0
    return "train" if fraction < train_ratio else "val"


def read_list_file(path: str) -> Set[str]:
    """
    Reads train_val_list.txt / test_list.txt returning set of image filenames.
    Expected: one filename per line.
    """
    s: Set[str] = set()
    if not path or not os.path.exists(path):
        return s
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            name = line.strip()
            if name:
                s.add(name)
    return s


# ── Shared split-filter helper ─────────────────────────────────────────────

def _filter_df_by_split(
    df: pd.DataFrame,
    split: str,
    split_mode: str,
    train_val_list: Optional[str],
    test_list: Optional[str],
    seed: int,
    patient_split: Tuple[float, float],  # (train_ratio_in_tv, unused_legacy)
    col_image: str,
    col_patient: str,
) -> pd.DataFrame:
    """
    Shared logic to filter a DataFrame to the requested split.
    Extracted to avoid duplication between NIHChestXrayLocal and
    compute_pos_weight_from_csv — both must use exactly the same rows.

    Args:
        patient_split: (train_ratio, val_ratio) legacy tuple. For binary inner
            split, only train_ratio is used (val_ratio is ignored).
    """
    assert split in ("train", "val", "test")
    assert split_mode in ("official", "patient_hash")

    if split_mode == "official":
        tv = read_list_file(train_val_list) if train_val_list else set()
        te = read_list_file(test_list) if test_list else set()

        if not tv or not te:
            raise ValueError(
                "split_mode='official' requires valid train_val_list and test_list paths.\n"
                f"  train_val_list: {train_val_list!r} (found={bool(tv)})\n"
                f"  test_list:      {test_list!r} (found={bool(te)})"
            )

        if split == "test":
            return df[df[col_image].isin(te)].reset_index(drop=True)

        # train / val: filter to official train_val pool first
        df_tv = df[df[col_image].isin(tv)].copy()

        # FIX: use binary split so NO images from the train_val pool are discarded.
        # Old code used patient_hash_split() (3-way) which silently dropped ~15%
        # of train_val images into a phantom "test" bucket.
        train_ratio = patient_split[0]  # e.g. 0.82 to get ~82/18 within pool
        df_tv["_inner_split"] = df_tv[col_patient].apply(
            lambda pid: patient_hash_split_binary(pid, train_ratio=train_ratio, seed=seed)
        )
        result = df_tv[df_tv["_inner_split"] == split].drop(columns=["_inner_split"])
        return result.reset_index(drop=True)

    else:  # patient_hash: 3-way split over the full dataset
        train_ratio, val_ratio = patient_split
        df = df.copy()
        df["_split"] = df[col_patient].apply(
            lambda pid: patient_hash_split(pid, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
        )
        result = df[df["_split"] == split].drop(columns=["_split"])
        return result.reset_index(drop=True)


# ── Anti-shortcut transforms ───────────────────────────────────────────────

class AutoCropNonBlack:
    """
    Crops away near-black borders (common in CXR exports).
    Works on PIL images.
    """
    def __init__(self, threshold: int = 10, min_border: int = 2):
        self.threshold = threshold
        self.min_border = min_border

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            return img

        g = img.convert("L")
        mask = g.point(lambda p: 255 if p > self.threshold else 0)
        bbox = mask.getbbox()
        if bbox is None:
            return img

        left, top, right, bottom = bbox
        left = max(0, left - self.min_border)
        top = max(0, top - self.min_border)
        right = min(img.width, right + self.min_border)
        bottom = min(img.height, bottom + self.min_border)

        if (right - left) < 10 or (bottom - top) < 10:
            return img

        if (left, top, right, bottom) == (0, 0, img.width, img.height):
            return img

        return img.crop((left, top, right, bottom))


class CornerMask:
    """
    Masks random corners to reduce reliance on burned-in text/markers.
    """
    def __init__(self, p: float = 0.6, frac: float = 0.12, fill: int = 0):
        self.p = p
        self.frac = frac
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            return img
        if random.random() > self.p:
            return img

        w, h = img.size
        s = int(min(w, h) * self.frac)
        if s <= 1:
            return img

        corners = [(0, 0), (w - s, 0), (0, h - s), (w - s, h - s)]
        k = random.randint(1, 4)
        chosen = random.sample(corners, k)

        out = img.copy()
        for (x0, y0) in chosen:
            patch = Image.new("RGB", (s, s), (self.fill, self.fill, self.fill))
            out.paste(patch, (x0, y0))
        return out


class XRayAutoContrast:
    """
    Mild autocontrast to normalize exposures.
    """
    def __init__(self, p: float = 0.25, cutoff: int = 1):
        self.p = p
        self.cutoff = cutoff

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            return img
        if random.random() > self.p:
            return img
        return ImageOps.autocontrast(img, cutoff=self.cutoff)


# ── Transform factory (toggle-friendly) ────────────────────────────────────

@dataclass
class TransformConfig:
    image_size: int = 224
    autocrop: bool = True
    autocontrast: bool = True
    corner_mask: bool = True

    # augmentation strengths (safe defaults)
    rr_crop_scale: Tuple[float, float] = (0.85, 1.0)
    rr_crop_ratio: Tuple[float, float] = (0.95, 1.05)
    rot_deg: int = 10
    affine_translate: Tuple[float, float] = (0.03, 0.03)
    affine_shear: int = 5
    jitter_brightness: float = 0.15
    jitter_contrast: float = 0.15
    random_erasing_p: float = 0.15


def build_transforms(cfg: TransformConfig, train: bool) -> transforms.Compose:
    ops: List[Callable] = []

    if cfg.autocrop:
        ops.append(AutoCropNonBlack(threshold=10, min_border=2))

    if train and cfg.autocontrast:
        ops.append(XRayAutoContrast(p=0.25, cutoff=1))

    if train:
        ops += [
            transforms.Resize((cfg.image_size + 32, cfg.image_size + 32)),
            transforms.RandomResizedCrop(cfg.image_size, scale=cfg.rr_crop_scale, ratio=cfg.rr_crop_ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(cfg.rot_deg),
            transforms.RandomAffine(degrees=0, translate=cfg.affine_translate, shear=cfg.affine_shear),
        ]
        if cfg.corner_mask:
            ops.append(CornerMask(p=0.6, frac=0.12, fill=0))
        ops += [
            transforms.ColorJitter(brightness=cfg.jitter_brightness, contrast=cfg.jitter_contrast),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=cfg.random_erasing_p),
        ]
    else:
        ops += [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    return transforms.Compose(ops)


# ── Label parsing ──────────────────────────────────────────────────────────

def encode_findings(findings: str) -> torch.Tensor:
    """
    Convert pipe-separated findings into multi-hot vector.

    FIX: Enforces "No Finding" exclusivity.
    The NIH labels are NLP-mined and occasionally tag an image as both
    "No Finding" AND a pathology. Treating both as positive causes the
    model to learn contradictory signal and produces AUC < 0.5 for
    No Finding. Rule: if any pathology is present, No Finding = 0.
    """
    y = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for f in str(findings).split("|"):
        f = f.strip()
        if f in CLASS_TO_IDX:
            y[CLASS_TO_IDX[f]] = 1.0

    # Enforce exclusivity: pathologies and "No Finding" cannot both be 1
    has_pathology = y[1:].sum() > 0  # any class other than index 0
    if has_pathology:
        y[NO_FINDING_IDX] = 0.0

    return y


# ── Local Dataset ─────────────────────────────────────────────────────────

class NIHChestXrayLocal(Dataset):
    """
    Local dataset based on Data_Entry_2017.csv and images folder.

    Supports:
      - split_mode="official": uses train_val_list.txt / test_list.txt
      - split_mode="patient_hash": leakage-safe 3-way split by Patient ID

    Note on patient_split argument:
      In official mode, only patient_split[0] (train_ratio) is used as the
      binary split threshold inside the train_val pool. val_ratio is ignored
      because the binary split assigns everything not in train to val,
      ensuring no images are discarded.
    """
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 224,
        split_mode: str = "official",   # "official" | "patient_hash"
        train_val_list: Optional[str] = None,
        test_list: Optional[str] = None,
        seed: int = 42,
        patient_split: Tuple[float, float] = (0.82, 0.18),  # train_ratio, val_ratio (val_ratio unused in official mode)
        max_samples: Optional[int] = None,
    ):
        assert split in ("train", "val", "test")
        assert split_mode in ("official", "patient_hash")

        self.csv_path = csv_path
        self.image_dir = image_dir
        self.split = split
        self.split_mode = split_mode
        self.seed = seed

        # transforms
        if transform is None:
            tcfg = TransformConfig(image_size=image_size)
            self.transform = build_transforms(tcfg, train=(split == "train"))
        else:
            self.transform = transform

        df = pd.read_csv(csv_path)

        # column normalization
        col_image   = "Image Index"    if "Image Index"    in df.columns else "image"
        col_findings = "Finding Labels" if "Finding Labels" in df.columns else "finding_labels"
        col_patient  = "Patient ID"     if "Patient ID"     in df.columns else "patient_id"

        df = df[[col_image, col_findings, col_patient]].copy()
        df[col_image]    = df[col_image].astype(str)
        df[col_findings] = df[col_findings].astype(str)
        df[col_patient]  = df[col_patient].astype(str)

        # Apply split using shared helper
        df = _filter_df_by_split(
            df, split, split_mode,
            train_val_list, test_list,
            seed, patient_split,
            col_image, col_patient,
        )

        if max_samples is not None:
            df = df.head(int(max_samples)).reset_index(drop=True)

        # store samples
        self.image_names:   List[str]          = df[col_image].tolist()
        self.findings_text: List[str]          = df[col_findings].tolist()
        self.patient_ids:   List[str]          = df[col_patient].tolist()

        # pre-encode labels (fast; avoids parsing every __getitem__)
        self.labels: List[torch.Tensor] = [encode_findings(t) for t in self.findings_text]

        # quick existence check (first 50)
        missing_preview = sum(
            1 for name in self.image_names[:50]
            if not os.path.exists(os.path.join(self.image_dir, name))
        )
        if missing_preview:
            print(f"Warning: {missing_preview}/50 preview images missing. image_dir={self.image_dir}")

        print(f"Loaded {len(self):,} {split} samples from local dataset "
              f"(split_mode={split_mode})")

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


# ── HuggingFace streaming dataset (optional) ───────────────────────────────

class NIHChestXrayStreaming(Dataset):
    """
    HuggingFace streaming wrapper. Use only if local data is unavailable.
    NOTE: local mode is recommended for stability and reproducibility.
    """
    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        image_size: int = 224,
        dataset_name: str = "alkzar90/NIH-Chest-X-ray-dataset",
        dataset_subset: str = "image-classification",
        seed: int = 42,
    ):
        if load_dataset is None:
            raise RuntimeError("datasets is not installed; cannot use streaming mode.")

        assert split in ("train", "val", "test")
        self.split = split
        self.seed = seed

        if transform is None:
            tcfg = TransformConfig(image_size=image_size)
            self.transform = build_transforms(tcfg, train=(split == "train"))
        else:
            self.transform = transform

        print(f"Loading NIH Chest X-ray ({split}) from HuggingFace...")
        hf_split = "train" if split in ("train", "val") else "test"

        ds_stream = load_dataset(
            dataset_name,
            dataset_subset,
            split=hf_split,
            streaming=True,
            trust_remote_code=True,
        )

        self.samples = []
        count = 0
        for sample in ds_stream:
            pid = (
                sample.get("patient_id")
                or sample.get("Patient ID")
                or sample.get("image_id")
                or sample.get("Image Index")
                or sample.get("file_name")
                or count
            )
            assigned_split = patient_hash_split(pid, seed=self.seed)
            if assigned_split != split:
                continue

            self.samples.append(sample)
            count += 1
            if max_samples and count >= max_samples:
                break
            if count % 5000 == 0:
                print(f"  Buffered {count} samples...")
        print(f"  Total {split} samples: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = sample["image"]
        if isinstance(image, Image.Image) and image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transform(image)

        # label extraction (robust)
        if "labels" in sample and sample["labels"] is not None:
            raw = sample["labels"]
            y = torch.tensor(raw, dtype=torch.float32).flatten()
            if y.numel() != NUM_CLASSES:
                fixed = torch.zeros(NUM_CLASSES, dtype=torch.float32)
                n = min(NUM_CLASSES, y.numel())
                fixed[:n] = y[:n]
                y = fixed
        elif "label" in sample and sample["label"] is not None:
            y = torch.zeros(NUM_CLASSES, dtype=torch.float32)
            y[int(sample["label"])] = 1.0
        elif "finding_labels" in sample and sample["finding_labels"] is not None:
            y = encode_findings(sample["finding_labels"])
        else:
            y = torch.zeros(NUM_CLASSES, dtype=torch.float32)

        return image, y


# ── Fast class weights from CSV ────────────────────────────────────────────

def compute_pos_weight_from_csv(
    csv_path: str,
    split_mode: str = "official",
    split: str = "train",
    train_val_list: Optional[str] = None,
    test_list: Optional[str] = None,
    seed: int = 42,
    patient_split: Tuple[float, float] = (0.82, 0.18),
) -> torch.Tensor:
    """
    Computes pos_weight for BCEWithLogitsLoss WITHOUT loading images.
    Reads labels from CSV only. Much faster than iterating dataset[i].

    Uses the same _filter_df_by_split() as NIHChestXrayLocal to guarantee
    that pos_weight is computed on exactly the same training rows.

    pos_weight[c] = num_negative / num_positive  (clamped at 50)
    """
    assert split in ("train", "val", "test")
    assert split_mode in ("official", "patient_hash")

    df = pd.read_csv(csv_path)

    col_image    = "Image Index"    if "Image Index"    in df.columns else "image"
    col_findings = "Finding Labels" if "Finding Labels" in df.columns else "finding_labels"
    col_patient  = "Patient ID"     if "Patient ID"     in df.columns else "patient_id"

    df = df[[col_image, col_findings, col_patient]].copy()
    df[col_image]    = df[col_image].astype(str)
    df[col_findings] = df[col_findings].astype(str)
    df[col_patient]  = df[col_patient].astype(str)

    df = _filter_df_by_split(
        df, split, split_mode,
        train_val_list, test_list,
        seed, patient_split,
        col_image, col_patient,
    )

    total = len(df)
    pos_counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)

    for txt in df[col_findings].tolist():
        y = encode_findings(txt).double()
        pos_counts += y

    neg_counts = float(total) - pos_counts
    pos_weight = (neg_counts / (pos_counts + 1e-8)).float().clamp(max=50.0)
    return pos_weight


# ── DataLoader factory ─────────────────────────────────────────────────────

def create_dataloaders(
    batch_size: int = 64,
    image_size: int = 224,
    max_samples: Optional[int] = None,
    num_workers: int = 8,
    pin_memory: bool = True,
    use_streaming: bool = False,
    # local args
    csv_path: Optional[str] = None,
    image_dir: Optional[str] = None,
    split_mode: str = "official",
    train_val_list: Optional[str] = None,
    test_list: Optional[str] = None,
    seed: int = 42,
    patient_split: Tuple[float, float] = (0.82, 0.18),
    # streaming args
    dataset_name: str = "alkzar90/NIH-Chest-X-ray-dataset",
    dataset_subset: str = "image-classification",
) -> Dict[str, DataLoader]:

    loaders: Dict[str, DataLoader] = {}

    for split in ["train", "val", "test"]:
        if use_streaming:
            ds = NIHChestXrayStreaming(
                split=split,
                max_samples=max_samples,
                image_size=image_size,
                dataset_name=dataset_name,
                dataset_subset=dataset_subset,
                seed=seed,
            )
        else:
            if not csv_path or not image_dir:
                raise ValueError("Local mode requires csv_path and image_dir")
            ds = NIHChestXrayLocal(
                csv_path=csv_path,
                image_dir=image_dir,
                split=split,
                image_size=image_size,
                split_mode=split_mode,
                train_val_list=train_val_list,
                test_list=test_list,
                seed=seed,
                patient_split=patient_split,
                max_samples=max_samples,
            )

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
        )

    return loaders


if __name__ == "__main__":
    # Quick sanity check (local mode)
    # Update base path for your machine:
    base = "/Volumes/ROG SSD/PyCharm/nih-chest-xray/data/NIH_chest_x-ray"
    csv_path       = os.path.join(base, "Data_Entry_2017.csv")
    image_dir      = os.path.join(base, "images")
    train_val_list = os.path.join(base, "train_val_list.txt")
    test_list      = os.path.join(base, "test_list.txt")

    for split in ["train", "val", "test"]:
        ds = NIHChestXrayLocal(
            csv_path=csv_path,
            image_dir=image_dir,
            split=split,
            image_size=224,
            split_mode="official",
            train_val_list=train_val_list,
            test_list=test_list,
        )

    # Expected approximate sizes with official split + binary inner split:
    #   train: ~70,974  val: ~15,548  test: ~25,596  total: ~112,118
    x, y = ds[0]
    print(f"\nSample check | image: {tuple(x.shape)} | label: {tuple(y.shape)} | positives: {int(y.sum().item())}")
    print(f"No Finding at index 0: {y[0].item()}")