"""
Unit tests for src/dataset.py

Covers:
  - encode_findings: multi-hot encoding, No Finding exclusivity
  - patient_hash_split: determinism, valid outputs, distribution
  - patient_hash_split_binary: determinism, no overlap, coverage
  - _filter_df_by_split: patient_hash mode correctness
  - compute_pos_weight_from_csv: shape, clamping, no division by zero
"""

import pandas as pd
import pytest
import torch

from src.dataset import (
    CLASS_NAMES,
    NUM_CLASSES,
    NO_FINDING_IDX,
    encode_findings,
    patient_hash_split,
    patient_hash_split_binary,
    _filter_df_by_split,
    compute_pos_weight_from_csv,
)


# ── encode_findings ────────────────────────────────────────────────────────────

class TestEncodeFindings:
    def test_no_finding(self):
        y = encode_findings("No Finding")
        assert y[NO_FINDING_IDX] == 1.0
        assert y[1:].sum() == 0.0

    def test_single_pathology(self):
        y = encode_findings("Atelectasis")
        assert y[CLASS_NAMES.index("Atelectasis")] == 1.0
        assert y[NO_FINDING_IDX] == 0.0

    def test_multiple_pathologies(self):
        y = encode_findings("Atelectasis|Effusion")
        assert y[CLASS_NAMES.index("Atelectasis")] == 1.0
        assert y[CLASS_NAMES.index("Effusion")] == 1.0
        assert y[NO_FINDING_IDX] == 0.0

    def test_no_finding_exclusivity(self):
        # NIH noise: "No Finding" co-occurring with a pathology — No Finding must be forced to 0
        y = encode_findings("No Finding|Atelectasis")
        assert y[CLASS_NAMES.index("Atelectasis")] == 1.0
        assert y[NO_FINDING_IDX] == 0.0, "No Finding must be 0 when any pathology is present"

    def test_output_shape(self):
        y = encode_findings("Effusion")
        assert y.shape == (NUM_CLASSES,)
        assert y.dtype == torch.float32

    def test_unknown_label_ignored(self):
        y = encode_findings("UnknownDisease")
        assert y.sum() == 0.0

    def test_empty_string(self):
        y = encode_findings("")
        assert y.sum() == 0.0

    def test_all_classes_encodable(self):
        for name in CLASS_NAMES:
            y = encode_findings(name)
            assert y[CLASS_NAMES.index(name)] == 1.0


# ── patient_hash_split ─────────────────────────────────────────────────────────

class TestPatientHashSplit:
    def test_deterministic(self):
        assert patient_hash_split(1234, seed=42) == patient_hash_split(1234, seed=42)

    def test_valid_output(self):
        for pid in range(100):
            assert patient_hash_split(pid) in ("train", "val", "test")

    def test_seed_changes_output(self):
        # Different seeds should produce different assignments for at least some patients
        results_42  = [patient_hash_split(i, seed=42)  for i in range(200)]
        results_99  = [patient_hash_split(i, seed=99)  for i in range(200)]
        assert results_42 != results_99

    def test_approximate_distribution(self):
        splits = [patient_hash_split(i, train_ratio=0.7, val_ratio=0.15, seed=42) for i in range(5000)]
        train_frac = splits.count("train") / len(splits)
        val_frac   = splits.count("val")   / len(splits)
        test_frac  = splits.count("test")  / len(splits)
        assert 0.65 < train_frac < 0.75, f"train fraction out of range: {train_frac:.3f}"
        assert 0.10 < val_frac   < 0.20, f"val fraction out of range: {val_frac:.3f}"
        assert 0.10 < test_frac  < 0.20, f"test fraction out of range: {test_frac:.3f}"


# ── patient_hash_split_binary ──────────────────────────────────────────────────

class TestPatientHashSplitBinary:
    def test_deterministic(self):
        assert patient_hash_split_binary(42, seed=42) == patient_hash_split_binary(42, seed=42)

    def test_valid_output(self):
        for pid in range(100):
            assert patient_hash_split_binary(pid) in ("train", "val")

    def test_no_test_split(self):
        results = [patient_hash_split_binary(i) for i in range(1000)]
        assert "test" not in results

    def test_full_coverage(self):
        # Every patient is assigned — nothing is discarded
        results = [patient_hash_split_binary(i, train_ratio=0.82) for i in range(1000)]
        assert len(results) == 1000
        assert all(r in ("train", "val") for r in results)

    def test_approximate_distribution(self):
        splits = [patient_hash_split_binary(i, train_ratio=0.82, seed=42) for i in range(5000)]
        train_frac = splits.count("train") / len(splits)
        assert 0.78 < train_frac < 0.86, f"train fraction out of range: {train_frac:.3f}"

    def test_no_overlap_between_splits(self):
        train_pids = {i for i in range(2000) if patient_hash_split_binary(i, seed=42) == "train"}
        val_pids   = {i for i in range(2000) if patient_hash_split_binary(i, seed=42) == "val"}
        assert train_pids.isdisjoint(val_pids), "Train and val sets must not share patients"

    def test_seed_changes_output(self):
        r42 = [patient_hash_split_binary(i, seed=42) for i in range(500)]
        r99 = [patient_hash_split_binary(i, seed=99) for i in range(500)]
        assert r42 != r99


# ── _filter_df_by_split ────────────────────────────────────────────────────────

def _make_df(n=200):
    """Minimal DataFrame matching NIH CSV schema."""
    return pd.DataFrame({
        "Image Index":    [f"{i:08d}_000.png" for i in range(n)],
        "Finding Labels": ["No Finding"] * n,
        "Patient ID":     [str(i % 50) for i in range(n)],  # 50 unique patients
    })


class TestFilterDfBySplit:
    def test_patient_hash_no_overlap(self):
        df = _make_df(500)
        train = _filter_df_by_split(df, "train", "patient_hash", None, None, 42, (0.7, 0.15), "Image Index", "Patient ID")
        val   = _filter_df_by_split(df, "val",   "patient_hash", None, None, 42, (0.7, 0.15), "Image Index", "Patient ID")
        test  = _filter_df_by_split(df, "test",  "patient_hash", None, None, 42, (0.7, 0.15), "Image Index", "Patient ID")

        train_pids = set(train["Patient ID"])
        val_pids   = set(val["Patient ID"])
        test_pids  = set(test["Patient ID"])

        assert train_pids.isdisjoint(val_pids),  "Train/val patient overlap"
        assert train_pids.isdisjoint(test_pids), "Train/test patient overlap"
        assert val_pids.isdisjoint(test_pids),   "Val/test patient overlap"

    def test_patient_hash_full_coverage(self):
        df = _make_df(500)
        train = _filter_df_by_split(df, "train", "patient_hash", None, None, 42, (0.7, 0.15), "Image Index", "Patient ID")
        val   = _filter_df_by_split(df, "val",   "patient_hash", None, None, 42, (0.7, 0.15), "Image Index", "Patient ID")
        test  = _filter_df_by_split(df, "test",  "patient_hash", None, None, 42, (0.7, 0.15), "Image Index", "Patient ID")

        total = len(train) + len(val) + len(test)
        assert total == len(df), f"Images lost in split: {len(df)} -> {total}"

    def test_deterministic(self):
        df = _make_df(200)
        a = _filter_df_by_split(df, "train", "patient_hash", None, None, 42, (0.7, 0.15), "Image Index", "Patient ID")
        b = _filter_df_by_split(df, "train", "patient_hash", None, None, 42, (0.7, 0.15), "Image Index", "Patient ID")
        assert list(a["Image Index"]) == list(b["Image Index"])

    def test_invalid_split_raises(self):
        df = _make_df()
        with pytest.raises(AssertionError):
            _filter_df_by_split(df, "invalid", "patient_hash", None, None, 42, (0.7, 0.15), "Image Index", "Patient ID")

    def test_invalid_split_mode_raises(self):
        df = _make_df()
        with pytest.raises(AssertionError):
            _filter_df_by_split(df, "train", "random", None, None, 42, (0.7, 0.15), "Image Index", "Patient ID")


# ── compute_pos_weight_from_csv ────────────────────────────────────────────────

def _make_csv(tmp_path, n=300):
    """Write a minimal CSV to a temp file and return the path."""
    diseases = ["Atelectasis", "Effusion", "No Finding", "Cardiomegaly", "Pneumonia"]
    rows = []
    for i in range(n):
        label = diseases[i % len(diseases)]
        rows.append({
            "Image Index":    f"{i:08d}_000.png",
            "Finding Labels": label,
            "Patient ID":     str(i % 60),
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "Data_Entry_2017.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestComputePosWeight:
    def test_output_shape(self, tmp_path):
        csv = _make_csv(tmp_path)
        w = compute_pos_weight_from_csv(csv, split_mode="patient_hash", split="train")
        assert w.shape == (NUM_CLASSES,)

    def test_output_dtype(self, tmp_path):
        csv = _make_csv(tmp_path)
        w = compute_pos_weight_from_csv(csv, split_mode="patient_hash", split="train")
        assert w.dtype == torch.float32

    def test_all_positive(self, tmp_path):
        csv = _make_csv(tmp_path)
        w = compute_pos_weight_from_csv(csv, split_mode="patient_hash", split="train")
        assert (w > 0).all(), "All weights should be positive"

    def test_clamped_at_50(self, tmp_path):
        # Hernia is very rare — weight should be clamped at 50
        csv = _make_csv(tmp_path)
        w = compute_pos_weight_from_csv(csv, split_mode="patient_hash", split="train")
        assert (w <= 50.0).all(), "Weights should be clamped at 50"

    def test_no_nan_or_inf(self, tmp_path):
        csv = _make_csv(tmp_path)
        w = compute_pos_weight_from_csv(csv, split_mode="patient_hash", split="train")
        assert not torch.isnan(w).any(), "Weights contain NaN"
        assert not torch.isinf(w).any(), "Weights contain Inf"
