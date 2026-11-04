"""Load original UCI signal windows without mixing participants across splits."""
from pathlib import Path
import hashlib
import io
import json
import zipfile

import numpy as np
import requests

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ARCHIVE_SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
CHANNELS = [f"{signal}_{axis}" for signal in ("body_acc", "body_gyro", "total_acc") for axis in "xyz"]
LABELS = ["Walking", "Walking upstairs", "Walking downstairs", "Sitting", "Standing", "Lying down"]


def download(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    target = root / "har.npz"
    if target.exists():
        return target
    response = requests.get(URL, timeout=180)
    response.raise_for_status()
    archive_bytes = response.content
    if hashlib.sha256(archive_bytes).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Dataset archive checksum changed; review the upstream source before proceeding")
    archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    # UCI's outer zip contains the original archive. Read only expected files;
    # never extract arbitrary archive paths onto disk.
    nested = [n for n in archive.namelist() if n.endswith(".zip")]
    if nested:
        archive = zipfile.ZipFile(io.BytesIO(archive.read(nested[0])))
    result = {}
    for split in ("train", "test"):
        prefix = "UCI HAR Dataset/" + split + "/"
        signals = [np.loadtxt(io.BytesIO(archive.read(prefix + f"Inertial Signals/{c}_{split}.txt")), dtype=np.float32) for c in CHANNELS]
        result["x_" + split] = np.stack(signals, axis=1)
        result["y_" + split] = np.loadtxt(io.BytesIO(archive.read(prefix + f"y_{split}.txt")), dtype=np.int64) - 1
        result["subject_" + split] = np.loadtxt(io.BytesIO(archive.read(prefix + f"subject_{split}.txt")), dtype=np.int64)
    np.savez_compressed(target, **result)
    (root / "provenance.json").write_text(json.dumps({"source": URL, "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(), "prepared_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}, indent=2))
    return target


def split_subjects(subjects, seed=42):
    people = np.random.default_rng(seed).permutation(np.unique(subjects))
    if len(people) < 8:
        raise ValueError("Need at least eight participants")
    groups = {"train": people[:-7], "validation": people[-7:-3], "calibration": people[-3:]}
    return {k: np.flatnonzero(np.isin(subjects, v)) for k, v in groups.items()}


def fit_scaler(x):
    return x.mean(axis=(0, 2), keepdims=True), np.maximum(x.std(axis=(0, 2), keepdims=True), 1e-6)


def statistical_features(x):
    """Same nine channels as the CNN; no UCI 561-feature shortcut."""
    return np.concatenate([x.mean(2), x.std(2), x.min(2), x.max(2), np.sqrt((x*x).mean(2))], axis=1)
