"""python -m motionsense.train --epochs 35"""
import argparse
import copy
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .data import CHANNELS, LABELS, download, fit_scaler, split_subjects, statistical_features
from .model import TemporalCNN, calibration_metrics, fit_temperature

ROOT = Path(__file__).resolve().parents[1]


def scores(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro"))}


def cluster_accuracy_interval(y, pred, subjects, seed):
    """Bootstrap participants, preserving dependence of overlapping windows."""
    people = np.unique(subjects)
    correct = np.array([(pred[subjects == s] == y[subjects == s]).sum() for s in people])
    count = np.array([(subjects == s).sum() for s in people])
    draws = np.random.default_rng(seed).integers(len(people), size=(2000, len(people)))
    return np.quantile(correct[draws].sum(1) / count[draws].sum(1), [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    data_file = download(ROOT / "data")
    data = np.load(data_file)
    x, y, subjects = (data[k] for k in ("x_train", "y_train", "subject_train"))
    ids = split_subjects(subjects, args.seed)
    assert not set(subjects) & set(data["subject_test"])
    mean, std = fit_scaler(x[ids["train"]])
    scaled = (x - mean) / std
    net = TemporalCNN()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.001)
    loader = DataLoader(TensorDataset(torch.from_numpy(scaled[ids["train"]]), torch.from_numpy(y[ids["train"]])), batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(args.seed))

    def logits_for(values):
        net.eval()
        with torch.inference_mode():
            return torch.cat([net(torch.from_numpy(v)) for v in np.array_split(values, max(1, int(np.ceil(len(values)/256))))]).numpy()

    start = time.perf_counter()
    history, best, best_state, stale = [], float("inf"), None, 0
    for epoch in range(args.epochs):
        net.train()
        total = 0
        for features, labels in loader:
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(features), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        validation_logits = logits_for(scaled[ids["validation"]])
        val = calibration_metrics(validation_logits, y[ids["validation"]])
        history.append({"epoch": epoch + 1, "train_loss": total / len(ids["train"]), "validation_nll": val["nll"]})
        print(json.dumps(history[-1]), flush=True)
        if val["nll"] < best:
            best, best_state, stale = val["nll"], copy.deepcopy(net.state_dict()), 0
        else:
            stale += 1
        if stale >= 8:
            break
    train_seconds = time.perf_counter() - start
    net.load_state_dict(best_state)
    calibration_logits = logits_for(scaled[ids["calibration"]])
    temperature = fit_temperature(calibration_logits, y[ids["calibration"]])
    # Test labels are first consumed here, after checkpoint and temperature selection.
    xt, yt, st = (data[k] for k in ("x_test", "y_test", "subject_test"))
    test_scaled = (xt - mean) / std
    test_logits = logits_for(test_scaled)
    pred = test_logits.argmax(1)
    probability = softmax(test_logits / temperature, axis=1)
    baseline = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000))
    baseline.fit(statistical_features(x[ids["train"]]), y[ids["train"]])
    baseline_pred = baseline.predict(statistical_features(xt))
    # Batch-one model-only CPU timing; no HTTP, JSON serialization, or preprocessing.
    single = torch.from_numpy(test_scaled[:1])
    timings = []
    net.eval()
    with torch.inference_mode():
        for _ in range(25):
            net(single)
        for _ in range(200):
            t = time.perf_counter()
            net(single)
            timings.append((time.perf_counter() - t)*1000)
    accepted = probability.max(1) >= 0.8
    metrics = {
        "seed": args.seed,
        "dataset_sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
        "split_subjects": {**{k: sorted(np.unique(subjects[v]).tolist()) for k, v in ids.items()}, "test": sorted(np.unique(st).tolist())},
        "split_windows": {**{k: len(v) for k, v in ids.items()}, "test": len(yt)},
        "cnn": scores(yt, pred), "logistic_baseline": scores(yt, baseline_pred),
        "accuracy_95pct_subject_bootstrap": cluster_accuracy_interval(yt, pred, st, args.seed),
        "calibration": {"temperature": temperature, "test_before": calibration_metrics(test_logits, yt), "test_after": calibration_metrics(test_logits, yt, temperature), "fit_before": calibration_metrics(calibration_logits, y[ids["calibration"]]), "fit_after": calibration_metrics(calibration_logits, y[ids["calibration"]], temperature)},
        "selective_prediction": {"threshold": 0.8, "coverage": float(accepted.mean()), "accepted_accuracy": float((pred[accepted] == yt[accepted]).mean()) if accepted.any() else None},
        "per_subject": {str(s): scores(yt[st == s], pred[st == s]) for s in np.unique(st)},
        "classification_report": classification_report(yt, pred, target_names=LABELS, output_dict=True),
        "confusion_matrix": confusion_matrix(yt, pred).tolist(),
        "training_seconds": train_seconds, "best_epoch": min(history, key=lambda h: h["validation_nll"])["epoch"],
        "parameters": sum(p.numel() for p in net.parameters()),
        "model_only_latency_ms": {"p50": float(np.median(timings)), "p95": float(np.quantile(timings, .95)), "samples": 200, "threads": 4},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform(), "processor": platform.processor()},
    }
    torch.save({"state_dict": net.state_dict(), "mean": torch.tensor(mean), "std": torch.tensor(std), "temperature": temperature, "labels": LABELS, "channels": CHANNELS, "schema_version": 1}, out / "model.pt")
    metrics["model_sha256"] = hashlib.sha256((out / "model.pt").read_bytes()).hexdigest()
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "provenance.json").write_text((ROOT / "data" / "provenance.json").read_text())
    # One real held-out example per activity, compact enough to ship with the demo.
    examples = []
    for label in range(6):
        idx = int(np.flatnonzero(yt == label)[0])
        examples.append({"label": LABELS[label], "subject": int(st[idx]), "test_index": idx, "window": xt[idx].T.tolist()})
    (out / "examples.json").write_text(json.dumps(examples))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    matrix = confusion_matrix(yt, pred)
    axes[0].imshow(matrix, cmap="Blues")
    for i in range(6):
        for j in range(6):
            axes[0].text(j, i, matrix[i,j], ha="center", va="center", color="white" if matrix[i,j] > matrix.max()/2 else "#172554")
    short = ["Walk", "Upstairs", "Downstairs", "Sit", "Stand", "Lie"]
    axes[0].set(xticks=range(6), yticks=range(6), xticklabels=short, yticklabels=short, xlabel="Predicted", ylabel="Actual", title="Held-out participant confusion matrix")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="Training")
    axes[1].plot([h["epoch"] for h in history], [h["validation_nll"] for h in history], label="Validation participants")
    axes[1].set(xlabel="Epoch", ylabel="Cross entropy", title="Learning curves")
    axes[1].legend()
    fig.savefig(out / "evaluation.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: metrics[k] for k in ("cnn", "logistic_baseline", "training_seconds", "model_only_latency_ms")}, indent=2))


if __name__ == "__main__":
    main()
