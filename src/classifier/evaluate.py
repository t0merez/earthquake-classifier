"""Evaluation and error analysis for the CNN classifier."""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
)
from scipy.signal import spectrogram as _spectrogram

from src.classifier.model import SpectrogramCNN


def _load_test_arrays(data_dir: str):
    """Load and normalise spectrograms, plus raw waveforms and metadata for the test split."""
    X_spec = np.load(f"{data_dir}/spectrograms/spectrograms_10x10.npy")[:, np.newaxis, :, :]
    X_wave = np.load(f"{data_dir}/waveforms.npy")
    meta   = pd.read_csv(f"{data_dir}/metadata.csv", low_memory=False)

    # Normalise using training-set statistics only to match training
    train_mask = (meta["split"] == "train").values
    mu  = float(X_spec[train_mask].mean())
    std = float(X_spec[train_mask].std())
    X_spec = ((X_spec - mu) / (std + 1e-8)).astype(np.float32)

    test_mask  = (meta["split"] == "test").values
    test_meta  = meta[test_mask].reset_index(drop=True)
    labels     = meta.loc[test_mask, "label"].values

    return X_spec[test_mask], X_wave[test_mask], test_meta, labels


def _infer(checkpoint_path: str, X_test: np.ndarray, device: str) -> np.ndarray:
    """Run inference and return sigmoid probabilities for the test array."""
    model = SpectrogramCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    loader = DataLoader(TensorDataset(torch.from_numpy(X_test)), batch_size=512, shuffle=False)
    probs  = []
    with torch.no_grad():
        for (batch,) in loader:
            probs.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    return np.concatenate(probs)


def compute_metrics(checkpoint_path: str, data_dir: str = "data") -> dict:
    """Compute and print test-set metrics; plot confusion matrix and ROC curve.

    Returns:
        dict with keys: f1, accuracy, precision, recall, auc_roc
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_test, _, _, labels = _load_test_arrays(data_dir)
    probs = _infer(checkpoint_path, X_test, device)
    preds = (probs >= 0.5).astype(int)

    print("=" * 50)
    print("Test set metrics")
    print("=" * 50)
    print(classification_report(labels, preds, target_names=["Noise", "Earthquake"], digits=4))

    metrics = {
        "f1":        f1_score(labels, preds, average="macro"),
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall":    recall_score(labels, preds),
        "auc_roc":   roc_auc_score(labels, probs),
    }
    for k, v in metrics.items():
        print(f"{k:<12}: {v:.4f}")

    # Confusion matrix + ROC curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(labels, preds), display_labels=["Noise", "Earthquake"]
    ).plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion Matrix — Test Set")

    RocCurveDisplay.from_predictions(labels, probs, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "k--", label="Random")
    axes[1].set_title("ROC Curve — Test Set")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return metrics


def error_analysis(checkpoint_path: str, data_dir: str = "data") -> None:
    """Full FN/FP error analysis with waveform and spectrogram plots."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_test, X_wave, test_meta, labels = _load_test_arrays(data_dir)
    probs = _infer(checkpoint_path, X_test, device)
    preds = (probs >= 0.5).astype(int)

    tp = (labels == 1) & (preds == 1)
    tn = (labels == 0) & (preds == 0)
    fn = (labels == 1) & (preds == 0)   # missed earthquakes
    fp = (labels == 0) & (preds == 1)   # false alarms

    n_eq    = (labels == 1).sum()
    n_noise = (labels == 0).sum()
    print(f"Test earthquakes : {n_eq:,}")
    print(f"  Caught (TP)    : {tp.sum():,}")
    print(f"  Missed (FN)    : {fn.sum():,}  ({100*fn.sum()/n_eq:.2f}% miss rate)")
    print(f"\nTest noise       : {n_noise:,}")
    print(f"  Correct (TN)   : {tn.sum():,}")
    print(f"  False alarm(FP): {fp.sum():,}  ({100*fp.sum()/n_noise:.2f}% false alarm rate)")

    _plot_confidence_histograms(probs, fn, fp)
    _fn_metadata_analysis(test_meta, tp, fn)
    _fp_snr_analysis(test_meta, tn, fp)
    _plot_waveform_examples(X_wave, test_meta, probs, fn, "FN — Missed earthquakes", "steelblue")
    _plot_waveform_examples(X_wave, test_meta, probs, fp, "FP — False alarms", "darkorange")
    _plot_spectrogram_examples(X_wave, test_meta, probs, fn, "FN — Missed earthquakes", data_dir)
    _plot_spectrogram_examples(X_wave, test_meta, probs, fp, "FP — False alarms", data_dir)


# ── Private plot helpers ──────────────────────────────────────────────────────

def _plot_confidence_histograms(probs, fn, fp):
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(probs[fn], bins=50, color="steelblue")
    axes[0].axvline(0.5, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted probability (earthquake)")
    axes[0].set_title(f"FN — Missed earthquakes (n={fn.sum():,})")

    axes[1].hist(probs[fp], bins=50, color="darkorange")
    axes[1].axvline(0.5, color="red", linestyle="--")
    axes[1].set_xlabel("Predicted probability (earthquake)")
    axes[1].set_title(f"FP — False alarms (n={fp.sum():,})")

    plt.tight_layout()
    plt.show()


def _fn_metadata_analysis(test_meta, tp, fn):
    tp_meta = test_meta[tp]
    fn_meta = test_meta[fn]
    cols = [
        ("source_magnitude",   "Magnitude"),
        ("source_distance_km", "Distance (km)"),
        ("source_depth_km",    "Depth (km)"),
        ("snr_db",             "SNR (dB)"),
    ]

    print(f"\n{'Metric':<22}  {'Caught (TP)':>14}  {'Missed (FN)':>14}")
    print("-" * 55)
    valid_cols = []
    for col, label in cols:
        tp_vals = pd.to_numeric(tp_meta[col], errors="coerce").dropna()
        fn_vals = pd.to_numeric(fn_meta[col], errors="coerce").dropna()
        if len(fn_vals) == 0:
            continue
        print(f"{label:<22}  {tp_vals.mean():>8.3f} ± {tp_vals.std():.2f}  "
              f"{fn_vals.mean():>8.3f} ± {fn_vals.std():.2f}")
        valid_cols.append((col, label, tp_vals, fn_vals))

    fig, axes = plt.subplots(1, len(valid_cols), figsize=(4 * len(valid_cols), 4))
    fig.suptitle("Missed (FN) vs Caught (TP) — test set", fontsize=12)
    if len(valid_cols) == 1:
        axes = [axes]
    for ax, (_, label, tp_vals, fn_vals) in zip(axes, valid_cols):
        ax.boxplot([tp_vals, fn_vals], labels=["Caught", "Missed"], sym="")
        ax.set_title(label)
    plt.tight_layout()
    plt.show()


def _fp_snr_analysis(test_meta, tn, fp):
    fp_snr = pd.to_numeric(test_meta[fp]["snr_db"], errors="coerce").dropna()
    tn_snr = pd.to_numeric(test_meta[tn]["snr_db"], errors="coerce").dropna()
    print(f"\nSNR — False alarms (FP): mean={fp_snr.mean():.3f}  std={fp_snr.std():.3f}")
    print(f"SNR — Correct noise (TN): mean={tn_snr.mean():.3f}  std={tn_snr.std():.3f}")

    if len(fp_snr) > 0 and len(tn_snr) > 0:
        _, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot([tn_snr, fp_snr], labels=["Correct (TN)", "False alarm (FP)"], sym="")
        ax.set_title("SNR — false alarms vs correct noise")
        ax.set_ylabel("SNR (dB)")
        plt.tight_layout()
        plt.show()


def _top_n_by_confidence(probs, mask, n=5):
    """Indices of the n most confident errors (furthest from the 0.5 boundary)."""
    indices = np.where(mask)[0]
    order   = np.argsort(np.abs(probs[indices] - 0.5))[::-1]
    return indices[order[:n]]


def _plot_waveform_examples(X_wave, test_meta, probs, mask, title, color, n=5):
    top  = _top_n_by_confidence(probs, mask, n)
    time = np.arange(6000) / 100.0

    fig, axes = plt.subplots(len(top), 1, figsize=(13, 2.2 * len(top)), sharex=True)
    fig.suptitle(title, fontsize=12)
    if len(top) == 1:
        axes = [axes]
    for ax, idx in zip(axes, top):
        ax.plot(time, X_wave[idx], lw=0.5, color=color)
        row = test_meta.iloc[idx]
        p_a = row.get("p_arrival_sample")
        s_a = row.get("s_arrival_sample")
        if pd.notna(p_a): ax.axvline(float(p_a) / 100, color="red",   lw=1, ls="--", label="P")
        if pd.notna(s_a): ax.axvline(float(s_a) / 100, color="green", lw=1, ls="--", label="S")
        ax.set_title(
            f"prob={probs[idx]:.3f}  mag={row.get('source_magnitude', '?')}  "
            f"dist={row.get('source_distance_km', '?')} km",
            fontsize=9,
        )
        ax.set_ylabel("Amp")
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def _plot_spectrogram_examples(X_wave, test_meta, probs, mask, title, data_dir, n=5):
    top = _top_n_by_confidence(probs, mask, n)

    with open(f"{data_dir}/statistics.json") as f:
        stats = json.load(f)
    vmin = stats["spectrogram_db_range"]["p2"]
    vmax = stats["spectrogram_db_range"]["p98"]
    fs, nperseg, noverlap = 100, 256, 200

    fig, axes = plt.subplots(len(top), 1, figsize=(13, 3.2 * len(top)))
    fig.suptitle(f"Spectrograms — {title}  (most confident, n={len(top)})", fontsize=12)
    if len(top) == 1:
        axes = [axes]
    for ax, idx in zip(axes, top):
        f_a, t_a, Sxx = _spectrogram(X_wave[idx], fs=fs, nperseg=nperseg, noverlap=noverlap)
        mesh = ax.pcolormesh(
            t_a, f_a, 10 * np.log10(Sxx + 1e-10),
            shading="gouraud", cmap="inferno", vmin=vmin, vmax=vmax,
        )
        plt.colorbar(mesh, ax=ax, label="Power (dB)")
        row = test_meta.iloc[idx]
        p_a = row.get("p_arrival_sample")
        s_a = row.get("s_arrival_sample")
        if pd.notna(p_a): ax.axvline(float(p_a) / fs, color="cyan", lw=1.5, ls="--", label="P")
        if pd.notna(s_a): ax.axvline(float(s_a) / fs, color="lime", lw=1.5, ls="--", label="S")
        ax.set_title(
            f"prob={probs[idx]:.3f}  mag={row.get('source_magnitude', '?')}  "
            f"dist={row.get('source_distance_km', '?')} km",
            fontsize=9,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.show()