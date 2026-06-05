"""Peak amplitude threshold baseline classifier."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
)


def _load(data_dir: str):
    X      = np.load(f"{data_dir}/waveforms.npy")
    meta   = pd.read_csv(f"{data_dir}/metadata.csv", low_memory=False)
    labels = meta["label"].values
    splits = meta["split"].values
    return X, labels, splits


def plot_feature_distributions(data_dir: str = "data") -> None:
    """Histogram of max|x| and kurtosis for earthquakes vs noise."""
    X, labels, _ = _load(data_dir)
    max_abs = np.abs(X).max(axis=1)
    kurt    = kurtosis(X, axis=1)

    _, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, feat, name, xlim in [
        (axes[0], max_abs, "max |x|  (peak deviation, σ)", (0, 30)),
        (axes[1], kurt,    "Excess kurtosis",               (-5, 50)),
    ]:
        ax.hist(feat[labels == 0], bins=200, range=xlim, alpha=0.6, color="darkorange", label="Noise",      density=True)
        ax.hist(feat[labels == 1], bins=200, range=xlim, alpha=0.6, color="steelblue",  label="Earthquake", density=True)
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.legend()
    plt.tight_layout()
    plt.show()


def find_threshold(data_dir: str = "data") -> float:
    """Sweep 1000 thresholds on the training set and return the one with the highest F1.

    Also plots the F1 vs threshold curve.
    """
    X, labels, splits = _load(data_dir)
    max_abs      = np.abs(X).max(axis=1)
    train_mask   = splits == "train"
    train_feat   = max_abs[train_mask]
    train_labels = labels[train_mask]

    thresholds = np.linspace(train_feat.min(), train_feat.max(), 1000)
    f1_scores  = [
        f1_score(train_labels, (train_feat > t).astype(int), zero_division=0)
        for t in thresholds
    ]

    best_idx       = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    print(f"Best threshold : {best_threshold:.3f} σ")
    print(f"Train F1       : {f1_scores[best_idx]:.4f}")

    plt.figure(figsize=(8, 3))
    plt.plot(thresholds, f1_scores)
    plt.axvline(best_threshold, color="red", linestyle="--", label=f"Best = {best_threshold:.2f} σ")
    plt.xlabel("Threshold (σ)")
    plt.ylabel("F1 (train)")
    plt.title("F1 vs threshold on training set")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return best_threshold


def evaluate(threshold: float, data_dir: str = "data") -> dict:
    """Evaluate the threshold classifier on the test set.

    Returns:
        dict with keys: f1, accuracy, precision, recall, auc_roc
    """
    X, labels, splits = _load(data_dir)
    max_abs    = np.abs(X).max(axis=1)
    test_mask  = splits == "test"
    test_feat  = max_abs[test_mask]
    test_labels = labels[test_mask]
    test_preds  = (test_feat > threshold).astype(int)

    print("=" * 50)
    print("Test set metrics — classical baseline")
    print("=" * 50)
    print(classification_report(test_labels, test_preds, target_names=["Noise", "Earthquake"], digits=4))

    metrics = {
        "f1":        f1_score(test_labels, test_preds, average="macro"),
        "accuracy":  accuracy_score(test_labels, test_preds),
        "precision": precision_score(test_labels, test_preds),
        "recall":    recall_score(test_labels, test_preds),
        "auc_roc":   roc_auc_score(test_labels, test_feat),
    }
    for k, v in metrics.items():
        print(f"{k:<12}: {v:.4f}")

    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(test_labels, test_preds), display_labels=["Noise", "Earthquake"]
    ).plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion Matrix — Test Set")

    RocCurveDisplay.from_predictions(test_labels, test_feat, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "k--", label="Random")
    axes[1].set_title("ROC Curve — Test Set")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return metrics