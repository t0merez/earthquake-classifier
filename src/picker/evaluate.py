"""Evaluation for the P-wave arrival picker."""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.picker.dataset import PickerDataset
from src.picker.model import PickerCNN


def evaluate(checkpoint_path: str, data_dir: str = "data", n_examples: int = 5) -> float:
    """Load the model and test set once, then print MAE, plot residuals, and plot examples.

    Returns:
        MAE in seconds.
    """
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = PickerCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Load test set into RAM once
    print("Loading test set...")
    test_ds = PickerDataset("test", data_dir)
    loader  = DataLoader(test_ds, batch_size=512, shuffle=False)

    # Run inference once, reuse results for all plots
    preds = []
    with torch.no_grad():
        for X, _ in loader:
            preds.append(model(X.to(device)).cpu())
    preds = torch.cat(preds).numpy()
    true  = test_ds.arrivals

    # ---- MAE ----
    mae = float(np.abs(preds - true).mean())
    print(f"Test MAE: {mae:.1f} samples  ({mae/100:.4f} s)")

    # ---- Residual histogram ----
    residuals = preds - true   # positive = predicted too late
    _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=100, range=(-200, 200), color="steelblue", edgecolor="none")
    ax.axvline(0, color="red", linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Residual (samples)   [positive = predicted late]")
    ax.set_ylabel("Count")
    ax.set_title(f"Residuals — test set  (MAE = {mae:.1f} samples)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ---- Example waveforms ----
    indices = np.random.choice(len(test_ds), n_examples, replace=False)
    time    = np.arange(6000) / 100.0

    fig, axes = plt.subplots(n_examples, 1, figsize=(13, 2.5 * n_examples), sharex=True)
    fig.suptitle("P-wave picker — test set examples", fontsize=12)
    if n_examples == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        ax.plot(time, test_ds.waveforms[idx], lw=0.5, color="steelblue")
        ax.axvline(true[idx]  / 100, color="green", lw=1.5, ls="--", label="True")
        ax.axvline(preds[idx] / 100, color="red",   lw=1.5, ls="--", label="Predicted")
        ax.set_title(f"residual = {(preds[idx] - true[idx])/100:+.3f} s", fontsize=9)
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

    return mae / 100