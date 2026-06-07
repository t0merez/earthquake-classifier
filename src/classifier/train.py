"""Training loop for the spectrogram CNN classifier."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.classifier.model import SpectrogramCNN


class SpectrogramDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def _load_data(data_dir: str, batch_size: int, device: str):
    # Add channel dim: (N, 10, 10) → (N, 1, 10, 10) as required by Conv2d
    X = np.load(f"{data_dir}/spectrograms/spectrograms_10x10.npy")[:, np.newaxis, :, :]
    meta = pd.read_csv(f"{data_dir}/metadata.csv", low_memory=False)

    # Normalise using training-set statistics only to avoid data leakage
    train_mask = (meta["split"] == "train").values
    mu = float(X[train_mask].mean())
    std = float(X[train_mask].std())
    X = ((X - mu) / (std + 1e-8)).astype(np.float32)

    labels = meta["label"].values.astype(np.float32)

    def make_loader(split, shuffle=False):
        mask = (meta["split"] == split).values
        return DataLoader(
            SpectrogramDataset(X[mask], labels[mask]),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(device == "cuda"),
        )

    return make_loader("train", shuffle=True), make_loader("val"), make_loader("test")


def train(data_dir: str = "data", checkpoint_path: str = "checkpoints/classifier_best.pt") -> dict:
    """Train the CNN classifier and save the best checkpoint by validation F1.

    Args:
        config: optional overrides. Keys:
            data_dir        (str,   default "data")
            checkpoint_path (str,   default "checkpoints/classifier_best.pt")
            batch_size      (int,   default 256)
            epochs          (int,   default 30)
            lr              (float, default 1e-3)
            patience        (int,   default 5)
            device          (str,   default "cuda" if available else "cpu")

    Returns:
        history dict with keys "train_loss", "val_loss", "val_f1".
    """
    cfg = {
        "data_dir": data_dir,
        "checkpoint_path": checkpoint_path,
        "batch_size": 256,
        "epochs": 30,
        "lr": 1e-3,
        "patience": 5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    device = cfg["device"]

    train_loader, val_loader, _ = _load_data(cfg["data_dir"], cfg["batch_size"], device)

    model = SpectrogramCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    from sklearn.metrics import f1_score
    import os

    os.makedirs(os.path.dirname(cfg["checkpoint_path"]) or ".", exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_f1 = 0.0
    patience_counter = 0

    try:
        for epoch in range(0, cfg["epochs"]):
            # ---- Train ----
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y_batch)
            train_loss /= len(train_loader.dataset)

            # ---- Validate ----
            model.eval()
            val_loss = 0.0
            val_probs, val_true = [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = model(X_batch)
                    val_loss += criterion(logits, y_batch).item() * len(y_batch)
                    val_probs.append(torch.sigmoid(logits).cpu().numpy())
                    val_true.append(y_batch.cpu().numpy())
            val_loss /= len(val_loader.dataset)

            val_probs = np.concatenate(val_probs)
            val_true = np.concatenate(val_true)
            val_f1 = f1_score(val_true, (val_probs >= 0.5).astype(int), zero_division=0)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)

            # Halve LR when val_loss plateaus for 3 epochs
            scheduler.step(val_loss)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), cfg["checkpoint_path"])
            else:
                patience_counter += 1

            print(
                f"Epoch {epoch:2d}/{cfg['epochs']}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_f1={val_f1:.4f}"
            )

            if patience_counter >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch}.")
                break

    except KeyboardInterrupt:
        print(f"\nTraining interrupted at epoch {epoch}.")

    print(f"Best val F1: {best_val_f1:.4f}  —  checkpoint saved to {cfg['checkpoint_path']}")
    return history


if __name__ == "__main__":
    train()