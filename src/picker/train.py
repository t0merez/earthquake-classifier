"""Training loop for the P-wave arrival regression CNN."""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.picker.dataset import PickerDataset
from src.picker.model import PickerCNN


def train(config: dict = None) -> dict:
    """Train the picker and save the best checkpoint by validation MAE.

    Args:
        config: optional overrides. Keys:
            data_dir        (str,   default "data")
            checkpoint_path (str,   default "checkpoints/picker_best.pt")
            batch_size      (int,   default 256)
            epochs          (int,   default 30)
            lr              (float, default 1e-3)
            patience        (int,   default 7)
            num_workers     (int,   default 4)
            device          (str,   default "cuda" if available else "cpu")

    Returns:
        history dict with keys "train_loss" and "val_mae" (both in seconds).
    """
    cfg = {
        "data_dir":        "data",
        "checkpoint_path": "checkpoints/picker_best.pt",
        "batch_size":      256,
        "epochs":          30,
        "lr":              1e-3,
        "patience":        7,
        "num_workers":     4,
        "device":          "cuda" if torch.cuda.is_available() else "cpu",
    }
    if config:
        cfg.update(config)

    device = cfg["device"]
    print(f"Using device: {device}")

    os.makedirs(os.path.dirname(cfg["checkpoint_path"]) or ".", exist_ok=True)

    # Dataset init reads ~2.8 GB of earthquake waveforms into RAM once
    print("Loading datasets into RAM...")
    train_ds = PickerDataset("train", cfg["data_dir"])
    val_ds   = PickerDataset("val",   cfg["data_dir"])
    print(f"Train: {len(train_ds):,} traces  |  Val: {len(val_ds):,} traces")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=(device == "cuda"),
    )

    model     = PickerCNN().to(device)
    criterion = nn.L1Loss()   # MAE loss — directly matches the evaluation metric
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # OneCycleLR ramps LR up then down in one sweep; typically converges
    # in fewer epochs than ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"],
        epochs=cfg["epochs"], steps_per_epoch=len(train_loader),
    )

    # GradScaler speeds up training on CUDA via float16; no-op on CPU
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    history          = {"train_loss": [], "val_mae": []}
    best_val_mae     = float("inf")
    patience_counter = 0

    try:
        for epoch in range(1, cfg["epochs"] + 1):
            # ---- Train ----
            model.train()
            train_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.autocast(device_type=device, enabled=(device == "cuda")):
                    loss = criterion(model(X), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()   # OneCycleLR steps every batch, not every epoch
                train_loss += loss.item() * len(y)
            train_loss /= len(train_loader.dataset)

            # ---- Validate ----
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    preds.append(model(X.to(device)).cpu())
                    trues.append(y)
            val_mae = (torch.cat(preds) - torch.cat(trues)).abs().mean().item()

            # Store in seconds for readability
            history["train_loss"].append(train_loss / 100)
            history["val_mae"].append(val_mae / 100)

            print(
                f"Epoch {epoch:2d}/{cfg['epochs']}  "
                f"train_loss={train_loss/100:.4f} s  "
                f"val_MAE={val_mae/100:.4f} s"
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), cfg["checkpoint_path"])
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break

    except KeyboardInterrupt:
        print(f"\nTraining interrupted at epoch {epoch}.")

    print(f"Best val MAE: {best_val_mae/100:.4f} s  —  checkpoint saved to {cfg['checkpoint_path']}")
    return history


if __name__ == "__main__":
    train()