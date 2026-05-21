import h5py
import numpy as np
import pandas as pd
from pathlib import Path


def _assign_splits(meta, train_frac=0.60, val_frac=0.15, seed=42):
    """
    Returns a split label array ("train"/"val"/"test") for each row in meta.

    Earthquake traces are grouped by source_id so every trace from the same
    event lands in the same split.  Noise traces have no shared source event
    so they are split at the trace level.
    """
    rng = np.random.default_rng(seed)
    splits = np.empty(len(meta), dtype=object)

    # --- earthquakes: event-level split ---
    eq_mask = (meta["label"] == 1).values
    if eq_mask.any():
        events = meta.loc[eq_mask, "source_id"].dropna().unique()
        rng.shuffle(events)
        n = len(events)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        event_split = {}
        for i, e in enumerate(events):
            if i < n_train:
                event_split[e] = "train"
            elif i < n_train + n_val:
                event_split[e] = "val"
            else:
                event_split[e] = "test"
        eq_indices = np.where(eq_mask)[0]
        for idx in eq_indices:
            sid = meta.iloc[idx]["source_id"]
            splits[idx] = event_split.get(sid, "train")

    # --- noise: trace-level split ---
    noise_indices = np.where(~eq_mask)[0]
    rng.shuffle(noise_indices)
    n = len(noise_indices)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    for rank, idx in enumerate(noise_indices):
        if rank < n_train:
            splits[idx] = "train"
        elif rank < n_train + n_val:
            splits[idx] = "val"
        else:
            splits[idx] = "test"

    return splits

METADATA_COLS = [
    "trace_name", "trace_category", "source_magnitude", "source_distance_km",
    "source_depth_km", "p_arrival_sample", "s_arrival_sample",
    "snr_db", "receiver_code", "source_id",
]


def build_dataset(chunk_dirs, out_dir="data"):
    """
    Process one or more STEAD chunk directories and save two files to out_dir:

        waveforms.npy   — float32 (N, 6000)  z-score normalised Z-channel
        metadata.csv    — N rows, one per waveform, with label + metadata

    Row i in waveforms.npy corresponds exactly to row i in metadata.csv.
    label = 1 for earthquake, 0 for noise.
    """
    Path(out_dir).mkdir(exist_ok=True)
    all_waves, all_meta = [], []

    for chunk_dir in chunk_dirs:
        chunk_dir = Path(chunk_dir)
        hdf5 = next(chunk_dir.glob("*.hdf5"))
        csv  = next(chunk_dir.glob("*.csv"))

        print(f"Reading {csv.name} ...")
        meta = pd.read_csv(csv, low_memory=False)
        meta = meta[[c for c in METADATA_COLS if c in meta.columns]]

        names = meta["trace_name"].tolist()
        n     = len(names)

        # Pre-allocate the output array (avoids doubling RAM with list + stack)
        waves  = np.zeros((n, 6000), dtype=np.float32)
        kept   = []

        print(f"Loading waveforms from {hdf5.name} ...")
        with h5py.File(hdf5, "r") as hf:
            root = hf.get("data", hf)
            for i, name in enumerate(names):
                if i % 5000 == 0:
                    print(f"  {i:,} / {n:,}", end="\r", flush=True)

                if name not in root:
                    continue
                w = root[name][:]
                if w.shape != (6000, 3):
                    continue

                z = w[:, 2].astype(np.float32)   # Z channel

                if not np.all(np.isfinite(z)):    continue  # NaN / Inf
                if z.var() == 0:                  continue  # flat / dead
                if z.max() >= 32767 or z.min() <= -32768:   continue  # clipped

                std = z.std()
                waves[len(kept)] = (z - z.mean()) / std    # z-score in-place

                kept.append(i)

        n_kept = len(kept)
        print(f"  kept {n_kept:,} / {n:,} traces          ")

        all_waves.append(waves[:n_kept])
        all_meta.append(meta.iloc[kept].reset_index(drop=True))

    # Combine both chunks
    X    = np.concatenate(all_waves, axis=0)
    meta = pd.concat(all_meta, ignore_index=True)
    meta["label"] = (meta["trace_category"] == "earthquake_local").astype("int8")
    meta["split"] = _assign_splits(meta)

    # Save
    np.save(f"{out_dir}/waveforms.npy", X)
    meta.to_csv(f"{out_dir}/metadata.csv", index=False)

    print(f"\nSaved {X.shape[0]:,} traces to '{out_dir}/'")
    print(f"  Earthquakes : {(meta['label']==1).sum():,}")
    print(f"  Noise       : {(meta['label']==0).sum():,}")
