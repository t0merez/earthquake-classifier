"""Dataset for P-wave arrival time regression."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PickerDataset(Dataset):
    """Earthquake-only waveforms paired with their P-wave arrival sample index.

    Filters to rows where:
      - split matches the requested split ("train", "val", or "test")
      - label == 1 (earthquake, not noise)
      - p_arrival_sample is not NaN (arrival was annotated)

    Args:
        split:    one of "train", "val", "test"
        data_dir: directory containing waveforms.npy and metadata.csv
    """

    def __init__(self, split: str, data_dir: str = "data"):
        meta = pd.read_csv(f"{data_dir}/metadata.csv", low_memory=False)

        mask = (
            (meta["split"] == split) &
            (meta["label"] == 1) &
            (meta["p_arrival_sample"].notna())
        )

        indices       = np.where(mask)[0]
        self.arrivals = meta.loc[mask, "p_arrival_sample"].values.astype(np.float32)

        # Load only the earthquake rows into RAM upfront.
        # Full waveforms.npy is ~9 GB (427K traces); the ~116K earthquake rows
        # are ~2.8 GB — fits in RAM and eliminates per-sample disk I/O during training.
        all_waveforms  = np.load(f"{data_dir}/waveforms.npy", mmap_mode="r")
        self.waveforms = np.array(all_waveforms[indices], dtype=np.float32)  # copies to RAM

    def __len__(self) -> int:
        return len(self.arrivals)

    def __getitem__(self, idx: int):
        """Return (waveform, arrival) where:
            waveform: float32 tensor of shape (1, 6000)  — the z-scored Z-channel
            arrival:  float32 scalar                     — p_arrival_sample index
        """
        waveform = torch.from_numpy(self.waveforms[idx][np.newaxis, :])
        arrival  = torch.tensor(self.arrivals[idx])
        return waveform, arrival