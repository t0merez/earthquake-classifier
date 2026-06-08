import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import spectrogram as _spectrogram

FS       = 100
NPERSEG  = 256
NOVERLAP = 200


def plot_spectrogram(trace_id, data_dir="data", ax=None):
    """
    Plot a full-resolution spectrogram for the given trace_name.

    The dB color scale is anchored to the global p2/p98 range stored in
    statistics.json, so the same color always represents the same power
    level across any two calls.  Run data_code/statistics.py first to
    populate that range.

    Parameters
    ----------
    trace_id : str
        Value of the trace_name column in metadata.csv.
    data_dir : str or Path
        Directory containing waveforms.npy, metadata.csv, statistics.json.
    ax : matplotlib Axes, optional
        Axes to draw on.  If None, a new figure is created and shown.

    Returns
    -------
    ax : matplotlib Axes
    """
    data_dir = Path(data_dir)

    meta    = pd.read_csv(data_dir / "metadata.csv", low_memory=False)
    matches = meta[meta["trace_name"] == trace_id]
    if len(matches) == 0:
        raise ValueError(f"trace_name '{trace_id}' not found in metadata")

    row      = matches.iloc[0]
    iloc_idx = matches.index[0]

    X        = np.load(data_dir / "waveforms.npy", mmap_mode="r")
    waveform = np.array(X[iloc_idx])

    stats_path = data_dir / "statistics.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"{stats_path} not found — run data_code/statistics.py first."
        )
    with open(stats_path) as f:
        stats = json.load(f)
    if "spectrogram_db_range" not in stats:
        raise KeyError(
            "'spectrogram_db_range' missing from statistics.json — "
            "re-run data_code/statistics.py with spectrograms present."
        )
    vmin = stats["spectrogram_db_range"]["p2"]
    vmax = stats["spectrogram_db_range"]["p98"]

    f_arr, t_arr, Sxx = _spectrogram(waveform, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    S_db = 10 * np.log10(Sxx + 1e-10)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(13, 5))

    mesh = ax.pcolormesh(t_arr, f_arr, S_db, shading="gouraud",
                         cmap="inferno", vmin=vmin, vmax=vmax)
    plt.colorbar(mesh, ax=ax, label="Power (dB)")

    kind = "Earthquake" if row["label"] == 1 else "Noise"
    ax.set_title(f"Spectrogram of {kind}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a full-resolution spectrogram for a given trace."
    )
    parser.add_argument("trace_id", help="trace_name identifier from metadata.csv")
    parser.add_argument("--data-dir", default="data",
                        help="path to the data directory (default: data)")
    args = parser.parse_args()
    plot_spectrogram(args.trace_id, data_dir=args.data_dir)
