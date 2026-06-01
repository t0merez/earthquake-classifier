import numpy as np
import pandas as pd
import json
from pathlib import Path


def _describe(series):
    """Return summary statistics dict for a numeric series, handling NaN / mixed types."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return None
    return {
        "count":  int(len(s)),
        "mean":   round(float(s.mean()),            4),
        "std":    round(float(s.std()),             4),
        "min":    round(float(s.min()),             4),
        "p25":    round(float(s.quantile(0.25)),    4),
        "median": round(float(s.median()),          4),
        "p75":    round(float(s.quantile(0.75)),    4),
        "max":    round(float(s.max()),             4),
    }


def compute_statistics(data_dir="data"):
    data_dir = Path(data_dir)

    print("Loading data...")
    X    = np.load(data_dir / "waveforms.npy")
    meta = pd.read_csv(data_dir / "metadata.csv", low_memory=False)

    eq    = meta[meta["label"] == 1]
    noise = meta[meta["label"] == 0]

    stats = {}

    # ── Overall counts ──────────────────────────────────────────────────────
    stats["total_traces"]        = int(len(meta))
    stats["earthquakes"]         = int(len(eq))
    stats["noise"]               = int(len(noise))
    stats["earthquake_fraction"] = round(float(len(eq) / len(meta)), 4)

    # ── Split breakdown ─────────────────────────────────────────────────────
    stats["splits"] = {}
    for split in ["train", "val", "test"]:
        mask = meta["split"] == split
        sub  = meta[mask]
        stats["splits"][split] = {
            "total":       int(mask.sum()),
            "earthquakes": int((sub["label"] == 1).sum()),
            "noise":       int((sub["label"] == 0).sum()),
        }

    # ── Earthquake metadata distributions ───────────────────────────────────
    stats["magnitude"]   = _describe(eq["source_magnitude"])
    stats["distance_km"] = _describe(eq["source_distance_km"])
    stats["depth_km"]    = _describe(eq["source_depth_km"])

    # ── SNR (all traces that have a value) ───────────────────────────────────
    stats["snr_db"] = _describe(meta["snr_db"])

    # ── Unique seismic events ────────────────────────────────────────────────
    stats["unique_earthquake_events"] = int(
        meta.loc[meta["label"] == 1, "source_id"].dropna().nunique()
    )

    # ── Waveform array info ──────────────────────────────────────────────────
    stats["waveform_shape"]   = list(X.shape)
    stats["sample_rate_hz"]   = 100
    stats["trace_duration_s"] = int(X.shape[1]) // 100

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = data_dir / "statistics.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved: {out_path}\n")
    print(f"  Total traces      : {stats['total_traces']:,}")
    print(f"  Earthquakes       : {stats['earthquakes']:,}  ({stats['earthquake_fraction']*100:.1f}%)")
    print(f"  Noise             : {stats['noise']:,}")
    print(f"  Unique EQ events  : {stats['unique_earthquake_events']:,}")
    print()
    for split, v in stats["splits"].items():
        print(f"  {split:5s}  total={v['total']:,}  eq={v['earthquakes']:,}  noise={v['noise']:,}")
    print()
    for key in ("magnitude", "distance_km", "depth_km", "snr_db"):
        v = stats[key]
        if v:
            print(f"  {key:15s}  mean={v['mean']:.2f}  std={v['std']:.2f}"
                  f"  [{v['min']:.2f}, {v['max']:.2f}]")

    return stats


if __name__ == "__main__":
    compute_statistics()
