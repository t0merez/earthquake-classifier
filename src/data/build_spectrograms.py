import argparse
from pathlib import Path
import numpy as np
from scipy.signal import spectrogram

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=10, help="Output grid size (n x n patches)")
args = parser.parse_args()
n = args.n

X = np.load("data/waveforms.npy")
N = len(X)
print(f"Loaded waveforms: {X.shape}")
print(f"Output size: {n} x {n}")

FS       = 100
NPERSEG  = 256
NOVERLAP = 200

# Trim raw spectrogram dims to nearest multiple of n
FREQ_KEEP = (128 // n) * n
TIME_KEEP = (96  // n) * n
PATCH_F   = FREQ_KEEP // n
PATCH_T   = TIME_KEEP // n

X_spec = np.empty((N, n, n), dtype=np.float32)
for i in range(N):
    if i % 5000 == 0:
        print(f"\r{i:,} / {N:,}", end="", flush=True)
    _, _, Sxx = spectrogram(X[i], fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    S = 10 * np.log10(Sxx + 1e-10)
    S = S[:FREQ_KEEP, :TIME_KEEP]
    S = S.reshape(n, PATCH_F, TIME_KEEP).mean(axis=1)
    S = S.reshape(n, n, PATCH_T).mean(axis=2)
    X_spec[i] = S

out_dir = Path("data/spectrograms")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"spectrograms_{n}x{n}.npy"

np.save(out_path, X_spec)
print(f"\nSaved {out_path}  shape={X_spec.shape}  dtype={X_spec.dtype}")
print(f"Range: [{X_spec.min():.3f}, {X_spec.max():.3f}]")
