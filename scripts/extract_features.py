import os
import sys
import glob
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import matplotlib
matplotlib.use("Agg")
from audio_features import load_audio, extract_mel_spectrogram, extract_chroma, normalize_features
from feature_extraction import segment_features

AUDIO_DIR = "data/raw/fma_medium"
OUT_DIR = "data/processed/audio_features"


def process(path):
    tid = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(OUT_DIR, tid + ".npz")
    if os.path.exists(out):
        return tid, "cached"
    try:
        y, sr = load_audio(path, duration=30)
        if y.size < sr:
            return tid, "too_short"
        mel = segment_features(normalize_features(extract_mel_spectrogram(y, sr)), 6)
        chroma = segment_features(normalize_features(extract_chroma(y, sr)), 6)
        np.savez(out, mel=mel.astype(np.float32), chroma=chroma.astype(np.float32),
                 sample_rate=22050)
        return tid, "ok"
    except Exception as exc:
        return tid, f"error: {type(exc).__name__}"


def main(workers=30, limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(AUDIO_DIR, "**", "*.mp3"), recursive=True))
    if limit:
        files = files[:limit]
    print(f"{len(files)} mp3 files, {workers} workers", flush=True)

    t0 = time.time()
    failed, done = [], 0
    with Pool(workers) as pool:
        for tid, status in pool.imap_unordered(process, files, chunksize=16):
            done += 1
            if status not in ("ok", "cached"):
                failed.append((tid, status))
            if done % 1000 == 0:
                rate = done / (time.time() - t0)
                print(f"{done}/{len(files)}  {rate:.1f}/s  eta {(len(files)-done)/rate/60:.1f} min",
                      flush=True)

    print(f"done in {(time.time()-t0)/60:.1f} min, {len(failed)} failed")
    with open("results/extraction_failures.txt", "w") as f:
        for tid, status in failed:
            f.write(f"{tid}\t{status}\n")


if __name__ == "__main__":
    main(workers=int(sys.argv[1]) if len(sys.argv) > 1 else 30,
         limit=int(sys.argv[2]) if len(sys.argv) > 2 else None)
