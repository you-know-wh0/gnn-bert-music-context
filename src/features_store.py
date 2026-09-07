import json
import os

import numpy as np

STORE = "data/processed/store"
MEL_PATH = f"{STORE}/mel_f16.npy"
CHROMA_PATH = f"{STORE}/chroma_f16.npy"
IDS_PATH = f"{STORE}/track_ids.json"
SEGS, MELS, CHROMA, FRAMES = 6, 128, 12, 215

_cache = {}


FAILURES = "results/extraction_failures.json"


NPZ_DIR = "data/processed/audio_features"


def exists():
    return os.path.exists(MEL_PATH) and os.path.exists(IDS_PATH)


def load(mode="r"):
    if "ids" not in _cache:
        bad = set(json.load(open(FAILURES))) if os.path.exists(FAILURES) else set()
        if exists():
            ids = json.load(open(IDS_PATH))
            _cache["ids"] = ids
            _cache["pos"] = {t: i for i, t in enumerate(ids)}
            _cache["usable"] = [t for t in ids if t not in bad]
            _cache["mel"] = np.load(MEL_PATH, mmap_mode=mode)
            _cache["chroma"] = np.load(CHROMA_PATH, mmap_mode=mode)
        elif os.path.isdir(NPZ_DIR):
            files = [f[:-4] for f in os.listdir(NPZ_DIR) if f.endswith(".npz")]
            files.sort()
            _cache["ids"] = files
            _cache["pos"] = {t: i for i, t in enumerate(files)}
            _cache["usable"] = [t for t in files if t not in bad]
            _cache["mel"] = None
            _cache["chroma"] = None
        else:
            _cache["ids"] = []
            _cache["pos"] = {}
            _cache["usable"] = []
            _cache["mel"] = None
            _cache["chroma"] = None
    return _cache


def track_ids():
    return load()["ids"]


def usable_ids():
    """Track ids whose audio decoded successfully (excludes FMA's corrupt files)."""
    return load()["usable"]


def has(tid):
    if "usable_set" not in _cache:
        _cache["usable_set"] = set(load()["usable"])
    return tid in _cache["usable_set"]


def get(tid):
    s = load()
    if s["mel"] is not None:
        i = s["pos"][tid]
        return s["mel"][i].astype(np.float32), s["chroma"][i].astype(np.float32)
    npz_file = os.path.join(NPZ_DIR, f"{tid}.npz")
    with np.load(npz_file) as d:
        mel = d["mel"].astype(np.float32)
        chroma = d["chroma"].astype(np.float32)
    return mel, chroma


def stitch(tid):
    """Return mel (128, 1290) and chroma (12, 1290) with segments concatenated in time."""
    mel, chroma = get(tid)
    return (np.concatenate(list(mel), axis=-1), np.concatenate(list(chroma), axis=-1))

