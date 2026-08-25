import ast
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

TRACKS_CSV = "data/raw/metadata/fma_metadata/tracks.csv"
GENRES_CSV = "data/raw/metadata/fma_metadata/genres.csv"
SPLITS_DIR = "data/splits"
SPLIT_FILES = {"training": "training.json", "validation": "validation.json", "test": "test.json"}


def read_tracks(tracks_csv=TRACKS_CSV):
    reader = csv.reader(open(tracks_csv, encoding="utf-8"))
    h0, h1 = next(reader), next(reader)
    next(reader)
    col = {f"{a}.{b}": i for i, (a, b) in enumerate(zip(h0, h1))}
    out = {}
    for row in reader:
        if not row or row[col["set.subset"]] not in ("small", "medium"):
            continue
        out[row[0].strip().zfill(6)] = {
            "genre_top": row[col["track.genre_top"]].strip(),
            "genres_all": ast.literal_eval(row[col["track.genres_all"]] or "[]"),
            "artist_id": row[col["artist.id"]].strip(),
            "split": row[col["set.split"]].strip(),
        }
    return out


def genre_names(genres_csv=GENRES_CSV):
    return {int(r["genre_id"]): r["title"] for r in csv.DictReader(open(genres_csv))}


def load_split(name, splits_dir=SPLITS_DIR):
    return json.load(open(Path(splits_dir) / SPLIT_FILES[name]))


def augment_splits(splits_dir=SPLITS_DIR, tracks_csv=TRACKS_CSV):
    meta = read_tracks(tracks_csv)
    for name, fname in SPLIT_FILES.items():
        path = Path(splits_dir) / fname
        split = json.load(open(path))
        for tid, info in split.items():
            m = meta.get(tid, {})
            info["genres_all"] = m.get("genres_all", [])
            info["artist_id"] = m.get("artist_id", "")
        json.dump(split, open(path, "w"), indent=2)
        print(f"augmented {name}: {len(split)} tracks")


def check_artist_leakage(splits_dir=SPLITS_DIR):
    art = {n: {i["artist_id"] for i in load_split(n, splits_dir).values()} for n in SPLIT_FILES}
    return {
        "train_val": len(art["training"] & art["validation"]),
        "train_test": len(art["training"] & art["test"]),
        "val_test": len(art["validation"] & art["test"]),
    }


def build_vocab(mode, top_k=20, splits_dir=SPLITS_DIR):
    train = load_split("training", splits_dir)
    if mode == "top":
        return sorted({i["genre"] for i in train.values() if i["genre"]})
    names = genre_names()
    freq = Counter(g for i in train.values() for g in i["genres_all"])
    return [names.get(g, str(g)) for g, _ in freq.most_common(top_k)]


def encode_labels(split, vocab, mode):
    index = {g: i for i, g in enumerate(vocab)}
    names = genre_names() if mode == "multi" else None
    ids, y = [], []
    for tid, info in split.items():
        if mode == "top":
            if info["genre"] not in index:
                continue
            vec = index[info["genre"]]
        else:
            vec = np.zeros(len(vocab), dtype=np.float32)
            for g in info["genres_all"]:
                j = index.get(names.get(g, str(g)))
                if j is not None:
                    vec[j] = 1.0
            if vec.sum() == 0:
                continue
        ids.append(tid)
        y.append(vec)
    return ids, np.array(y)


if __name__ == "__main__":
    augment_splits()
    print("artist leakage:", check_artist_leakage())
    for mode in ("top", "multi"):
        vocab = build_vocab(mode)
        ids, y = encode_labels(load_split("training"), vocab, mode)
        print(f"{mode}: {len(vocab)} classes, {len(ids)} tracks, y={y.shape}")
