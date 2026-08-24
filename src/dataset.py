import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
import graph_builder as gb
from labels import build_vocab, encode_labels, load_split

FEATURE_DIR = "data/processed/audio_features"
CACHE_DIR = "data/processed/cache"
_CFG = {}


def _seg_worker(tid):
    mel, chroma = gb.stitch(f"{FEATURE_DIR}/{tid}.npz")
    return gb.node_features(mel, chroma, _CFG["n_nodes"], _CFG["feat"], _CFG["with_std"])


def _chord_worker(tid):
    _, chroma = gb.stitch(f"{FEATURE_DIR}/{tid}.npz")
    x, ei, ea = gb.chord_graph(chroma)
    return x, ei.numpy(), ea.numpy()


def _init(cfg):
    global _CFG
    _CFG = cfg


def build_cache(graph="segment", n_nodes=30, feat="mel+chroma", with_std=False, workers=24):
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    tag = "chord" if graph == "chord" else f"seg_{n_nodes}_{feat.replace('+','')}_{int(with_std)}"
    path = f"{CACHE_DIR}/{tag}.pt"
    if os.path.exists(path):
        return path

    ids = sorted(f[:-4] for f in os.listdir(FEATURE_DIR) if f.endswith(".npz"))
    cfg = dict(n_nodes=n_nodes, feat=feat, with_std=with_std)
    fn = _chord_worker if graph == "chord" else _seg_worker
    print(f"caching {tag}: {len(ids)} tracks", flush=True)

    with Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        out = pool.map(fn, ids, chunksize=32)

    if graph == "chord":
        store = {"ids": ids, "graphs": out}
    else:
        store = {"ids": ids, "x": np.stack(out)}
    torch.save(store, path)
    print(f"saved {path}", flush=True)
    return path


def load_cache(path):
    return torch.load(path, weights_only=False)


def fit_scaler(x):
    flat = x.reshape(-1, x.shape[-1])
    return flat.mean(0), flat.std(0) + 1e-6


def make_datasets(labels="multi", top_k=20, graph="segment", n_nodes=30, feat="mel+chroma",
                  with_std=False, edge_mode="knn", k=3, tau=0.7, workers=24):
    cache = load_cache(build_cache(graph, n_nodes, feat, with_std, workers))
    pos = {t: i for i, t in enumerate(cache["ids"])}
    vocab = build_vocab("top" if labels == "top" else "multi", top_k)

    splits = {}
    for name in ("training", "validation", "test"):
        ids, y = encode_labels(load_split(name), vocab, labels)
        keep = [i for i, t in enumerate(ids) if t in pos]
        splits[name] = ([ids[i] for i in keep], y[keep])

    if graph == "segment":
        train_idx = [pos[t] for t in splits["training"][0]]
        mu, sd = fit_scaler(cache["x"][train_idx])

    out = {}
    for name, (ids, y) in splits.items():
        graphs = []
        for tid, yi in zip(ids, y):
            if graph == "chord":
                x, ei, ea = cache["graphs"][pos[tid]]
                ei, ea = torch.from_numpy(ei), torch.from_numpy(ea)
                x = torch.from_numpy(x)
            else:
                arr = (cache["x"][pos[tid]] - mu) / sd
                ei, ea = gb.build_edges(arr, edge_mode, k, tau)
                x = torch.from_numpy(arr.astype(np.float32))
            label = torch.tensor(yi).unsqueeze(0) if labels == "multi" else torch.tensor([int(yi)])
            d = Data(x=x, edge_index=ei, edge_attr=ea, y=label, num_nodes=x.shape[0])
            d.track_id = tid
            graphs.append(d)
        out[name] = graphs
    return out, vocab


def make_loaders(datasets, batch_size=256):
    return {
        name: DataLoader(g, batch_size=batch_size, shuffle=(name == "training"))
        for name, g in datasets.items()
    }
