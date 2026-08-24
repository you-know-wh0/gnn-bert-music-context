import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORD_NAMES = [f"{p}{q}" for q in ("maj", "min") for p in PITCHES]


def chord_templates():
    t = np.zeros((24, 12), dtype=np.float32)
    for r in range(12):
        t[r, [r, (r + 4) % 12, (r + 7) % 12]] = 1.0
        t[12 + r, [r, (r + 3) % 12, (r + 7) % 12]] = 1.0
    return t / np.linalg.norm(t, axis=1, keepdims=True)


TEMPLATES = chord_templates()


def stitch(npz_path):
    d = np.load(npz_path)
    mel = np.concatenate(list(d["mel"]), axis=-1).astype(np.float32)
    chroma = np.concatenate(list(d["chroma"]), axis=-1).astype(np.float32)
    return mel, chroma


def pool_windows(mat, n_nodes, with_std=False):
    edges = np.linspace(0, mat.shape[1], n_nodes + 1).astype(int)
    parts = [mat[:, a:b] for a, b in zip(edges[:-1], edges[1:])]
    means = np.stack([p.mean(axis=1) for p in parts])
    if not with_std:
        return means
    stds = np.stack([p.std(axis=1) for p in parts])
    return np.concatenate([means, stds], axis=1)


def node_features(mel, chroma, n_nodes=30, feat="mel+chroma", with_std=False):
    parts = []
    if "mel" in feat:
        parts.append(pool_windows(mel, n_nodes, with_std))
    if "chroma" in feat:
        parts.append(pool_windows(chroma, n_nodes, with_std))
    return np.concatenate(parts, axis=1).astype(np.float32)


def cosine_sim(x):
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    return xn @ xn.T


def build_edges(x, mode="knn", k=3, tau=0.7):
    n = x.shape[0]
    src = np.concatenate([np.arange(n - 1), np.arange(1, n)])
    dst = np.concatenate([np.arange(1, n), np.arange(n - 1)])
    w = np.ones(len(src), dtype=np.float32)

    if mode != "temporal":
        sim = cosine_sim(x)
        far = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) > 1
        if mode == "knn":
            masked = np.where(far, sim, -np.inf)
            kk = min(k, max(int(far.sum(axis=1).min()), 0))
            keep = np.zeros_like(far)
            if kk > 0:
                idx = np.argsort(-masked, axis=1)[:, :kk]
                keep[np.arange(n)[:, None], idx] = True
                keep &= far
        elif mode == "threshold":
            keep = far & (sim > tau)
        elif mode == "full":
            keep = far
        else:
            raise ValueError(mode)
        s2, d2 = np.nonzero(keep)
        src = np.concatenate([src, s2])
        dst = np.concatenate([dst, d2])
        w = np.concatenate([w, sim[s2, d2].astype(np.float32)])

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return edge_index, torch.tensor(w, dtype=torch.float32)


def chord_sequence(chroma):
    c = chroma - chroma.min(axis=0, keepdims=True)
    c = c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-8)
    return np.argmax(TEMPLATES @ c, axis=0)


def chord_graph(chroma, min_frames=3):
    seq = chord_sequence(chroma)
    counts = np.bincount(seq, minlength=24)
    nodes = np.nonzero(counts >= min_frames)[0]
    if len(nodes) < 2:
        nodes = np.unique(seq)
    remap = {c: i for i, c in enumerate(nodes)}

    onehot = np.eye(24, dtype=np.float32)[nodes]
    mean_chroma = np.stack([chroma[:, seq == c].mean(axis=1) for c in nodes])
    dur = (counts[nodes] / counts.sum()).astype(np.float32)[:, None]
    x = np.concatenate([onehot, mean_chroma, dur], axis=1).astype(np.float32)

    trans = {}
    prev = seq[0]
    for cur in seq[1:]:
        if cur != prev:
            if prev in remap and cur in remap:
                key = (remap[prev], remap[cur])
                trans[key] = trans.get(key, 0) + 1
            prev = cur
    if not trans:
        trans = {(i, (i + 1) % len(nodes)): 1 for i in range(len(nodes))}

    pairs = np.array(list(trans.keys())).T
    w = np.array(list(trans.values()), dtype=np.float32)
    w = w / w.max()
    return x, torch.tensor(pairs, dtype=torch.long), torch.tensor(w)


def build_graph(npz_path, y, graph="segment", n_nodes=30, feat="mel+chroma",
                with_std=False, edge_mode="knn", k=3, tau=0.7):
    mel, chroma = stitch(npz_path)
    if graph == "chord":
        x, edge_index, edge_attr = chord_graph(chroma)
    else:
        x = node_features(mel, chroma, n_nodes, feat, with_std)
        edge_index, edge_attr = build_edges(x, edge_mode, k, tau)
    y = torch.tensor(y).unsqueeze(0) if np.ndim(y) else torch.tensor([y])
    return Data(x=torch.from_numpy(x), edge_index=edge_index, edge_attr=edge_attr,
                y=y, num_nodes=x.shape[0])


def graph_to_json(data, track_id, labels):
    return {
        "track_id": track_id,
        "labels": labels,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.shape[1]),
        "node_feat_dim": int(data.x.shape[1]),
        "edge_index": data.edge_index.t().tolist(),
        "edge_weights": [round(float(v), 4) for v in data.edge_attr],
    }
