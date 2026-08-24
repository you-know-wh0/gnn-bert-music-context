import os
import sys

import numpy as np
import torch
from scipy.ndimage import median_filter
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(__file__))
import features_store

SMOOTH = 21  # frames; 43 frames/s at hop 512 -> ~0.5 s
PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORD_NAMES = [f"{p}{q}" for q in ("maj", "min") for p in PITCHES]


def chord_templates():
    t = np.zeros((24, 12), dtype=np.float32)
    for r in range(12):
        t[r, [r, (r + 4) % 12, (r + 7) % 12]] = 1.0
        t[12 + r, [r, (r + 3) % 12, (r + 7) % 12]] = 1.0
    return t / np.linalg.norm(t, axis=1, keepdims=True)


TEMPLATES = chord_templates()


def stitch(track_id):
    return features_store.stitch(track_id)


def pool_batch(mat, n_nodes, with_std=False):
    """Window-pool a batch (B, D, T) into (B, n_nodes, D) or (B, n_nodes, 2D)."""
    mat = mat.astype(np.float32)
    edges = np.linspace(0, mat.shape[2], n_nodes + 1).astype(int)
    starts, counts = edges[:-1], np.diff(edges).astype(np.float32)
    means = np.add.reduceat(mat, starts, axis=2) / counts
    if not with_std:
        return means.transpose(0, 2, 1)
    sq = np.add.reduceat(mat ** 2, starts, axis=2) / counts
    stds = np.sqrt(np.maximum(sq - means ** 2, 0.0))
    return np.concatenate([means.transpose(0, 2, 1), stds.transpose(0, 2, 1)], axis=2)


def node_features_batch(mel, chroma, n_nodes=30, feat="mel+chroma", with_std=False):
    parts = []
    if "mel" in feat:
        parts.append(pool_batch(mel, n_nodes, with_std))
    if "chroma" in feat:
        parts.append(pool_batch(chroma, n_nodes, with_std))
    return np.concatenate(parts, axis=2).astype(np.float32)


def chord_sequence_batch(chroma, smooth=SMOOTH):
    """Chord index per frame for a batch (B, 12, T) -> (B, T)."""
    c = chroma.astype(np.float32)
    c = c - c.min(axis=1, keepdims=True)
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)
    seq = np.argmax(np.einsum("kd,bdt->bkt", TEMPLATES, c), axis=1)
    return median_filter(seq, size=(1, smooth), mode="nearest") if smooth > 1 else seq


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


def chord_sequence(chroma, smooth=SMOOTH):
    c = chroma - chroma.min(axis=0, keepdims=True)
    c = c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-8)
    seq = np.argmax(TEMPLATES @ c, axis=0)
    return median_filter(seq, size=smooth, mode="nearest") if smooth > 1 else seq


def chord_graph_from_sequence(seq, chroma, min_frames=3):
    """Assemble a chord-transition graph; returns numpy (x, edge_index, edge_weight)."""
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

    pairs = np.array(list(trans.keys()), dtype=np.int64).T
    w = np.array(list(trans.values()), dtype=np.float32)
    return x, pairs, w / w.max()


def chord_graph(chroma, min_frames=3, smooth=SMOOTH):
    x, ei, ew = chord_graph_from_sequence(chord_sequence(chroma, smooth), chroma, min_frames)
    return x, torch.from_numpy(ei), torch.from_numpy(ew)


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
