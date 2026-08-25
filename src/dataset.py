import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

class MusicCapsDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=128):
        self.rows = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.rows.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(
            row['caption'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(row['label_vector'], dtype=torch.float32),
            'caption': row['caption']
        }
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
import features_store
import graph_builder as gb
from labels import build_vocab, encode_labels, load_split

CACHE_DIR = "data/processed/cache"
CHUNK = 512


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def as_time_major(block, bins):
    """(B, 6, bins, 215) -> (B, bins, 1290)."""
    b = block.shape[0]
    return block.transpose(0, 2, 1, 3).reshape(b, bins, -1)


def build_cache(graph="segment", n_nodes=30, feat="mel+chroma", with_std=False,
                chord_smooth=21):
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    tag = (f"chord_{chord_smooth}" if graph == "chord"
           else f"seg_{n_nodes}_{feat.replace('+','')}_{int(with_std)}")
    path = f"{CACHE_DIR}/{tag}.pt"
    if os.path.exists(path):
        return path

    store = features_store.load()
    ids = features_store.usable_ids()
    rows = [store["pos"][t] for t in ids]
    print(f"caching {tag}: {len(ids)} tracks", flush=True)

    if graph == "chord":
        out = []
        for block in chunks(rows, CHUNK):
            chroma = as_time_major(store["chroma"][block], features_store.CHROMA)
            seqs = gb.chord_sequence_batch(chroma, chord_smooth)
            for seq, ch in zip(seqs, chroma):
                out.append(gb.chord_graph_from_sequence(seq, ch.astype(np.float32)))
        payload = {"ids": ids, "graphs": out}
    else:
        parts = []
        for block in chunks(rows, CHUNK):
            mel = as_time_major(store["mel"][block], features_store.MELS)
            chroma = as_time_major(store["chroma"][block], features_store.CHROMA)
            parts.append(gb.node_features_batch(mel, chroma, n_nodes, feat, with_std))
        payload = {"ids": ids, "x": np.concatenate(parts)}

    torch.save(payload, path)
    print(f"saved {path}", flush=True)
    return path


def fit_scaler(x):
    flat = x.reshape(-1, x.shape[-1])
    return flat.mean(0), flat.std(0) + 1e-6


def make_datasets(labels="multi", top_k=20, graph="segment", n_nodes=30, feat="mel+chroma",
                  with_std=False, edge_mode="knn", k=3, tau=0.7, chord_smooth=21, **_):
    cache = torch.load(build_cache(graph, n_nodes, feat, with_std, chord_smooth),
                       weights_only=False)
    pos = {t: i for i, t in enumerate(cache["ids"])}
    vocab = build_vocab("top" if labels == "top" else "multi", top_k)

    splits = {}
    for name in ("training", "validation", "test"):
        ids, y = encode_labels(load_split(name), vocab, labels)
        keep = [i for i, t in enumerate(ids) if t in pos]
        splits[name] = ([ids[i] for i in keep], y[keep])

    if graph == "segment":
        mu, sd = fit_scaler(cache["x"][[pos[t] for t in splits["training"][0]]])

    out = {}
    for name, (ids, y) in splits.items():
        graphs = []
        for tid, yi in zip(ids, y):
            if graph == "chord":
                x, ei, ea = cache["graphs"][pos[tid]]
                x, ei, ea = torch.from_numpy(x), torch.from_numpy(ei), torch.from_numpy(ea)
            else:
                arr = ((cache["x"][pos[tid]] - mu) / sd).astype(np.float32)
                ei, ea = gb.build_edges(arr, edge_mode, k, tau)
                x = torch.from_numpy(arr)
            label = torch.tensor(yi).unsqueeze(0) if labels == "multi" else torch.tensor([int(yi)])
            d = Data(x=x, edge_index=ei, edge_attr=ea, y=label, num_nodes=x.shape[0])
            d.track_id = tid
            graphs.append(d)
        out[name] = graphs
    return out, vocab


def make_loaders(datasets, batch_size=256):
    return {name: DataLoader(g, batch_size=batch_size, shuffle=(name == "training"))
            for name, g in datasets.items()}
