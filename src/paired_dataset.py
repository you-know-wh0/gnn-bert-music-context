"""Paired dataset linking FMA audio structure graphs with BERT textual context.

Constructs paired (graph, text, label) samples for Task 3 (Fusion) and
Task 4 (Contrastive Cross-Modal Alignment) using the official FMA splits.
"""
import ast
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from transformers import BertTokenizerFast

sys.path.insert(0, os.path.dirname(__file__))
import features_store
import graph_builder as gb
from labels import build_vocab, encode_labels, load_split

TRACKS_CSV = "data/raw/fma_medium/metadata/fma_metadata/tracks.csv"
if not os.path.exists(TRACKS_CSV):
    TRACKS_CSV = "data/raw/metadata/fma_metadata/tracks.csv"

GENRES_CSV = "data/raw/fma_medium/metadata/fma_metadata/genres.csv"
if not os.path.exists(GENRES_CSV):
    GENRES_CSV = "data/raw/metadata/fma_metadata/genres.csv"

PAIRED_CACHE_DIR = "data/processed/paired_cache"


def load_metadata_dict(tracks_csv=TRACKS_CSV):
    """Load metadata dictionary mapping 6-digit track_id -> dict of metadata fields."""
    reader = csv.reader(open(tracks_csv, encoding="utf-8"))
    h0, h1 = next(reader), next(reader)
    next(reader)  # skip 3rd header row
    col = {f"{a}.{b}": i for i, (a, b) in enumerate(zip(h0, h1))}
    meta = {}
    for row in reader:
        if not row:
            continue
        tid = row[0].strip().zfill(6)
        subset = row[col.get("set.subset", 32)].strip()
        if subset not in ("small", "medium"):
            continue
        title = row[col.get("track.title", 39)].strip()
        artist = row[col.get("artist.name", 24)].strip()
        album = row[col.get("album.title", 11)].strip()
        genre_top = row[col.get("track.genre_top", 40)].strip()
        tags = row[col.get("track.tags", 38)].strip()
        genres_all_raw = row[col.get("track.genres_all", 42)].strip()
        try:
            genres_all = ast.literal_eval(genres_all_raw or "[]")
        except Exception:
            genres_all = []

        meta[tid] = {
            "title": title,
            "artist": artist,
            "album": album,
            "genre_top": genre_top,
            "tags": tags,
            "genres_all": genres_all,
            "subset": subset,
        }
    return meta


def format_description(info):
    """Format metadata into natural contextual description for BERT."""
    title = info.get("title", "Untitled")
    artist = info.get("artist", "Unknown Artist")
    album = info.get("album", "")
    genre = info.get("genre_top", "")
    tags = info.get("tags", "")

    desc = f"A {genre} music piece" if genre else "A music track"
    if title and title != "Untitled":
        desc += f" titled '{title}'"
    if artist and artist != "Unknown Artist":
        desc += f" by {artist}"
    if album:
        desc += f" from the album '{album}'"
    desc += "."
    if tags and tags != "[]":
        desc += f" Characterized by musical styles: {tags}."
    return desc


class PairedMusicDataset(Dataset):
    """Paired dataset holding graphs, tokenized text, and labels."""

    def __init__(self, samples, tokenizer, max_length=64):
        self.samples = samples  # list of dicts: {'track_id', 'graph', 'text', 'label'}
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "track_id": item["track_id"],
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "graph": item["graph"],
            "label": item["label"],
            "text": item["text"],
        }


def paired_collate(batch):
    """Custom collator compatible with fusion_collate and PyG Batch."""
    graphs = Batch.from_data_list([b["graph"] for b in batch])
    return {
        "track_id": [b["track_id"] for b in batch],
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "x": graphs.x,
        "edge_index": graphs.edge_index,
        "edge_attr": getattr(graphs, "edge_attr", None),
        "batch": graphs.batch,
        "label": torch.stack([b["label"] for b in batch]),
        "text": [b["text"] for b in batch],
    }


def make_paired_datasets(split_limit=1500, n_nodes=60, edge_mode="knn", k=3, max_length=64):
    """Build paired datasets for training, validation, and test.

    Uses the best configuration from Task 2 (best_gat_knn: n_nodes=60, mel+chroma, with_std=True, k=3).
    """
    Path(PAIRED_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    cache_file = f"{PAIRED_CACHE_DIR}/paired_samples_{split_limit}_{n_nodes}.pt"

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    vocab = build_vocab("multi", top_k=20)

    if os.path.exists(cache_file):
        print(f"[paired_dataset] loading cached paired samples from {cache_file}")
        cached = torch.load(cache_file, weights_only=False)
        datasets = {name: PairedMusicDataset(cached[name], tokenizer, max_length)
                    for name in ("training", "validation", "test")}
        return datasets, vocab

    print(f"[paired_dataset] building paired dataset (limit={split_limit} per split)...")
    meta = load_metadata_dict()
    splits_raw = {name: load_split(name) for name in ("training", "validation", "test")}

    # Get usable track ids from features store
    usable = set(features_store.usable_ids())

    # Pre-collect feature arrays for training split to fit scaler
    train_ids = [tid for tid in splits_raw["training"] if tid in usable and tid in meta]
    if split_limit:
        train_ids = train_ids[:split_limit]

    print(f"[paired_dataset] computing scaler on {len(train_ids)} training tracks...")
    train_feats = []
    for tid in train_ids[:min(500, len(train_ids))]:
        m, c = features_store.stitch(tid)
        feat = gb.node_features_batch(m[None], c[None], n_nodes=n_nodes, feat="mel+chroma", with_std=True)[0]
        train_feats.append(feat)

    all_feats = np.concatenate(train_feats, axis=0)
    mu = all_feats.mean(0)
    sd = all_feats.std(0) + 1e-6

    cached_data = {}
    for name in ("training", "validation", "test"):
        raw_split = splits_raw[name]
        ids, y_vecs = encode_labels(raw_split, vocab, mode="multi")
        valid_indices = [i for i, tid in enumerate(ids) if tid in usable and tid in meta]
        if split_limit and name == "training":
            valid_indices = valid_indices[:split_limit]
        elif split_limit and name in ("validation", "test"):
            valid_indices = valid_indices[:min(split_limit // 2, 500)]

        samples = []
        print(f"[paired_dataset] processing {len(valid_indices)} tracks for {name} split...")
        for idx in valid_indices:
            tid = ids[idx]
            y = torch.tensor(y_vecs[idx], dtype=torch.float32)
            m, c = features_store.stitch(tid)
            raw_node_feats = gb.node_features_batch(m[None], c[None], n_nodes=n_nodes, feat="mel+chroma", with_std=True)[0]
            norm_node_feats = ((raw_node_feats - mu) / sd).astype(np.float32)

            ei, ea = gb.build_edges(norm_node_feats, mode=edge_mode, k=k, tau=0.7)
            graph = Data(
                x=torch.from_numpy(norm_node_feats),
                edge_index=ei,
                edge_attr=ea,
                y=y.unsqueeze(0),
                num_nodes=n_nodes
            )
            graph.track_id = tid

            text = format_description(meta[tid])
            samples.append({
                "track_id": tid,
                "graph": graph,
                "text": text,
                "label": y,
            })
        cached_data[name] = samples

    torch.save(cached_data, cache_file)
    print(f"[paired_dataset] saved paired cache to {cache_file}")

    datasets = {name: PairedMusicDataset(cached_data[name], tokenizer, max_length)
                for name in ("training", "validation", "test")}
    return datasets, vocab


if __name__ == "__main__":
    ds, vocab = make_paired_datasets(split_limit=200)
    print("Paired dataset built successfully!")
    print(f"Train size: {len(ds['training'])}, Val size: {len(ds['validation'])}, Test size: {len(ds['test'])}")
    sample = ds["training"][0]
    print("Sample track_id:", sample["track_id"])
    print("Sample text:", sample["text"])
    print("Sample graph x:", sample["graph"].x.shape)
    print("Sample label sum:", sample["label"].sum().item())
