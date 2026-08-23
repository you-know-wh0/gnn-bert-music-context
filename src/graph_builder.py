import os
import json
import numpy as np
import torch
from pathlib import Path
from torch_geometric.data import Data
from typing import Dict, Optional, Tuple, List

N_SEGMENTS     = 6
N_MEL_BINS     = 128
N_CHROMA_BINS  = 12
N_TIME_FRAMES  = 215
NODE_FEAT_DIM  = N_MEL_BINS + N_CHROMA_BINS

def load_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Person 1's saved features for one track.

    Returns
    -------
    mel    : (6, 128, 215)  normalised log-mel spectrogram segments
    chroma : (6, 12, 215)   normalised chroma STFT segments
    """
    data = np.load(npz_path)
    return data["mel"].astype(np.float32), data["chroma"].astype(np.float32)


def compute_node_features(mel: np.ndarray, chroma: np.ndarray) -> np.ndarray:
    """
    Mean-pool each segment over the time dimension, then concatenate.

    mel    : (6, 128, 215) → mean → (6, 128)
    chroma : (6, 12,  215) → mean → (6, 12)
    concat : (6, 140)

    This is exactly what Person 1's handover recommends:
        node_features_mel    = np.mean(mel_features,    axis=-1)
        node_features_chroma = np.mean(chroma_features, axis=-1)
    """
    mel_pooled    = np.mean(mel,    axis=-1)
    chroma_pooled = np.mean(chroma, axis=-1)
    return np.concatenate([mel_pooled, chroma_pooled], axis=-1)


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """
    Compute the (N, N) pairwise cosine-similarity matrix.
    features : (N, D)
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    normed = features / norms
    return normed @ normed.T


def build_edge_index(
    node_features: np.ndarray,
    sim_threshold: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the edge set for a music segment graph.

    Two types of edges (both bidirectional):
        1. Temporal adjacency  : segment i ↔ segment i+1
        2. Cosine similarity   : sim(i, j) > sim_threshold  (non-adjacent pairs)

    Returns
    -------
    edge_index : LongTensor  (2, E)   — source / destination node pairs
    edge_attr  : FloatTensor (E,)     — edge weight (1.0 for temporal, sim for others)
    """
    n = node_features.shape[0]
    sim_mat = cosine_similarity_matrix(node_features)

    src_list, dst_list, wgt_list = [], [], []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            is_adjacent = abs(i - j) == 1

            if is_adjacent:
                src_list.append(i)
                dst_list.append(j)
                wgt_list.append(1.0)
            elif sim_mat[i, j] > sim_threshold:
                src_list.append(i)
                dst_list.append(j)
                wgt_list.append(float(sim_mat[i, j]))

    if not src_list:
        for i in range(n - 1):
            src_list += [i, i + 1]
            dst_list += [i + 1, i]
            wgt_list += [1.0, 1.0]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(wgt_list,             dtype=torch.float32)
    return edge_index, edge_attr


def build_graph(
    track_id:      str,
    processed_dir: str,
    genre_idx:     int,
    sim_threshold: float = 0.7,
) -> Data:
    """
    Build a single PyG Data object for one FMA track.

    Parameters
    ----------
    track_id      : 6-digit padded string, e.g. "000002"
    processed_dir : path to Person 1's audio_features/ directory
    genre_idx     : integer genre label
    sim_threshold : cosine similarity threshold for non-temporal edges

    Returns
    -------
    graph : torch_geometric.data.Data
        .x          (6, 140)  node feature matrix
        .edge_index (2, E)    edge indices
        .edge_attr  (E,)      edge weights
        .y          (1,)      genre label
        .track_id   str       for traceability
    """
    npz_path = Path(processed_dir) / f"{track_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    mel, chroma          = load_npz(str(npz_path))
    node_features        = compute_node_features(mel, chroma)
    edge_index, edge_attr = build_edge_index(node_features, sim_threshold)

    graph = Data(
        x          = torch.tensor(node_features, dtype=torch.float32),
        edge_index = edge_index,
        edge_attr  = edge_attr,
        y          = torch.tensor([genre_idx],   dtype=torch.long),
        num_nodes  = N_SEGMENTS,
    )
    graph.track_id = track_id
    return graph


def graph_to_json_meta(graph: Data, track_id: str, genre: str) -> dict:
    """
    Serialise graph metadata to a human-readable dict (for the .json sidecar).
    """
    return {
        "track_id"   : track_id,
        "genre"      : genre,
        "num_nodes"  : int(graph.num_nodes),
        "num_edges"  : int(graph.edge_index.shape[1]),
        "node_feat_dim" : int(graph.x.shape[1]),
        "edge_index" : graph.edge_index.t().tolist(),
        "edge_weights": graph.edge_attr.tolist(),
        "label"      : int(graph.y.item()),
    }


def build_and_save_graphs(
    split_dict:    Dict[str, dict],
    genre_to_idx:  Dict[str, int],
    processed_dir: str,
    output_dir:    str,
    sim_threshold: float = 0.7,
    max_graphs:    Optional[int] = None,
    verbose:       bool = True,
) -> List[str]:
    """
    Build and cache all graphs for a split as .pt + .json files.

    Parameters
    ----------
    split_dict    : loaded JSON split (track_id → {"genre": ..., "subset": ...})
    genre_to_idx  : genre name → integer index mapping
    processed_dir : Person 1's audio_features/ directory
    output_dir    : where to save .pt and .json files
    sim_threshold : edge similarity threshold
    max_graphs    : optional cap (useful for quick smoke-tests)

    Returns
    -------
    saved_ids : list of track_ids successfully saved
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_ids = []

    items = list(split_dict.items())
    if max_graphs:
        items = items[:max_graphs]

    for i, (track_id, info) in enumerate(items):
        genre = info.get("genre", "")
        if not genre or genre not in genre_to_idx:
            continue

        pt_path   = Path(output_dir) / f"{track_id}.pt"
        json_path = Path(output_dir) / f"{track_id}.json"

        if pt_path.exists() and json_path.exists():
            saved_ids.append(track_id)
            continue

        npz_path = Path(processed_dir) / f"{track_id}.npz"
        if not npz_path.exists():
            continue

        try:
            graph = build_graph(
                track_id      = track_id,
                processed_dir = processed_dir,
                genre_idx     = genre_to_idx[genre],
                sim_threshold = sim_threshold,
            )
            torch.save(graph, str(pt_path))

            meta = graph_to_json_meta(graph, track_id, genre)
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)

            saved_ids.append(track_id)

            if verbose and (i + 1) % 500 == 0:
                print(f"  Built {i+1}/{len(items)} graphs")

        except Exception as exc:
            if verbose:
                print(f"  [WARN] {track_id}: {exc}")

    if verbose:
        print(f"Done. {len(saved_ids)} graphs saved to {output_dir}")
    return saved_ids


def build_genre_mapping(train_split: Dict[str, dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Derive genre ↔ index mapping from the training split.
    Always use training split to avoid test-set leakage.

    Returns
    -------
    genre_to_idx : dict  str → int
    idx_to_genre : dict  int → str
    """
    unique_genres = sorted(set(
        v["genre"] for v in train_split.values() if v.get("genre")
    ))
    genre_to_idx = {g: i for i, g in enumerate(unique_genres)}
    idx_to_genre = {i: g for g, i in genre_to_idx.items()}
    return genre_to_idx, idx_to_genre


if __name__ == "__main__":
    import json as _json

    SPLITS_DIR    = "data/splits"
    PROCESSED_DIR = "data/processed/audio_features"
    GRAPH_DIR     = "data/processed/graphs"

    with open(f"{SPLITS_DIR}/training.json") as f:
        train_split = _json.load(f)

    genre_to_idx, idx_to_genre = build_genre_mapping(train_split)
    print(f"Genres ({len(genre_to_idx)}): {list(genre_to_idx.keys())}")

    saved = build_and_save_graphs(
        split_dict    = train_split,
        genre_to_idx  = genre_to_idx,
        processed_dir = PROCESSED_DIR,
        output_dir    = GRAPH_DIR,
        max_graphs    = 5,
        verbose       = True,
    )
    print(f"Saved {len(saved)} graphs: {saved}")